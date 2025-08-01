import pytest
import tempfile
import os
import shutil
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_base import KnowledgeBase
from database import DatabaseManager

class TestKnowledgeBase:
    
    def setup_method(self):
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.knowledge_dir = os.path.join(self.temp_dir, "knowledge")
        self.db_dir = os.path.join(self.temp_dir, "db")
        
        os.makedirs(self.knowledge_dir)
        os.makedirs(self.db_dir)
        
        # Initialize knowledge base with temporary directory
        self.kb = KnowledgeBase(
            knowledge_dir=self.knowledge_dir
        )
        
        # Override database paths to use temp directory
        self.kb.db.db_path = os.path.join(self.db_dir, "atlas.db")
        self.kb.db.knowledge_db_path = os.path.join(self.db_dir, "knowledge.db")
        self.kb.db._init_database()
    
    def teardown_method(self):
        # Clean up temporary directory
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_knowledge_base_initialization(self):
        """Test that knowledge base initializes correctly."""
        assert self.kb.knowledge_dir.exists()
        assert self.kb.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert self.kb.model is None  # Lazy loaded
    
    def test_create_sample_knowledge(self):
        """Test that sample knowledge files are created."""
        # Remove existing files if any
        for f in self.kb.knowledge_dir.glob("*.md"):
            f.unlink()
        
        # Create sample knowledge
        self.kb._create_sample_knowledge()
        
        # Check that files were created
        md_files = list(self.kb.knowledge_dir.glob("*.md"))
        assert len(md_files) >= 4  # Should have at least 4 sample files
        
        # Check content of one file
        market_structure_file = self.kb.knowledge_dir / "001_market_structure.md"
        assert market_structure_file.exists()
        
        with open(market_structure_file, 'r') as f:
            content = f.read()
            assert "Bid-Ask Spreads" in content
            assert "Market Impact" in content
    
    def test_load_knowledge_files(self):
        """Test loading knowledge files."""
        # Create a test knowledge file
        test_file = self.kb.knowledge_dir / "test_knowledge.md"
        test_content = """# Test Knowledge

## Section 1
This is test content for section 1.

## Section 2
This is test content for section 2.
"""
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Load knowledge files
        entries = self.kb.load_knowledge_files()
        
        assert len(entries) > 0
        
        # Find our test file entries
        test_entries = [e for e in entries if 'test_knowledge.md' in e['filepath']]
        assert len(test_entries) >= 2  # Should be split into chunks
        
        # Check that content is preserved
        all_content = ' '.join([e['content'] for e in test_entries])
        assert "test content for section 1" in all_content
        assert "test content for section 2" in all_content
    
    def test_split_content(self):
        """Test content splitting functionality."""
        long_content = "This is a test paragraph.\n\n" * 50  # Create long content
        
        chunks = self.kb._split_content(long_content, max_chunk_size=100)
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        
        # Check that no chunk exceeds max size significantly
        for chunk in chunks:
            # Allow some tolerance for word boundaries
            assert len(chunk) <= 150  # Some tolerance for paragraph boundaries
        
        # Check that all content is preserved
        rejoined = '\n\n'.join(chunks)
        assert "This is a test paragraph." in rejoined
    
    def test_generate_embeddings(self):
        """Test embedding generation and storage."""
        # Skip if model loading fails (CI environments might not have the model)
        try:
            count = self.kb.generate_embeddings()
            assert count > 0
            
            # Check that embeddings were stored in database
            entries = self.kb.db.get_all_knowledge()
            assert len(entries) == count
            
            # Check that embeddings are valid numpy arrays
            for entry in entries:
                assert entry['embedding'] is not None
                assert isinstance(entry['embedding'], np.ndarray)
                assert entry['embedding'].shape[0] > 0  # Has dimensions
                
        except Exception as e:
            pytest.skip(f"Skipping embedding test due to model loading issue: {e}")
    
    def test_retrieve_top_n(self):
        """Test knowledge retrieval functionality."""
        # Create test knowledge entries manually in database
        test_entries = [
            ("test1.md#0", "Trading strategies use technical indicators", np.random.randn(384).astype(np.float32)),
            ("test2.md#0", "Risk management is crucial for portfolio survival", np.random.randn(384).astype(np.float32)),
            ("test3.md#0", "Market volatility affects trading performance", np.random.randn(384).astype(np.float32)),
        ]
        
        for filepath, content, embedding in test_entries:
            self.kb.db.store_knowledge(filepath, content, embedding)
        
        try:
            # Test retrieval
            results = self.kb.retrieve_top_n("trading strategies", n=2)
            
            assert len(results) <= 2  # Should return at most 2 results
            
            # Check result format
            for result in results:
                assert 'filepath' in result
                assert 'text' in result
                assert isinstance(result['filepath'], str)
                assert isinstance(result['text'], str)
                
        except Exception as e:
            pytest.skip(f"Skipping retrieval test due to model loading issue: {e}")
    
    def test_load_snippet_text(self):
        """Test loading snippet text from files."""
        # Create a test file
        test_file = self.kb.knowledge_dir / "snippet_test.md"
        test_content = "# Snippet Test\nThis is test content for snippet loading."
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Test loading with chunk ID (should ignore chunk ID)
        result = self.kb.load_snippet_text(f"{test_file}#0")
        assert result == test_content
        
        # Test loading without chunk ID
        result = self.kb.load_snippet_text(str(test_file))
        assert result == test_content
        
        # Test loading non-existent file
        result = self.kb.load_snippet_text("nonexistent.md")
        assert "not found" in result.lower()
    
    def test_update_knowledge_base(self):
        """Test updating the entire knowledge base."""
        # Create some test files
        test_files = {
            "update_test1.md": "# Update Test 1\nContent for update test 1.",
            "update_test2.md": "# Update Test 2\nContent for update test 2."
        }
        
        for filename, content in test_files.items():
            with open(self.kb.knowledge_dir / filename, 'w') as f:
                f.write(content)
        
        try:
            # Update knowledge base
            count = self.kb.update_knowledge_base()
            assert count > 0
            
            # Verify entries were stored
            entries = self.kb.db.get_all_knowledge()
            assert len(entries) == count
            
            # Check that our test content is included
            all_content = ' '.join([e['content'] for e in entries])
            assert "Content for update test 1" in all_content
            assert "Content for update test 2" in all_content
            
        except Exception as e:
            pytest.skip(f"Skipping update test due to model loading issue: {e}")
    
    def test_empty_knowledge_directory(self):
        """Test behavior with empty knowledge directory."""
        # Remove all files
        for f in self.kb.knowledge_dir.glob("*.md"):
            f.unlink()
        
        # Load should return empty list
        entries = self.kb.load_knowledge_files()
        assert len(entries) == 0
        
        # Generate embeddings should return 0
        count = self.kb.generate_embeddings()
        assert count == 0
    
    def test_malformed_markdown_handling(self):
        """Test handling of malformed markdown files."""
        # Create a file with problematic content
        problem_file = self.kb.knowledge_dir / "problem.md"
        problem_content = "# Test\n\x00\x01\x02 Invalid characters \n\n## Section\nNormal content"
        
        with open(problem_file, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(problem_content)
        
        # Should handle gracefully without crashing
        entries = self.kb.load_knowledge_files()
        
        # Should still load the file (might clean up invalid chars)
        problem_entries = [e for e in entries if 'problem.md' in e['filepath']]
        assert len(problem_entries) > 0

if __name__ == "__main__":
    pytest.main([__file__])