import pytest
import sqlite3
import tempfile
import os
import json
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database import DatabaseManager

class TestDatabaseManager:
    
    def setup_method(self):
        # Use temporary databases for testing
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_atlas.db")
        self.knowledge_db_path = os.path.join(self.temp_dir, "test_knowledge.db")
        
        self.db = DatabaseManager(self.db_path, self.knowledge_db_path)
    
    def teardown_method(self):
        # Clean up temporary files
        try:
            os.unlink(self.db_path)
            os.unlink(self.knowledge_db_path)
            os.rmdir(self.temp_dir)
        except:
            pass
    
    def test_database_initialization(self):
        """Test that databases are properly initialized."""
        # Check that main database tables exist
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='strategies'
            """)
            assert cursor.fetchone() is not None
        
        # Check that knowledge database tables exist
        with sqlite3.connect(self.knowledge_db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='knowledge'
            """)
            assert cursor.fetchone() is not None
    
    def test_store_strategy(self):
        """Test storing a strategy."""
        code = "signals = pd.Series(1.0, index=price.index)"
        motivation = "Test strategy"
        metrics = {"sharpe": 1.5, "return": 0.15}
        
        strategy_id = self.db.store_strategy(
            code=code,
            motivation=motivation,
            metrics=metrics,
            status="completed"
        )
        
        assert isinstance(strategy_id, int)
        assert strategy_id > 0
        
        # Verify storage
        strategy = self.db.get_strategy(strategy_id)
        assert strategy is not None
        assert strategy['code'] == code
        assert strategy['motivation'] == motivation
        assert strategy['metrics'] == metrics
        assert strategy['status'] == "completed"
    
    def test_get_top_k_strategies(self):
        """Test retrieving top K strategies."""
        # Store multiple strategies with different Sharpe ratios
        strategies_data = [
            ("strategy1", {"test_sharpe": 1.5}),
            ("strategy2", {"test_sharpe": 2.0}),
            ("strategy3", {"test_sharpe": 1.0}),
            ("strategy4", {"test_sharpe": 1.8}),
        ]
        
        stored_ids = []
        for code, metrics in strategies_data:
            strategy_id = self.db.store_strategy(
                code=code,
                metrics=metrics,
                status="completed"
            )
            stored_ids.append(strategy_id)
        
        # Get top 3
        top_strategies = self.db.get_top_k(k=3)
        
        assert len(top_strategies) == 3
        
        # Should be ordered by test_sharpe descending
        sharpe_ratios = [s['metrics']['test_sharpe'] for s in top_strategies]
        assert sharpe_ratios == sorted(sharpe_ratios, reverse=True)
        assert sharpe_ratios[0] == 2.0  # Best strategy first
    
    def test_get_children(self):
        """Test retrieving child strategies."""
        # Create parent strategy
        parent_id = self.db.store_strategy(
            code="parent_code",
            motivation="parent strategy"
        )
        
        # Create child strategies
        child_ids = []
        for i in range(3):
            child_id = self.db.store_strategy(
                code=f"child_code_{i}",
                motivation=f"child strategy {i}",
                parent_id=parent_id
            )
            child_ids.append(child_id)
        
        # Get children
        children = self.db.get_children(parent_id)
        
        assert len(children) == 3
        for child in children:
            assert child['parent_id'] == parent_id
            assert child['id'] in child_ids
    
    def test_update_strategy_metrics(self):
        """Test updating strategy metrics."""
        strategy_id = self.db.store_strategy(code="test", motivation="test")
        
        new_metrics = {"sharpe": 2.5, "return": 0.25, "drawdown": -0.1}
        self.db.update_strategy_metrics(strategy_id, new_metrics)
        
        updated_strategy = self.db.get_strategy(strategy_id)
        assert updated_strategy['metrics'] == new_metrics
    
    def test_update_strategy_analysis(self):
        """Test updating strategy analysis."""
        strategy_id = self.db.store_strategy(code="test", motivation="test")
        
        analysis = "# Analysis Report\nThis is a test analysis."
        self.db.update_strategy_analysis(strategy_id, analysis)
        
        updated_strategy = self.db.get_strategy(strategy_id)
        assert updated_strategy['analysis'] == analysis
    
    def test_update_strategy_status(self):
        """Test updating strategy status."""
        strategy_id = self.db.store_strategy(code="test", motivation="test")
        
        self.db.update_strategy_status(strategy_id, "failed")
        
        updated_strategy = self.db.get_strategy(strategy_id)
        assert updated_strategy['status'] == "failed"
    
    def test_store_knowledge(self):
        """Test storing knowledge entries."""
        filepath = "test_knowledge.md"
        content = "This is test knowledge content."
        embedding = np.random.randn(384).astype(np.float32)
        
        knowledge_id = self.db.store_knowledge(filepath, content, embedding)
        
        assert isinstance(knowledge_id, int)
        assert knowledge_id > 0
        
        # Verify storage
        knowledge_entries = self.db.get_all_knowledge()
        assert len(knowledge_entries) == 1
        
        entry = knowledge_entries[0]
        assert entry['filepath'] == filepath
        assert entry['content'] == content
        assert np.array_equal(entry['embedding'], embedding)
    
    def test_get_all_knowledge(self):
        """Test retrieving all knowledge entries."""
        # Store multiple knowledge entries
        entries_data = [
            ("file1.md", "Content 1"),
            ("file2.md", "Content 2"),
            ("file3.md", "Content 3"),
        ]
        
        for filepath, content in entries_data:
            embedding = np.random.randn(384).astype(np.float32)
            self.db.store_knowledge(filepath, content, embedding)
        
        all_knowledge = self.db.get_all_knowledge()
        
        assert len(all_knowledge) == 3
        
        stored_filepaths = [entry['filepath'] for entry in all_knowledge]
        expected_filepaths = [entry[0] for entry in entries_data]
        
        for expected in expected_filepaths:
            assert expected in stored_filepaths
    
    def test_clear_knowledge(self):
        """Test clearing knowledge database."""
        # Store some knowledge
        self.db.store_knowledge("test.md", "test content", np.random.randn(384).astype(np.float32))
        
        # Verify it's stored
        assert len(self.db.get_all_knowledge()) == 1
        
        # Clear and verify
        self.db.clear_knowledge()
        assert len(self.db.get_all_knowledge()) == 0
    
    def test_json_metrics_serialization(self):
        """Test that complex metrics are properly serialized/deserialized."""
        complex_metrics = {
            "sharpe": 1.5,
            "returns": [0.1, 0.2, -0.05, 0.15],
            "drawdowns": {"max": -0.12, "avg": -0.03},
            "boolean_flag": True,
            "null_value": None
        }
        
        strategy_id = self.db.store_strategy(
            code="test",
            metrics=complex_metrics,
            status="completed"
        )
        
        retrieved_strategy = self.db.get_strategy(strategy_id)
        assert retrieved_strategy['metrics'] == complex_metrics
    
    def test_strategy_not_found(self):
        """Test retrieving non-existent strategy."""
        strategy = self.db.get_strategy(99999)
        assert strategy is None
    
    def test_empty_top_k(self):
        """Test getting top K when no completed strategies exist."""
        # Store a pending strategy (shouldn't be included in top K)
        self.db.store_strategy(code="test", status="pending")
        
        top_strategies = self.db.get_top_k(k=5)
        assert len(top_strategies) == 0
    
    def test_parent_child_lineage(self):
        """Test complete parent-child lineage tracking."""
        # Create grandparent
        grandparent_id = self.db.store_strategy(code="grandparent", motivation="gen 0")
        
        # Create parent
        parent_id = self.db.store_strategy(
            code="parent", 
            motivation="gen 1",
            parent_id=grandparent_id
        )
        
        # Create children
        child1_id = self.db.store_strategy(
            code="child1", 
            motivation="gen 2a",
            parent_id=parent_id
        )
        
        child2_id = self.db.store_strategy(
            code="child2", 
            motivation="gen 2b", 
            parent_id=parent_id
        )
        
        # Test lineage queries
        parent_children = self.db.get_children(parent_id)
        assert len(parent_children) == 2
        assert {c['id'] for c in parent_children} == {child1_id, child2_id}
        
        grandparent_children = self.db.get_children(grandparent_id)
        assert len(grandparent_children) == 1
        assert grandparent_children[0]['id'] == parent_id

if __name__ == "__main__":
    pytest.main([__file__])