import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .database import DatabaseManager

class KnowledgeBase:
    def __init__(self, knowledge_dir: str = "knowledge", 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.knowledge_dir = Path(knowledge_dir)
        self.model_name = model_name
        self.model = None
        self.db = DatabaseManager()
        
        # Ensure knowledge directory exists
        self.knowledge_dir.mkdir(exist_ok=True)
        
        # Create sample knowledge files if directory is empty
        if not any(self.knowledge_dir.glob("*.md")):
            self._create_sample_knowledge()
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
    
    def _create_sample_knowledge(self):
        """Create sample knowledge files for the MVP."""
        sample_files = {
            "001_market_structure.md": """# Market Structure and Trading

## Bid-Ask Spreads
The bid-ask spread represents the difference between the highest price buyers are willing to pay (bid) and the lowest price sellers are willing to accept (ask). In cryptocurrency markets, spreads typically range from 0.01% to 0.1% for major pairs like BTC-USD.

## Market Impact and Slippage
Large orders can move market prices unfavorably. Slippage of 10-20 basis points is common for moderate position sizes in BTC markets. Position sizes should generally not exceed 5% of average daily volume to minimize market impact.

## Volatility Clustering
Financial markets exhibit volatility clustering - periods of high volatility tend to be followed by high volatility periods. This is particularly pronounced in cryptocurrency markets where volatility can persist for weeks.
""",
            
            "002_risk_management.md": """# Risk Management Principles

## Position Sizing
Effective position sizing is crucial for long-term survival. The Kelly Criterion suggests optimal position size based on win rate and average win/loss ratio. For trading strategies, position sizes typically should not exceed 2-5% of portfolio value per trade.

## Leverage Constraints
Excessive leverage amplifies both gains and losses. Regulatory frameworks typically limit retail leverage to 2:1 for cryptocurrencies. Even professional traders rarely exceed 3:1 leverage for systematic strategies due to margin call risks.

## Maximum Drawdown
Drawdown measures the peak-to-trough decline in portfolio value. Strategies with maximum drawdowns exceeding 20% are generally considered high-risk. Institutional investors typically seek strategies with drawdowns below 15%.

## Sharpe Ratio Thresholds
The Sharpe ratio measures risk-adjusted returns. Values above 1.0 are considered good, above 1.5 are very good, and above 2.0 are exceptional. However, Sharpe ratios above 3.0 should be viewed with skepticism as they may indicate overfitting.
""",

            "003_backtesting_practices.md": """# Backtesting Best Practices

## Walk-Forward Validation
Walk-forward validation tests strategy robustness by progressively training on historical data and testing on future periods. This mimics real-world deployment better than simple train/test splits.

## Transaction Costs
Realistic transaction cost modeling is essential. Include bid-ask spreads, brokerage fees, and market impact. For crypto markets, total costs typically range from 0.1% to 0.2% per round-trip trade.

## Data Snooping Bias
Testing multiple strategies on the same data set leads to selection bias. The probability of finding a profitable strategy by chance increases with the number of strategies tested. Use separate validation periods and statistical significance tests.

## Survivorship Bias
Historical data may not include delisted or failed assets. This creates an upward bias in backtested returns. Always consider the full universe of assets that existed during the testing period.
""",

            "004_technical_indicators.md": """# Technical Analysis and Indicators

## Moving Averages
Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) are trend-following indicators. Crossover strategies using short and long-term averages (e.g., 50/200 day) are common but tend to lag market turns.

## Mean Reversion Indicators
RSI, Bollinger Bands, and other mean reversion indicators assume prices will return to historical averages. These work well in ranging markets but can generate false signals during strong trends.

## Momentum Indicators
MACD, Rate of Change, and momentum oscillators measure the speed of price changes. They can help identify trend acceleration or deceleration but are prone to whipsaws in volatile markets.

## Volume Analysis
Volume confirms price movements. Rising prices on increasing volume suggest strong trends, while rising prices on decreasing volume may indicate weakening momentum.
"""
        }
        
        for filename, content in sample_files.items():
            filepath = self.knowledge_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"Created {len(sample_files)} sample knowledge files in {self.knowledge_dir}")
    
    def load_knowledge_files(self) -> List[Dict]:
        """Load all markdown files from knowledge directory."""
        knowledge_entries = []
        
        for filepath in self.knowledge_dir.glob("*.md"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split content into chunks by headers or paragraphs
                chunks = self._split_content(content)
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Skip empty chunks
                        entry = {
                            'filepath': str(filepath),
                            'chunk_id': i,
                            'content': chunk.strip()
                        }
                        knowledge_entries.append(entry)
            
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        return knowledge_entries
    
    def _split_content(self, content: str, max_chunk_size: int = 500) -> List[str]:
        """Split content into manageable chunks for embedding."""
        # Split by double newlines (paragraphs) first
        paragraphs = content.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, save current chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_embeddings(self) -> int:
        """Generate embeddings for all knowledge files and store in database."""
        self._load_model()
        
        # Clear existing knowledge
        self.db.clear_knowledge()
        
        # Load knowledge files
        knowledge_entries = self.load_knowledge_files()
        
        if not knowledge_entries:
            print("No knowledge files found")
            return 0
        
        print(f"Generating embeddings for {len(knowledge_entries)} knowledge chunks...")
        
        # Generate embeddings in batches
        batch_size = 32
        stored_count = 0
        
        for i in range(0, len(knowledge_entries), batch_size):
            batch = knowledge_entries[i:i + batch_size]
            
            # Extract content for embedding
            texts = [entry['content'] for entry in batch]
            
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Store in database
            for entry, embedding in zip(batch, embeddings):
                try:
                    self.db.store_knowledge(
                        filepath=f"{entry['filepath']}#{entry['chunk_id']}",
                        content=entry['content'],
                        embedding=embedding
                    )
                    stored_count += 1
                except Exception as e:
                    print(f"Error storing knowledge entry: {e}")
        
        print(f"Stored {stored_count} knowledge entries with embeddings")
        return stored_count
    
    def retrieve_top_n(self, query: str, n: int = 3) -> List[Dict]:
        """Retrieve top N most relevant knowledge snippets for a query."""
        self._load_model()
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Get all knowledge entries
        knowledge_entries = self.db.get_all_knowledge()
        
        if not knowledge_entries:
            print("No knowledge entries found in database")
            return []
        
        # Calculate similarities
        similarities = []
        for entry in knowledge_entries:
            if entry['embedding'] is not None:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    entry['embedding'].reshape(1, -1)
                )[0, 0]
                similarities.append((similarity, entry))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        top_entries = []
        for similarity, entry in similarities[:n]:
            top_entries.append({
                'content': entry['content'],
                'filepath': entry['filepath'],
                'similarity': similarity
            })
        
        return top_entries
    
    def search_knowledge(self, query: str, n: int = 3) -> str:
        """Search knowledge base and return formatted citations."""
        relevant_entries = self.retrieve_top_n(query, n)
        
        if not relevant_entries:
            return "No relevant knowledge found."
        
        formatted_results = []
        for i, entry in enumerate(relevant_entries, 1):
            filepath = entry['filepath'].split('#')[0]  # Remove chunk ID
            filename = Path(filepath).name
            
            formatted_results.append(
                f"[{i}] From {filename} (similarity: {entry['similarity']:.3f}):\n"
                f"{entry['content'][:300]}{'...' if len(entry['content']) > 300 else ''}\n"
            )
        
        return "\n".join(formatted_results)
    
    def update_knowledge_base(self):
        """Reload and re-embed all knowledge files."""
        print("Updating knowledge base...")
        count = self.generate_embeddings()
        print(f"Knowledge base updated with {count} entries")
        return count