import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str = "atlas.db", knowledge_db_path: str = "knowledge.db"):
        self.db_path = db_path
        self.knowledge_db_path = knowledge_db_path
        self.init_database()
        self.init_knowledge_database()
    
    def init_database(self):
        """Initialize the main strategies database with required schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    parent_id INTEGER,
                    version INTEGER DEFAULT 1,
                    code TEXT NOT NULL,
                    motivation TEXT,
                    metrics TEXT,  -- JSON string
                    analysis TEXT,
                    status TEXT DEFAULT 'pending',
                    FOREIGN KEY (parent_id) REFERENCES strategies (id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_strategies_timestamp 
                ON strategies (timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_strategies_parent_id 
                ON strategies (parent_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_strategies_status 
                ON strategies (status)
            """)
    
    def init_knowledge_database(self):
        """Initialize the knowledge base database for embeddings."""
        with sqlite3.connect(self.knowledge_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filepath TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_filepath 
                ON knowledge (filepath)
            """)
    
    def store_strategy(self, code: str, motivation: str = None, 
                      parent_id: int = None, metrics: Dict = None, 
                      analysis: str = None, status: str = "pending") -> int:
        """Store a new strategy and return its ID."""
        timestamp = datetime.now().isoformat()
        metrics_json = json.dumps(metrics) if metrics else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO strategies 
                (timestamp, parent_id, code, motivation, metrics, analysis, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, parent_id, code, motivation, metrics_json, analysis, status))
            return cursor.lastrowid
    
    def get_top_k(self, k: int = 5, metric: str = "test_sharpe") -> List[Dict]:
        """Return top K strategies by specified metric (default: test Sharpe ratio)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM strategies 
                WHERE metrics IS NOT NULL AND status = 'completed'
                ORDER BY json_extract(metrics, '$.{}') DESC
                LIMIT ?
            """.format(metric), (k,))
            
            results = []
            for row in cursor.fetchall():
                strategy = dict(row)
                if strategy['metrics']:
                    strategy['metrics'] = json.loads(strategy['metrics'])
                results.append(strategy)
            return results
    
    def get_children(self, parent_id: int) -> List[Dict]:
        """Return all descendant strategies for lineage visualization."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM strategies 
                WHERE parent_id = ?
                ORDER BY timestamp ASC
            """, (parent_id,))
            
            results = []
            for row in cursor.fetchall():
                strategy = dict(row)
                if strategy['metrics']:
                    strategy['metrics'] = json.loads(strategy['metrics'])
                results.append(strategy)
            return results
    
    def get_strategy(self, strategy_id: int) -> Optional[Dict]:
        """Get a specific strategy by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM strategies WHERE id = ?
            """, (strategy_id,))
            
            row = cursor.fetchone()
            if row:
                strategy = dict(row)
                if strategy['metrics']:
                    strategy['metrics'] = json.loads(strategy['metrics'])
                return strategy
            return None
    
    def update_strategy_metrics(self, strategy_id: int, metrics: Dict):
        """Update the metrics for a specific strategy."""
        metrics_json = json.dumps(metrics)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE strategies 
                SET metrics = ? 
                WHERE id = ?
            """, (metrics_json, strategy_id))
    
    def update_strategy_analysis(self, strategy_id: int, analysis: str):
        """Update the analysis for a specific strategy."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE strategies 
                SET analysis = ? 
                WHERE id = ?
            """, (analysis, strategy_id))
    
    def update_strategy_status(self, strategy_id: int, status: str):
        """Update the status of a specific strategy."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE strategies 
                SET status = ? 
                WHERE id = ?
            """, (status, strategy_id))
    
    def store_knowledge(self, filepath: str, content: str, embedding: np.ndarray = None) -> int:
        """Store knowledge base content with optional embedding."""
        created_at = datetime.now().isoformat()
        embedding_blob = embedding.tobytes() if embedding is not None else None
        
        with sqlite3.connect(self.knowledge_db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO knowledge (filepath, content, embedding, created_at)
                VALUES (?, ?, ?, ?)
            """, (filepath, content, embedding_blob, created_at))
            return cursor.lastrowid
    
    def get_all_knowledge(self) -> List[Dict]:
        """Get all knowledge base entries."""
        with sqlite3.connect(self.knowledge_db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM knowledge ORDER BY created_at")
            
            results = []
            for row in cursor.fetchall():
                knowledge = dict(row)
                if knowledge['embedding']:
                    knowledge['embedding'] = np.frombuffer(knowledge['embedding'], dtype=np.float32)
                results.append(knowledge)
            return results
    
    def clear_knowledge(self):
        """Clear all knowledge base entries (useful for rebuilding embeddings)."""
        with sqlite3.connect(self.knowledge_db_path) as conn:
            conn.execute("DELETE FROM knowledge")