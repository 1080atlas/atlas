import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from typing import Optional

class DataLoader:
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.symbol = "BTC-USD"
    
    def get_cache_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Generate cache file path based on symbol and date range."""
        return self.cache_dir / f"{symbol}_{start_date}_{end_date}.pkl"
    
    def load_from_cache(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and recent."""
        cache_path = self.get_cache_path(symbol, start_date, end_date)
        
        if cache_path.exists():
            # Check if cache is less than 1 day old
            cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if cache_age < timedelta(days=1):
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Error loading cache: {e}")
                    cache_path.unlink()  # Remove corrupted cache
        return None
    
    def save_to_cache(self, data: pd.DataFrame, symbol: str, start_date: str, end_date: str):
        """Save data to cache."""
        cache_path = self.get_cache_path(symbol, start_date, end_date)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving to cache: {e}")
    
    def fetch_btc_data(self, start_date: str = "2018-01-01", 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch BTC-USD daily candles with local caching.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            DataFrame with OHLCV data indexed by date
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Try to load from cache first
        cached_data = self.load_from_cache(self.symbol, start_date, end_date)
        if cached_data is not None:
            print(f"Loaded {self.symbol} data from cache ({start_date} to {end_date})")
            return cached_data
        
        # Fetch from Yahoo Finance
        print(f"Fetching {self.symbol} data from Yahoo Finance ({start_date} to {end_date})")
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if data.empty:
                raise ValueError(f"No data returned for {self.symbol}")
            
            # Clean and validate data
            data = self.clean_data(data)
            
            # Save to cache
            self.save_to_cache(data, self.symbol, start_date, end_date)
            
            print(f"Successfully fetched {len(data)} days of data")
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the fetched data."""
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Basic data validation
        if (data['High'] < data['Low']).any():
            raise ValueError("Invalid data: High < Low found")
        
        if (data['High'] < data['Open']).any() or (data['High'] < data['Close']).any():
            raise ValueError("Invalid data: High < Open or High < Close found")
        
        if (data['Low'] > data['Open']).any() or (data['Low'] > data['Close']).any():
            raise ValueError("Invalid data: Low > Open or Low > Close found")
        
        if (data['Volume'] < 0).any():
            raise ValueError("Invalid data: Negative volume found")
        
        # Sort by date to ensure chronological order
        data = data.sort_index()
        
        return data
    
    def get_latest_data(self, lookback_days: int = 1000) -> pd.DataFrame:
        """Get the most recent data for strategy development."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        return self.fetch_btc_data(start_date, end_date)
    
    def get_training_data(self) -> pd.DataFrame:
        """Get sufficient data for walk-forward backtesting (4+ years)."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=4*365 + 100)).strftime("%Y-%m-%d")  # 4+ years
        
        return self.fetch_btc_data(start_date, end_date)
    
    def calculate_adv(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Average Daily Volume for position sizing checks."""
        return data['Volume'].rolling(window=window).mean()
    
    def get_price_series(self, data: pd.DataFrame, price_type: str = "Close") -> pd.Series:
        """Extract price series for backtesting."""
        if price_type not in data.columns:
            raise ValueError(f"Price type '{price_type}' not found in data")
        
        return data[price_type]