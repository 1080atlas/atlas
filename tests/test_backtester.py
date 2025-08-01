import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtester import WalkForwardBacktester

class TestWalkForwardBacktester:
    
    def setup_method(self):
        self.backtester = WalkForwardBacktester(
            train_months=12,  # Shorter for testing
            validation_months=3,
            test_months=1,
            roll_months=1
        )
        
        # Create sample data (2 years of daily data)
        dates = pd.date_range('2020-01-01', periods=730, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        prices = 100 * np.exp(np.cumsum(np.random.randn(730) * 0.02))
        volumes = np.random.randint(10000, 100000, 730)
        
        self.sample_data = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(730) * 0.001),
            'High': prices * (1 + np.abs(np.random.randn(730)) * 0.002),
            'Low': prices * (1 - np.abs(np.random.randn(730)) * 0.002),
            'Close': prices,
            'Volume': volumes
        }, index=dates)
    
    def test_walk_forward_windows_generation(self):
        """Test that walk-forward windows are generated correctly."""
        windows = self.backtester.generate_walk_forward_windows(self.sample_data)
        
        assert len(windows) > 0
        
        for window in windows:
            # Check that all required keys exist
            required_keys = ['train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end']
            for key in required_keys:
                assert key in window
            
            # Check temporal ordering
            assert window['train_start'] < window['train_end']
            assert window['train_end'] < window['val_start']
            assert window['val_start'] < window['val_end']
            assert window['val_end'] < window['test_start']
            assert window['test_start'] < window['test_end']
    
    def test_simple_strategy_execution(self):
        """Test execution of a simple strategy."""
        simple_strategy = """
import pandas as pd
import numpy as np

# Simple buy and hold
signals = pd.Series(1.0, index=price.index)
"""
        
        prices = self.sample_data['Close'][:100]  # Use smaller subset
        
        signals = self.backtester.execute_strategy(simple_strategy, prices)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(prices)
        assert signals.index.equals(prices.index)
        assert all(signals == 1.0)  # Buy and hold should be all 1.0
    
    def test_moving_average_strategy_execution(self):
        """Test execution of a moving average crossover strategy."""
        ma_strategy = """
import pandas as pd
import numpy as np

# Moving average crossover
short_ma = price.rolling(5).mean()
long_ma = price.rolling(10).mean()

signals = pd.Series(0.0, index=price.index)
signals[short_ma > long_ma] = 1.0
signals[short_ma < long_ma] = -1.0
signals = signals.fillna(0.0)
"""
        
        prices = self.sample_data['Close'][:50]
        
        signals = self.backtester.execute_strategy(ma_strategy, prices)
        
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(prices)
        assert all(signals.abs() <= 1.0)  # Signals should be in [-1, 1]
    
    def test_backtest_period_execution(self):
        """Test running backtest for a specific period."""
        simple_strategy = """
import pandas as pd
import numpy as np

signals = pd.Series(0.5, index=price.index)  # 50% long position
"""
        
        start_date = self.sample_data.index[100]
        end_date = self.sample_data.index[200]
        
        result = self.backtester.run_backtest_period(
            simple_strategy, self.sample_data, start_date, end_date
        )
        
        # Check that all required metrics are present
        required_metrics = ['total_return', 'volatility', 'sharpe_ratio', 'max_drawdown', 
                          'turnover', 'beta', 'num_trades', 'win_rate', 'profit_factor']
        
        for metric in required_metrics:
            assert metric in result
            assert isinstance(result[metric], (int, float, np.integer, np.floating))
    
    def test_transaction_cost_calculation(self):
        """Test transaction cost calculation."""
        positions = pd.Series([0, 1, 1, 0, -1, -1, 0], 
                            index=self.sample_data.index[:7])
        prices = self.sample_data['Close'][:7]
        
        costs = self.backtester.calculate_transaction_costs(positions, prices)
        
        assert isinstance(costs, pd.Series)
        assert len(costs) == len(positions)
        assert all(costs >= 0)  # Costs should be non-negative
        
        # First position (0->1) should have cost
        assert costs.iloc[1] > 0
        # No change (1->1) should have no cost
        assert costs.iloc[2] == 0
    
    def test_full_walk_forward_backtest(self):
        """Test complete walk-forward backtesting."""
        simple_strategy = """
import pandas as pd
import numpy as np

# Simple momentum strategy
returns = price.pct_change()
momentum = returns.rolling(5).mean()

signals = pd.Series(0.0, index=price.index)
signals[momentum > 0] = 1.0
signals[momentum < 0] = -1.0
signals = signals.fillna(0.0)
"""
        
        # Use subset of data to speed up test
        test_data = self.sample_data.iloc[:400]  # About 13 months
        
        results = self.backtester.run_walk_forward_backtest(simple_strategy, test_data)
        
        # Check structure of results
        assert 'train_results' in results
        assert 'validation_results' in results
        assert 'test_results' in results
        assert 'aggregate_metrics' in results
        assert 'windows' in results
        
        # Check that we have results for each period
        assert len(results['train_results']) > 0
        assert len(results['validation_results']) > 0
        assert len(results['test_results']) > 0
        
        # Check aggregate metrics
        agg_metrics = results['aggregate_metrics']
        expected_agg_keys = ['test_avg_sharpe', 'test_avg_return', 'test_avg_maxdd']
        for key in expected_agg_keys:
            assert key in agg_metrics
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        insufficient_data = self.sample_data.iloc[:30]  # Only 1 month
        
        simple_strategy = """
import pandas as pd
import numpy as np
signals = pd.Series(1.0, index=price.index)
"""
        
        # Should raise error or return empty results
        try:
            results = self.backtester.run_walk_forward_backtest(simple_strategy, insufficient_data)
            # If it doesn't raise an error, should have no windows
            assert len(results['windows']) == 0
        except ValueError:
            # This is acceptable - insufficient data should be caught
            pass
    
    def test_invalid_strategy_handling(self):
        """Test handling of invalid strategy code."""
        invalid_strategy = """
import pandas as pd
import numpy as np

# This strategy doesn't define 'signals'
price_mean = price.mean()
"""
        
        prices = self.sample_data['Close'][:50]
        
        with pytest.raises(ValueError, match="Strategy must define 'signals' variable"):
            self.backtester.execute_strategy(invalid_strategy, prices)
    
    def test_strategy_with_syntax_error(self):
        """Test handling of strategy with syntax error."""
        syntax_error_strategy = """
import pandas as pd
import numpy as np

# Missing closing parenthesis
signals = pd.Series(1.0, index=price.index
"""
        
        prices = self.sample_data['Close'][:50]
        
        with pytest.raises(RuntimeError):
            self.backtester.execute_strategy(syntax_error_strategy, prices)

if __name__ == "__main__":
    pytest.main([__file__])