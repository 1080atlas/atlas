import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class WalkForwardBacktester:
    def __init__(self, 
                 train_months: int = 36,  # 3 years
                 validation_months: int = 6,
                 test_months: int = 1,
                 roll_months: int = 1,
                 bid_ask_spread: float = 0.001,  # 10 bps
                 slippage: float = 0.001):  # 10 bps
        self.train_months = train_months
        self.validation_months = validation_months
        self.test_months = test_months
        self.roll_months = roll_months
        self.bid_ask_spread = bid_ask_spread
        self.slippage = slippage
    
    def generate_walk_forward_windows(self, data: pd.DataFrame) -> List[Dict]:
        """Generate walk-forward validation windows."""
        windows = []
        data_start = data.index[0]
        data_end = data.index[-1]
        
        # Start from the earliest possible test window
        current_start = data_start
        
        while True:
            # Calculate window boundaries
            train_end = current_start + pd.DateOffset(months=self.train_months)
            val_end = train_end + pd.DateOffset(months=self.validation_months)
            test_end = val_end + pd.DateOffset(months=self.test_months)
            
            # Check if we have enough data for this window
            if test_end > data_end:
                break
            
            # Find actual dates in the data index
            train_start_actual = data.index[data.index >= current_start][0]
            train_end_actual = data.index[data.index <= train_end][-1]
            val_start_actual = data.index[data.index > train_end_actual][0]
            val_end_actual = data.index[data.index <= val_end][-1]
            test_start_actual = data.index[data.index > val_end_actual][0]
            test_end_actual = data.index[data.index <= test_end][-1]
            
            window = {
                'train_start': train_start_actual,
                'train_end': train_end_actual,
                'val_start': val_start_actual,
                'val_end': val_end_actual,
                'test_start': test_start_actual,
                'test_end': test_end_actual
            }
            windows.append(window)
            
            # Move to next window
            current_start = current_start + pd.DateOffset(months=self.roll_months)
        
        return windows
    
    def execute_strategy(self, strategy_code: str, price_data: pd.Series) -> pd.Series:
        """Execute strategy code and return position signals."""
        try:
            # Create execution environment with price data
            execution_env = {
                'price': price_data,
                'data': price_data.to_frame('Close'),
                'pd': pd,
                'np': np,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'min': min,
                'max': max,
                'abs': abs,
                'sum': sum
            }
            
            # Execute strategy code
            exec(strategy_code, execution_env)
            
            # Extract signals (strategy should define 'signals' variable)
            if 'signals' not in execution_env:
                raise ValueError("Strategy must define 'signals' variable")
            
            signals = execution_env['signals']
            
            # Convert to pandas Series if not already
            if not isinstance(signals, pd.Series):
                signals = pd.Series(signals, index=price_data.index)
            
            # Ensure signals are properly aligned with price data
            signals = signals.reindex(price_data.index, method='ffill').fillna(0)
            
            # Clamp signals to [-1, 1] range
            signals = np.clip(signals, -1, 1)
            
            return signals
            
        except Exception as e:
            raise RuntimeError(f"Strategy execution failed: {str(e)}")
    
    def calculate_transaction_costs(self, positions: pd.Series, prices: pd.Series) -> pd.Series:
        """Calculate transaction costs including bid-ask spread and slippage."""
        # Calculate position changes (trades)
        position_changes = positions.diff().fillna(0)
        
        # Calculate costs: (bid_ask_spread + slippage) * |trade_size| * price
        total_cost_rate = self.bid_ask_spread + self.slippage
        costs = np.abs(position_changes) * total_cost_rate * prices
        
        return costs
    
    def run_backtest_period(self, strategy_code: str, data: pd.DataFrame, 
                           start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict:
        """Run backtest for a specific period."""
        # Extract period data
        period_data = data.loc[start_date:end_date]
        if len(period_data) < 2:
            raise ValueError(f"Insufficient data for period {start_date} to {end_date}")
        
        prices = period_data['Close']
        
        # Execute strategy
        signals = self.execute_strategy(strategy_code, prices)
        
        # Calculate positions (assuming full investment when signal != 0)
        positions = signals.copy()
        
        # Calculate returns
        price_returns = prices.pct_change().fillna(0)
        strategy_returns = positions.shift(1) * price_returns
        
        # Calculate transaction costs
        transaction_costs = self.calculate_transaction_costs(positions, prices)
        cost_returns = -transaction_costs / prices.shift(1)  # Convert to return terms
        
        # Net returns after costs
        net_returns = strategy_returns + cost_returns.fillna(0)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + net_returns).cumprod()
        
        # Calculate metrics using vectorbt
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=signals > 0,
            exits=signals < 0,
            freq='D'
        )
        
        # Extract key metrics
        total_return = cumulative_returns.iloc[-1] - 1
        volatility = net_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (net_returns.mean() * 252) / (net_returns.std() * np.sqrt(252)) if net_returns.std() > 0 else 0
        
        # Maximum drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Turnover (average daily position change)
        turnover = np.abs(positions.diff()).mean()
        
        # Beta vs buy-and-hold
        market_returns = price_returns
        if market_returns.std() > 0:
            beta = np.cov(net_returns, market_returns)[0, 1] / np.var(market_returns)
        else:
            beta = 0
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'turnover': turnover,
            'beta': beta,
            'num_trades': len(portfolio.orders.records_readable),
            'win_rate': portfolio.trades.win_rate if len(portfolio.trades.records_readable) > 0 else 0,
            'profit_factor': portfolio.trades.profit_factor if len(portfolio.trades.records_readable) > 0 else 0
        }
    
    def run_walk_forward_backtest(self, strategy_code: str, data: pd.DataFrame) -> Dict:
        """Run complete walk-forward backtest."""
        windows = self.generate_walk_forward_windows(data)
        
        if not windows:
            raise ValueError("Insufficient data for walk-forward backtesting")
        
        results = {
            'train_results': [],
            'validation_results': [],
            'test_results': [],
            'windows': windows
        }
        
        for i, window in enumerate(windows):
            try:
                # Train period
                train_result = self.run_backtest_period(
                    strategy_code, data, 
                    window['train_start'], window['train_end']
                )
                train_result['window_id'] = i
                results['train_results'].append(train_result)
                
                # Validation period
                val_result = self.run_backtest_period(
                    strategy_code, data,
                    window['val_start'], window['val_end']
                )
                val_result['window_id'] = i
                results['validation_results'].append(val_result)
                
                # Test period
                test_result = self.run_backtest_period(
                    strategy_code, data,
                    window['test_start'], window['test_end']
                )
                test_result['window_id'] = i
                results['test_results'].append(test_result)
                
            except Exception as e:
                print(f"Error in window {i}: {e}")
                continue
        
        # Calculate aggregate metrics
        results['aggregate_metrics'] = self.calculate_aggregate_metrics(results)
        
        return results
    
    def calculate_aggregate_metrics(self, results: Dict) -> Dict:
        """Calculate aggregate metrics across all windows."""
        metrics = {}
        
        for period in ['train', 'validation', 'test']:
            period_results = results[f'{period}_results']
            if not period_results:
                continue
            
            # Average metrics across windows
            metrics[f'{period}_avg_sharpe'] = np.mean([r['sharpe_ratio'] for r in period_results])
            metrics[f'{period}_avg_return'] = np.mean([r['total_return'] for r in period_results])
            metrics[f'{period}_avg_maxdd'] = np.mean([r['max_drawdown'] for r in period_results])
            metrics[f'{period}_avg_turnover'] = np.mean([r['turnover'] for r in period_results])
            metrics[f'{period}_avg_beta'] = np.mean([r['beta'] for r in period_results])
            
            # Stability metrics
            sharpe_ratios = [r['sharpe_ratio'] for r in period_results]
            metrics[f'{period}_sharpe_std'] = np.std(sharpe_ratios)
            metrics[f'{period}_min_sharpe'] = np.min(sharpe_ratios)
            metrics[f'{period}_unstable_windows'] = sum(1 for s in sharpe_ratios if s < 0.3)
        
        return metrics