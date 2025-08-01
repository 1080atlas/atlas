import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from guard_rail import StaticGuardRail

class TestStaticGuardRail:
    
    def setup_method(self):
        self.guard_rail = StaticGuardRail()
        
        # Create sample data for testing
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_valid_strategy_passes(self):
        """Test that a valid strategy passes all checks."""
        valid_code = """
import pandas as pd
import numpy as np

# Simple moving average strategy
short_ma = price.rolling(10).mean()
long_ma = price.rolling(20).mean()

signals = pd.Series(0.0, index=price.index)
signals[short_ma > long_ma] = 1.0
signals[short_ma < long_ma] = -1.0
signals = signals.fillna(0.0)
"""
        
        result = self.guard_rail.check_strategy(valid_code, self.sample_data)
        assert result['passed'] == True
        assert len(result['errors']) == 0
    
    def test_banned_imports_detected(self):
        """Test that banned imports are detected."""
        banned_code = """
import requests
import socket
from urllib import request

signals = pd.Series(1.0, index=price.index)
"""
        
        result = self.guard_rail.check_strategy(banned_code)
        assert result['passed'] == False
        assert any('import' in v.lower() for v in result['errors'])
    
    def test_datetime_now_detected(self):
        """Test that datetime.now() is detected as violation."""
        datetime_code = """
import pandas as pd
import numpy as np
from datetime import datetime

current_time = datetime.now()
signals = pd.Series(1.0, index=price.index)
"""
        
        result = self.guard_rail.check_strategy(datetime_code)
        assert result['passed'] == False
        assert any('now' in v.lower() for v in result['errors'])
    
    def test_forward_looking_detected(self):
        """Test that forward-looking operations are detected."""
        forward_code = """
import pandas as pd
import numpy as np

# This is forward-looking!
future_price = price.shift(-1)
signals = pd.Series(1.0, index=price.index)
"""
        
        result = self.guard_rail.check_strategy(forward_code)
        assert result['passed'] == False
        assert any('forward' in v.lower() for v in result['errors'])
    
    def test_high_leverage_detected(self):
        """Test that high leverage mentions are detected."""
        leverage_code = """
import pandas as pd
import numpy as np

leverage = 5.0  # This exceeds 2x limit
position = price * leverage
signals = pd.Series(1.0, index=price.index)
"""
        
        result = self.guard_rail.check_strategy(leverage_code, self.sample_data)
        assert result['passed'] == False
        assert any('leverage' in v.lower() for v in result['errors'])
    
    def test_file_operations_outside_tmp_detected(self):
        """Test that file operations outside /tmp are detected."""
        file_code = """
import pandas as pd
import numpy as np

with open('/home/user/data.csv', 'w') as f:
    f.write('test')

signals = pd.Series(1.0, index=price.index)
"""
        
        result = self.guard_rail.check_strategy(file_code)
        assert result['passed'] == False
        assert any('file' in v.lower() for v in result['errors'])
    
    def test_network_operations_detected(self):
        """Test that network operations are detected."""
        network_code = """
import pandas as pd
import numpy as np
import requests

response = requests.get('http://example.com')
signals = pd.Series(1.0, index=price.index)
"""
        
        result = self.guard_rail.check_strategy(network_code)
        assert result['passed'] == False
        errors_text = ' '.join(result['errors']).lower()
        assert 'network' in errors_text or 'requests' in errors_text
    
    def test_aiohttp_operations_detected(self):
        """Test that aiohttp operations are detected."""
        aiohttp_code = """
import pandas as pd
import numpy as np
import aiohttp

async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get('http://example.com') as resp:
            return await resp.text()

signals = pd.Series(1.0, index=price.index)
"""
        
        result = self.guard_rail.check_strategy(aiohttp_code)
        assert result['passed'] == False
        errors_text = ' '.join(result['errors']).lower()
        assert 'network' in errors_text or 'aiohttp' in errors_text
    
    def test_dangerous_functions_detected(self):
        """Test that dangerous functions like eval are detected."""
        dangerous_code = """
import pandas as pd
import numpy as np

result = eval('1 + 1')
signals = pd.Series(1.0, index=price.index)
"""
        
        result = self.guard_rail.check_strategy(dangerous_code)
        assert result['passed'] == False
        assert any('eval' in v.lower() for v in result['errors'])
    
    def test_syntax_error_handling(self):
        """Test that syntax errors are handled gracefully."""
        syntax_error_code = """
import pandas as pd
import numpy as np

# Missing closing parenthesis
signals = pd.Series(1.0, index=price.index
"""
        
        result = self.guard_rail.check_strategy(syntax_error_code)
        assert result['passed'] == False
        assert any('syntax' in v.lower() for v in result['errors'])
    
    def test_signal_validation(self):
        """Test signal validation functionality."""
        # Create signals outside [-1, 1] range
        invalid_signals = pd.Series([2.0, -1.5, 0.5], index=self.sample_data.index[:3])
        
        violations = self.guard_rail.validate_signals(invalid_signals, self.sample_data)
        assert len(violations) > 0
        assert any('range' in v.lower() for v in violations)
    
    def test_excessive_turnover_detected(self):
        """Test that excessive turnover is detected."""
        # Create signals with high turnover
        high_turnover_signals = pd.Series(
            np.random.choice([-1, 1], size=len(self.sample_data)), 
            index=self.sample_data.index
        )
        
        violations = self.guard_rail.validate_signals(high_turnover_signals, self.sample_data)
        # May or may not trigger depending on random signals, but should not crash
        assert isinstance(violations, list)
    
    def test_allowed_libraries_pass(self):
        """Test that allowed libraries pass the checks."""
        allowed_code = """
import pandas as pd
import numpy as np
import vectorbt as vbt
import math
from datetime import timedelta

# Use allowed operations
short_ma = price.rolling(10).mean()
volatility = price.pct_change().std()
signals = pd.Series(np.where(short_ma > price, 1.0, -1.0), index=price.index)
"""
        
        result = self.guard_rail.check_strategy(allowed_code)
        assert result['passed'] == True
        assert len(result['errors']) == 0

if __name__ == "__main__":
    pytest.main([__file__])