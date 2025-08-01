import ast
import re
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

class StaticGuardRail:
    def __init__(self):
        self.allowed_libraries = {
            'pandas', 'numpy', 'vectorbt', 'math', 'datetime'
        }
        
        self.allowed_builtins = {
            'pd', 'np', 'vbt', 'len', 'range', 'enumerate', 
            'min', 'max', 'abs', 'sum', 'price', 'data', 'signals'
        }
        
        self.banned_patterns = [
            r'\.now\(\)',
            r'time\.time\(\)',
            r'\.today\(\)',
            r'requests\.',
            r'urllib\.',
            r'http',
            r'socket\.',
            r'open\(',
            r'file\(',
            r'input\(',
            r'eval\(',
            r'exec\(',
        ]
        
        self.max_leverage = 2.0
        self.max_position_pct = 0.05  # 5% of ADV
    
    def check_strategy(self, strategy_code: str, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Comprehensive guard-rail check for strategy code.
        
        Returns:
            Dict with 'status' ('passed'/'failed') and 'violations' list
        """
        violations = []
        
        try:
            # Parse AST
            tree = ast.parse(strategy_code)
        except SyntaxError as e:
            return {
                'passed': False,
                'errors': [f"Syntax error: {str(e)}"]
            }
        
        # Check for banned patterns
        violations.extend(self._check_banned_patterns(strategy_code))
        
        # Check AST for dangerous operations
        violations.extend(self._check_ast(tree))
        
        # Check for forward-looking operations
        violations.extend(self._check_forward_looking(strategy_code))
        
        # Check leverage and position sizing (if data provided)
        if data is not None:
            violations.extend(self._check_position_sizing(strategy_code, data))
        
        # Check file operations
        violations.extend(self._check_file_operations(tree))
        
        # Check network operations
        violations.extend(self._check_network_operations(strategy_code))
        
        passed = len(violations) == 0
        
        return {
            'passed': passed,
            'errors': violations
        }
    
    def _check_banned_patterns(self, code: str) -> List[str]:
        """Check for banned regex patterns."""
        violations = []
        
        for pattern in self.banned_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Banned pattern detected: {pattern}")
        
        return violations
    
    def _check_ast(self, tree: ast.AST) -> List[str]:
        """Check AST for dangerous operations."""
        violations = []
        
        class DangerousVisitor(ast.NodeVisitor):
            def __init__(self):
                self.violations = []
            
            def visit_Import(self, node):
                for alias in node.names:
                    if alias.name not in self.parent.allowed_libraries:
                        self.violations.append(f"Banned import: {alias.name}")
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module and node.module not in self.parent.allowed_libraries:
                    # Allow specific pandas/numpy submodules
                    allowed_modules = ['pandas', 'numpy', 'vectorbt', 'math', 'datetime']
                    if not any(node.module.startswith(mod) for mod in allowed_modules):
                        self.violations.append(f"Banned import from: {node.module}")
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'input', '__import__']:
                        self.violations.append(f"Banned function call: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    # Check for dangerous method calls
                    if node.func.attr in ['now', 'today']:
                        self.violations.append(f"Banned method call: {node.func.attr}")
                
                self.generic_visit(node)
            
            def visit_Attribute(self, node):
                # Check for dangerous attribute access
                if isinstance(node.value, ast.Name):
                    if node.value.id == 'os' or node.value.id == 'sys':
                        self.violations.append(f"Banned module access: {node.value.id}.{node.attr}")
                
                self.generic_visit(node)
        
        visitor = DangerousVisitor()
        visitor.parent = self
        visitor.visit(tree)
        violations.extend(visitor.violations)
        
        return violations
    
    def _check_forward_looking(self, code: str) -> List[str]:
        """Check for forward-looking operations."""
        violations = []
        
        # Look for patterns that might indicate forward-looking
        forward_patterns = [
            r'\.shift\(\s*-',  # Negative shift (looking forward)
            r'future',
            r'tomorrow',
            r'next_',
            r'lead\(',
        ]
        
        for pattern in forward_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Potential forward-looking operation: {pattern}")
        
        return violations
    
    def _check_position_sizing(self, code: str, data: pd.DataFrame) -> List[str]:
        """Check position sizing and leverage constraints."""
        violations = []
        
        # Check for explicit leverage mentions
        leverage_patterns = [
            r'leverage\s*[=:]\s*([0-9.]+)',
            r'margin\s*[=:]\s*([0-9.]+)',
            r'\*\s*([3-9]|[1-9][0-9]+)',  # Multiplication by 3 or more
        ]
        
        for pattern in leverage_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and match.replace('.', '').isdigit():
                    value = float(match)
                    if value > self.max_leverage:
                        violations.append(f"Leverage {value}x exceeds maximum {self.max_leverage}x")
        
        # Check for position size patterns
        position_patterns = [
            r'position\s*[>>=]\s*([0-9.]+)',
            r'size\s*[>>=]\s*([0-9.]+)',
        ]
        
        for pattern in position_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and match.replace('.', '').isdigit():
                    value = float(match)
                    if value > self.max_position_pct:
                        violations.append(f"Position size {value*100}% exceeds maximum {self.max_position_pct*100}%")
        
        return violations
    
    def _check_file_operations(self, tree: ast.AST) -> List[str]:
        """Check for file operations outside /tmp."""
        violations = []
        
        class FileVisitor(ast.NodeVisitor):
            def __init__(self):
                self.violations = []
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['open', 'file']:
                        # Check if path argument contains /tmp
                        if node.args:
                            if isinstance(node.args[0], ast.Str):
                                path = node.args[0].s
                                if not path.startswith('/tmp'):
                                    self.violations.append(f"File operation outside /tmp: {path}")
                            elif isinstance(node.args[0], ast.Constant):
                                path = str(node.args[0].value)
                                if not path.startswith('/tmp'):
                                    self.violations.append(f"File operation outside /tmp: {path}")
                
                self.generic_visit(node)
        
        visitor = FileVisitor()
        visitor.visit(tree)
        violations.extend(visitor.violations)
        
        return violations
    
    def _check_network_operations(self, code: str) -> List[str]:
        """Check for network operations."""
        violations = []
        
        network_patterns = [
            r'requests\.',
            r'urllib\.',
            r'aiohttp\.',
            r'socket\.',
            r'http\.',
            r'ftp\.',
            r'smtp\.',
            r'\.get\(',
            r'\.post\(',
            r'\.put\(',
            r'\.delete\(',
            r'\.download',
            r'\.fetch',
        ]
        
        for pattern in network_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Network operation detected: {pattern}")
        
        return violations
    
    def validate_signals(self, signals: pd.Series, data: pd.DataFrame) -> List[str]:
        """Validate generated signals for additional constraints."""
        violations = []
        
        # Check signal range
        if signals.min() < -1 or signals.max() > 1:
            violations.append(f"Signals outside [-1, 1] range: min={signals.min():.3f}, max={signals.max():.3f}")
        
        # Check for excessive turnover (position changes)
        position_changes = signals.diff().abs()
        avg_turnover = position_changes.mean()
        
        if avg_turnover > 0.5:  # More than 50% position change per day on average
            violations.append(f"Excessive turnover: {avg_turnover:.3f} (daily average position change)")
        
        # Check for position size vs ADV if volume data available
        if 'Volume' in data.columns:
            avg_volume = data['Volume'].rolling(20).mean()  # 20-day ADV
            max_position_value = signals.abs().max() * data['Close']
            max_volume_pct = max_position_value / (avg_volume * data['Close'])
            
            if max_volume_pct.max() > self.max_position_pct:
                violations.append(f"Position size exceeds {self.max_position_pct*100}% of ADV")
        
        return violations