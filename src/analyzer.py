import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import openai
from .knowledge_base import KnowledgeBase

class StrategyAnalyzer:
    def __init__(self, reports_dir: str = "reports", openai_api_key: str = None):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
        if not openai_api_key:
            raise ValueError("OpenAI API key required for analyzer")
            
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.knowledge_base = KnowledgeBase()
        
        # Load analyzer prompt template
        prompt_path = Path("prompts/analyzer.txt")
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                self.prompt_template = f.read()
        else:
            raise FileNotFoundError("Analyzer prompt template not found at prompts/analyzer.txt")
    
    def analyze_backtest_results(self, strategy_id: int, backtest_results: Dict, 
                               strategy_code: str, motivation: str, unstable: bool = False) -> str:
        """
        Analyze backtest results and generate comprehensive report using GPT-4.
        
        Args:
            strategy_id: ID of the strategy
            backtest_results: Results from walk-forward backtesting
            strategy_code: The strategy code that was tested
            motivation: Original motivation for the strategy
            unstable: Whether any test fold had Sharpe < 0.3
            
        Returns:
            Markdown formatted analysis report
        """
        report = self._generate_ai_report(strategy_id, backtest_results, strategy_code, motivation, unstable)
        
        # Save report to file
        report_path = self.reports_dir / f"{strategy_id}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report
    
    def _generate_ai_report(self, strategy_id: int, backtest_results: Dict, 
                           strategy_code: str, motivation: str, unstable: bool = False) -> str:
        """Generate analysis report using GPT-4 and knowledge base citations."""
        try:
            metrics = backtest_results['aggregate_metrics']
            
            # Retrieve relevant knowledge for analysis
            knowledge_query = f"trading strategy analysis performance metrics risk management {motivation}"
            knowledge_snippets = self.knowledge_base.retrieve_top_n(knowledge_query, n=3)
            knowledge_context = self._format_knowledge_snippets(knowledge_snippets)
            
            # Format metrics for AI analysis
            metrics_summary = f"""
Strategy Performance Summary:
- Test Sharpe Ratio: {metrics.get('test_avg_sharpe', 0):.3f}
- Test Return: {metrics.get('test_avg_return', 0)*100:.2f}%
- Test Max Drawdown: {metrics.get('test_avg_maxdd', 0)*100:.2f}%
- Test Turnover: {metrics.get('test_avg_turnover', 0):.3f}
- Test Beta: {metrics.get('test_avg_beta', 0):.3f}
- Unstable Windows: {metrics.get('test_unstable_windows', 0)} out of {len(backtest_results.get('test_results', []))}
- Stability Score: {((len(backtest_results.get('test_results', [])) - metrics.get('test_unstable_windows', 0)) / max(len(backtest_results.get('test_results', [])), 1) * 100):.1f}%
"""
            
            # Construct analysis prompt
            analysis_prompt = f"""
{self.prompt_template}

## Analysis Task:

**Strategy ID:** {strategy_id}

**Strategy Code:**
```python
{strategy_code}
```

**Strategy Motivation:**
{motivation}

**Backtest Results:**
{metrics_summary}

**Full Metrics:**
- Train Sharpe: {metrics.get('train_avg_sharpe', 0):.3f}
- Validation Sharpe: {metrics.get('validation_avg_sharpe', 0):.3f}
- Test Sharpe: {metrics.get('test_avg_sharpe', 0):.3f}

**Stability Analysis:**
- Strategy marked as {'UNSTABLE' if unstable else 'STABLE'} (based on test fold Sharpe < 0.3 threshold)
- Unstable test windows: {metrics.get('test_unstable_windows', 0)} out of {len(backtest_results.get('test_results', []))}

**Relevant Knowledge Context:**
{knowledge_context}

---

Please generate a complete analysis report following the required structure above. Focus on actionable insights and specific improvement recommendations. Cite relevant knowledge snippets in your analysis.
"""

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert quantitative trading analyst specializing in strategy performance evaluation and improvement."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"Strategy analysis failed: {str(e)}")
    
    def get_performance_summary(self, results: Dict) -> Dict:
        """Extract key performance metrics for database storage."""
        metrics = results['aggregate_metrics']
        
        return {
            'test_sharpe': metrics.get('test_avg_sharpe', 0),
            'test_return': metrics.get('test_avg_return', 0),
            'test_maxdd': metrics.get('test_avg_maxdd', 0),
            'test_turnover': metrics.get('test_avg_turnover', 0),
            'test_beta': metrics.get('test_avg_beta', 0),
            'stability_score': len(results.get('test_results', [])) - metrics.get('test_unstable_windows', 0),
            'num_test_windows': len(results.get('test_results', []))
        }
    
    def _format_knowledge_snippets(self, snippets: List[Dict]) -> str:
        """Format knowledge snippets for inclusion in prompt."""
        if not snippets:
            return "No relevant knowledge snippets found."
        
        formatted_snippets = []
        for i, snippet in enumerate(snippets, 1):
            filepath = snippet['filepath'].split('#')[0]  # Remove chunk ID
            filename = Path(filepath).name
            
            formatted_snippets.append(
                f"[{i}] From {filename}:\n{snippet['text']}\n"
            )
        
        return "\n".join(formatted_snippets)