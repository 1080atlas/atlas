import openai
from pathlib import Path
from typing import Dict, List, Optional
from .knowledge_base import KnowledgeBase

class StrategyPlanner:
    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.knowledge_base = KnowledgeBase()
        
        # Load prompt template
        prompt_path = Path("prompts/planner.txt")
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                self.prompt_template = f.read()
        else:
            raise FileNotFoundError("Planner prompt template not found at prompts/planner.txt")
    
    def plan_strategy(self, parent_code: str, parent_motivation: str = "", 
                     analyzer_feedback: str = "", 
                     knowledge_query: str = "trading strategy improvement") -> Dict:
        """
        Generate an improved strategy based on parent strategy and feedback.
        
        Args:
            parent_code: Python code of the parent strategy
            parent_motivation: Explanation of parent strategy logic
            analyzer_feedback: Suggestions from previous analysis
            knowledge_query: Query for relevant knowledge retrieval
            
        Returns:
            Dict with 'code', 'motivation', and 'citations'
        """
        # Retrieve relevant knowledge
        knowledge_context = self.knowledge_base.search_knowledge(knowledge_query, n=3)
        
        # Construct the planning prompt
        planning_prompt = f"""
{self.prompt_template}

## Current Task:

**Parent Strategy Code:**
```python
{parent_code}
```

**Parent Motivation:**
{parent_motivation}

**Analyzer Feedback:**
{analyzer_feedback}

**Relevant Knowledge Context:**
{knowledge_context}

---

Please generate an improved strategy following the format specified above. Focus on making incremental improvements that address the analyzer feedback while incorporating insights from the knowledge context.
"""

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert quantitative trader and strategy developer."},
                    {"role": "user", "content": planning_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content
            
            # Parse the response to extract code and motivation
            parsed_response = self._parse_response(response_text)
            parsed_response['knowledge_citations'] = knowledge_context
            
            return parsed_response
            
        except Exception as e:
            raise RuntimeError(f"Strategy planning failed: {str(e)}")
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse the LLM response to extract code and motivation."""
        lines = response_text.split('\n')
        
        code_lines = []
        motivation_lines = []
        
        in_code_block = False
        in_motivation = False
        
        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                continue
            elif line.strip() == '```' and in_code_block:
                in_code_block = False
                continue
            elif line.strip().startswith('## Motivation:') or line.strip().startswith('**Motivation:**'):
                in_motivation = True
                continue
            elif line.strip().startswith('##') and in_motivation:
                in_motivation = False
                continue
            
            if in_code_block:
                code_lines.append(line)
            elif in_motivation:
                motivation_lines.append(line)
        
        # Extract code and motivation
        code = '\n'.join(code_lines).strip()
        motivation = '\n'.join(motivation_lines).strip()
        
        # Fallback parsing if structured format not found
        if not code:
            # Look for any code blocks
            import re
            code_blocks = re.findall(r'```python\n(.*?)\n```', response_text, re.DOTALL)
            if code_blocks:
                code = code_blocks[0].strip()
        
        if not motivation:
            # Extract text after "Motivation:" or similar
            import re
            motivation_match = re.search(r'(?:Motivation|Explanation):\s*(.*?)(?:\n\n|\n##|$)', 
                                       response_text, re.DOTALL | re.IGNORECASE)
            if motivation_match:
                motivation = motivation_match.group(1).strip()
        
        # Validate that we have both code and motivation
        if not code:
            raise ValueError("No strategy code found in LLM response")
        
        if not motivation:
            motivation = "Strategy modification based on feedback and knowledge insights."
        
        return {
            'code': code,
            'motivation': motivation,
            'raw_response': response_text
        }
    
    def validate_strategy_format(self, code: str) -> bool:
        """Validate that the strategy code follows the required format."""
        required_elements = [
            'import pandas as pd',
            'import numpy as np',
            'signals'
        ]
        
        for element in required_elements:
            if element not in code:
                return False
        
        # Check that signals is assigned (not just mentioned in comments)
        import re
        if not re.search(r'signals\s*=', code):
            return False
        
        return True
    
    def generate_seed_strategy(self) -> Dict:
        """Generate a simple seed strategy to start the evolution process."""
        seed_code = """import pandas as pd
import numpy as np

# Simple Moving Average Crossover Strategy
short_window = 20
long_window = 50

# Calculate moving averages
short_ma = price.rolling(window=short_window).mean()
long_ma = price.rolling(window=long_window).mean()

# Generate signals
signals = pd.Series(0.0, index=price.index)

# Long when short MA > long MA, short when short MA < long MA
signals[short_ma > long_ma] = 1.0
signals[short_ma < long_ma] = -1.0

# Forward fill signals to avoid look-ahead bias
signals = signals.fillna(method='ffill').fillna(0.0)"""

        motivation = """Initial seed strategy using a classic 20/50 day moving average crossover. 
This provides a simple trend-following approach that goes long when short-term momentum 
exceeds long-term momentum and vice versa. The strategy serves as a baseline for 
evolutionary improvement."""
        
        return {
            'code': seed_code,
            'motivation': motivation,
            'knowledge_citations': 'Seed strategy - no citations'
        }