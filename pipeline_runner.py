#!/usr/bin/env python3
"""
Atlas Pipeline Runner

Orchestrates the complete strategy research loop:
Sample → Plan → Guard-rail → Backtest → Analyze → Store

Usage:
    python pipeline_runner.py [--openai-key YOUR_KEY] [--iterations N]
"""

import os
import sys
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from database import DatabaseManager
from guard_rail import StaticGuardRail
from backtester import WalkForwardBacktester
from planner import StrategyPlanner
from analyzer import StrategyAnalyzer
from knowledge_base import KnowledgeBase

class AtlasPipeline:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        
        # Initialize components
        self.data_loader = DataLoader()
        self.db = DatabaseManager()
        self.guard_rail = StaticGuardRail()
        self.backtester = WalkForwardBacktester()
        self.planner = StrategyPlanner(openai_api_key)
        self.analyzer = StrategyAnalyzer(openai_api_key=self.openai_api_key)
        self.knowledge_base = KnowledgeBase()
        
        # Load and cache data
        print("Loading BTC-USD data...")
        self.data = self.data_loader.get_training_data()
        print(f"Loaded {len(self.data)} days of data from {self.data.index[0]} to {self.data.index[-1]}")
        
        # Initialize knowledge base
        if not self._knowledge_base_exists():
            print("Initializing knowledge base...")
            self.knowledge_base.update_knowledge_base()
    
    def _knowledge_base_exists(self) -> bool:
        """Check if knowledge base has been initialized."""
        try:
            entries = self.knowledge_base.db.get_all_knowledge()
            return len(entries) > 0
        except:
            return False
    
    def run_single_iteration(self) -> bool:
        """
        Run a single iteration of the research loop.
        
        Returns:
            bool: True if iteration completed successfully, False otherwise
        """
        try:
            print(f"\n{'='*60}")
            print(f"Starting Atlas iteration at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            # Step 1: Sample/Plan - Get parent strategies and generate new strategy
            print("\n1. PLANNING PHASE")
            print("-" * 40)
            
            parent_strategies = self.db.get_top_k(k=3)
            
            if not parent_strategies:
                # Generate seed strategy
                print("No parent strategies found. Generating seed strategy...")
                strategy_result = self.planner.generate_seed_strategy()
                parent_motivation = ""
                analyzer_feedback = ""
            else:
                # Use best parent strategy
                best_parent = parent_strategies[0]
                print(f"Using parent strategy ID {best_parent['id']} (Sharpe: {best_parent['metrics'].get('test_sharpe', 'N/A'):.3f})")
                
                # Extract analyzer feedback from parent
                analyzer_feedback = self._extract_next_action_from_analysis(best_parent.get('analysis', ''))
                
                # Plan new strategy
                strategy_result = self.planner.plan_strategy(
                    parent_code=best_parent['code'],
                    parent_motivation=best_parent.get('motivation', ''),
                    analyzer_feedback=analyzer_feedback,
                    knowledge_query="trading strategy improvement risk management"
                )
                parent_motivation = best_parent.get('motivation', '')
            
            new_code = strategy_result['code']
            new_motivation = strategy_result['motivation']
            
            print(f"Generated new strategy with motivation: {new_motivation[:100]}...")
            
            # Step 2: Guard-rail check
            print("\n2. GUARD-RAIL PHASE")
            print("-" * 40)
            
            guard_result = self.guard_rail.check_strategy(new_code, self.data)
            
            if guard_result['status'] == 'failed':
                print(f"❌ Guard-rail violations detected:")
                for violation in guard_result['violations']:
                    print(f"  - {violation}")
                
                # Store failed strategy
                strategy_id = self.db.store_strategy(
                    code=new_code,
                    motivation=new_motivation,
                    parent_id=parent_strategies[0]['id'] if parent_strategies else None,
                    status='failed'
                )
                
                print(f"Stored failed strategy with ID {strategy_id}")
                return False
            
            print("✅ Guard-rail checks passed")
            
            # Step 3: Backtest
            print("\n3. BACKTESTING PHASE")
            print("-" * 40)
            
            try:
                backtest_results = self.backtester.run_walk_forward_backtest(new_code, self.data)
                
                test_sharpe = backtest_results['aggregate_metrics'].get('test_avg_sharpe', 0)
                test_maxdd = backtest_results['aggregate_metrics'].get('test_avg_maxdd', 0)
                num_windows = len(backtest_results.get('test_results', []))
                
                print(f"✅ Backtest completed:")
                print(f"  - Test Sharpe: {test_sharpe:.3f}")
                print(f"  - Max Drawdown: {abs(test_maxdd)*100:.1f}%")
                print(f"  - Test Windows: {num_windows}")
                
            except Exception as e:
                print(f"❌ Backtesting failed: {str(e)}")
                
                # Store failed strategy
                strategy_id = self.db.store_strategy(
                    code=new_code,
                    motivation=new_motivation,
                    parent_id=parent_strategies[0]['id'] if parent_strategies else None,
                    status='backtest_failed'
                )
                
                return False
            
            # Step 4: Analysis
            print("\n4. ANALYSIS PHASE")
            print("-" * 40)
            
            # Store strategy first to get ID
            strategy_id = self.db.store_strategy(
                code=new_code,
                motivation=new_motivation,
                parent_id=parent_strategies[0]['id'] if parent_strategies else None,
                status='analyzing'
            )
            
            # Generate analysis
            analysis_report = self.analyzer.analyze_backtest_results(
                strategy_id=strategy_id,
                backtest_results=backtest_results,
                strategy_code=new_code,
                motivation=new_motivation
            )
            
            # Extract key metrics for database
            performance_summary = self.analyzer.get_performance_summary(backtest_results)
            
            print(f"✅ Analysis completed for strategy {strategy_id}")
            print(f"  - Performance summary: {performance_summary}")
            
            # Step 5: Store results
            print("\n5. STORAGE PHASE")
            print("-" * 40)
            
            # Update strategy with results
            self.db.update_strategy_metrics(strategy_id, performance_summary)
            self.db.update_strategy_analysis(strategy_id, analysis_report)
            self.db.update_strategy_status(strategy_id, 'completed')
            
            print(f"✅ Strategy {strategy_id} stored successfully")
            
            # Print iteration summary
            print(f"\n{'='*60}")
            print(f"ITERATION SUMMARY")
            print(f"{'='*60}")
            print(f"Strategy ID: {strategy_id}")
            print(f"Status: Completed")
            print(f"Test Sharpe: {test_sharpe:.3f}")
            print(f"Max Drawdown: {abs(test_maxdd)*100:.1f}%")
            print(f"Parent ID: {parent_strategies[0]['id'] if parent_strategies else 'None (seed)'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline iteration failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _extract_next_action_from_analysis(self, analysis: str) -> str:
        """Extract the 'Next Action for Planner' from analysis report."""
        if not analysis:
            return ""
        
        lines = analysis.split('\n')
        in_next_action = False
        next_action_lines = []
        
        for line in lines:
            if '## Next Action for Planner' in line:
                in_next_action = True
                continue
            elif line.startswith('##') and in_next_action:
                break
            elif in_next_action:
                next_action_lines.append(line)
        
        return '\n'.join(next_action_lines).strip()
    
    def run_pipeline(self, max_iterations: int = 1) -> Dict:
        """
        Run the complete pipeline for specified iterations.
        
        Args:
            max_iterations: Maximum number of iterations to run
            
        Returns:
            Dict with summary statistics
        """
        successful_iterations = 0
        failed_iterations = 0
        
        for i in range(max_iterations):
            print(f"\n🚀 Starting iteration {i+1}/{max_iterations}")
            
            success = self.run_single_iteration()
            
            if success:
                successful_iterations += 1
            else:
                failed_iterations += 1
            
            print(f"\nIteration {i+1} {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        # Generate summary
        summary = {
            'total_iterations': max_iterations,
            'successful_iterations': successful_iterations,
            'failed_iterations': failed_iterations,
            'success_rate': successful_iterations / max_iterations * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n{'='*60}")
        print(f"PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"Total iterations: {summary['total_iterations']}")
        print(f"Successful: {summary['successful_iterations']}")
        print(f"Failed: {summary['failed_iterations']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Atlas Strategy Research Pipeline')
    parser.add_argument('--openai-key', type=str, 
                       default=os.getenv('OPENAI_API_KEY'),
                       help='OpenAI API key (default: from OPENAI_API_KEY env var)')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of iterations to run (default: 1)')
    parser.add_argument('--seed', action='store_true',
                       help='Generate seed strategy only')
    
    args = parser.parse_args()
    
    if not args.openai_key:
        print("❌ Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --openai-key")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = AtlasPipeline(args.openai_key)
        
        if args.seed:
            # Generate seed strategy only
            print("Generating seed strategy...")
            strategy_result = pipeline.planner.generate_seed_strategy()
            
            strategy_id = pipeline.db.store_strategy(
                code=strategy_result['code'],
                motivation=strategy_result['motivation'],
                status='seed'
            )
            
            print(f"✅ Seed strategy generated with ID {strategy_id}")
        else:
            # Run full pipeline
            summary = pipeline.run_pipeline(args.iterations)
            
            if summary['success_rate'] > 0:
                print(f"\n🎉 Pipeline completed successfully!")
            else:
                print(f"\n❌ All iterations failed. Check logs for details.")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()