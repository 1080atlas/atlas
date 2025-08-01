# Atlas - Development Documentation

Atlas is an autonomous trading strategy research loop for BTC-USD that uses GPT-based agents to generate, validate, and store algorithmic trading ideas.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Loader   │    │   Knowledge     │    │    Database     │
│   (BTC-USD)     │    │     Base        │    │   (SQLite)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────────┐
                    │    Pipeline Runner      │
                    │   (Orchestration)       │
                    └─────────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
    ▼                            ▼                            ▼
┌─────────┐              ┌─────────────┐              ┌─────────────┐
│Planner  │─────────────▶│Guard-Rail   │─────────────▶│Backtester   │
│(GPT-4)  │              │(AST Check)  │              │(VectorBT)   │
└─────────┘              └─────────────┘              └─────────────┘
                                                               │
                                                               ▼
                                                      ┌─────────────┐
                                                      │  Analyzer   │
                                                      │(Reporting)  │
                                                      └─────────────┘
```

## Module Documentation

### Core Components

#### 1. Data Loader (`src/data_loader.py`)
- **Purpose**: Fetches and caches BTC-USD daily candles from Yahoo Finance
- **Key Features**:
  - Local caching with 1-day TTL
  - Data validation and cleaning
  - Support for different time ranges (training vs. recent data)
- **Main Methods**:
  - `fetch_btc_data()`: Primary data fetching interface
  - `get_training_data()`: Returns 4+ years for walk-forward testing
  - `calculate_adv()`: Average daily volume for position sizing

#### 2. Database Layer (`src/database.py`)
- **Purpose**: SQLite-based storage for strategies and knowledge embeddings
- **Schema**:
  - `strategies`: id, timestamp, parent_id, code, motivation, metrics, analysis, status
  - `knowledge`: id, filepath, content, embedding, created_at
- **Key Features**:
  - Strategy lineage tracking via parent_id
  - JSON metrics storage
  - Embedding storage as binary blobs
- **Main Methods**:
  - `get_top_k()`: Best strategies by test Sharpe ratio
  - `get_children()`: Strategy evolution tree traversal
  - `store_strategy()`: Complete strategy persistence

#### 3. Guard-Rail System (`src/guard_rail.py`)
- **Purpose**: AST-based security and constraint validation
- **Checks Enforced**:
  - Library restrictions (pandas, numpy, vectorbt only)
  - No forward-looking operations (negative shifts)
  - Leverage ≤ 2×, position size ≤ 5% ADV
  - No network calls or dangerous functions
  - File operations restricted to /tmp
- **Main Methods**:
  - `check_strategy()`: Comprehensive strategy validation
  - `validate_signals()`: Runtime signal constraint checking

#### 4. Backtester (`src/backtester.py`)
- **Purpose**: Walk-forward validation with realistic cost modeling
- **Configuration**:
  - Train: 36 months, Validate: 6 months, Test: 1 month
  - Monthly rolling windows
  - 10 bps bid-ask spread + 10 bps slippage
- **Metrics Calculated**:
  - Sharpe ratio, maximum drawdown, turnover, beta
  - Win rate, profit factor, number of trades
- **Main Methods**:
  - `run_walk_forward_backtest()`: Complete validation pipeline
  - `execute_strategy()`: Safe strategy code execution
  - `calculate_transaction_costs()`: Realistic cost modeling

#### 5. Knowledge Base (`src/knowledge_base.py`)
- **Purpose**: Semantic search over financial research using embeddings
- **Features**:
  - Sentence-transformers embeddings (all-MiniLM-L6-v2)
  - Cosine similarity search
  - Automatic chunking of markdown content
- **Sample Knowledge Areas**:
  - Market structure and trading costs
  - Risk management principles
  - Backtesting best practices
  - Technical analysis concepts
- **Main Methods**:
  - `retrieve_top_n()`: Semantic search interface
  - `update_knowledge_base()`: Rebuild embeddings
  - `search_knowledge()`: Formatted citation retrieval

### AI Agents

#### 6. Planner (`src/planner.py`)
- **Purpose**: GPT-4 powered strategy evolution
- **Input**: Parent strategy, analyzer feedback, knowledge context
- **Output**: Modified strategy code + motivation + citations
- **Constraints**: Must generate valid Python with 'signals' variable
- **Main Methods**:
  - `plan_strategy()`: Core planning with knowledge integration
  - `generate_seed_strategy()`: Bootstrap with MA crossover
  - `validate_strategy_format()`: Code structure verification

#### 7. Analyzer (`src/analyzer.py`)
- **Purpose**: Generate structured analysis reports and improvement suggestions
- **Analysis Modes**:
  - **Rule-based** (default): Fast, deterministic analysis using predefined rules
  - **AI-powered** (optional): GPT-4 analysis using `prompts/analyzer.txt` template
- **Report Structure**:
  - Summary paragraph
  - Metrics table (train/validation/test)
  - Stability analysis (Sharpe < 0.3 detection)
  - Strengths & weaknesses
  - Next action for planner
- **Main Methods**:
  - `analyze_backtest_results()`: Complete report generation (routes to AI or rule-based)
  - `_generate_ai_report()`: GPT-4 powered analysis with prompt template
  - `_generate_rule_based_report()`: Fast deterministic analysis
  - `get_performance_summary()`: Key metrics extraction

### Prompt Templates

The system uses structured prompt templates to guide AI agent behavior:

#### Planner Template (`prompts/planner.txt`)
- Guides GPT-4 strategy modification and evolution
- Specifies output format requirements (Python code + motivation)
- Enforces guard-rail constraints and coding standards
- Integrates knowledge base citations

#### Analyzer Template (`prompts/analyzer.txt`)
- Structures GPT-4 analysis of backtest results
- Defines required report sections and format
- Provides performance benchmarks and improvement priorities
- Ensures actionable "Next Action" recommendations

### Pipeline Orchestration

#### 8. Pipeline Runner (`pipeline_runner.py`)
- **Purpose**: End-to-end workflow orchestration
- **Workflow**: Sample → Plan → Guard-rail → Backtest → Analyze → Store
- **Features**:
  - Comprehensive error handling and logging
  - Performance tracking and statistics
  - Support for seed generation and multi-iteration runs
- **Usage**:
  ```bash
  python pipeline_runner.py --openai-key YOUR_KEY --iterations 5
  python pipeline_runner.py --seed  # Generate initial strategy only
  python pipeline_runner.py --ai-analysis --iterations 1  # Use AI-powered analysis
  ```

## Testing

The test suite covers all major components with pytest:

- `test_guard_rail.py`: Security and constraint validation
- `test_backtester.py`: Walk-forward backtesting logic
- `test_database.py`: Storage and retrieval operations

Run tests with:
```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Setup and Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Generate Seed Strategy**:
   ```bash
   python pipeline_runner.py --seed
   ```

4. **Run Pipeline**:
   ```bash
   python pipeline_runner.py --iterations 1
   ```

## Data Flow

1. **Initialization**: Load BTC-USD data, initialize knowledge embeddings
2. **Strategy Selection**: Get top-K parent strategies or generate seed
3. **Planning**: GPT-4 modifies strategy using knowledge citations
4. **Validation**: AST analysis ensures safety constraints
5. **Backtesting**: Walk-forward validation with transaction costs
6. **Analysis**: Structured report with improvement suggestions
7. **Storage**: Persist strategy, metrics, and analysis with lineage

## Configuration

Key parameters can be modified in the respective modules:

- **Backtester**: Window sizes, transaction costs, slippage
- **Guard-Rail**: Leverage limits, position size constraints
- **Knowledge Base**: Embedding model, chunk sizes
- **Pipeline**: Number of parent strategies, iteration limits

## Guard-Rail Constraints

The system enforces strict constraints for safety:

- **Libraries**: Only pandas, numpy, vectorbt, math, datetime
- **Market Data**: No forward-looking operations or datetime.now()
- **Risk Limits**: Leverage ≤ 2×, position ≤ 5% ADV
- **Security**: No network calls, file operations only in /tmp
- **Execution**: All code runs in sandboxed environment

## Performance Metrics

Strategies are evaluated on:

- **Risk-Adjusted Returns**: Sharpe ratio (target ≥ 0.7)
- **Drawdown Control**: Maximum drawdown (target ≤ 15%)
- **Stability**: Consistent performance across test windows
- **Transaction Costs**: Realistic bid-ask and slippage modeling

## Future Enhancements

- Multi-asset expansion after 30-day stable operation
- Advanced risk models and position sizing
- Real-time market data integration
- Extended knowledge base with regulatory content
- Performance attribution and factor analysis

## Troubleshooting

Common issues and solutions:

1. **OpenAI API Errors**: Check API key and rate limits
2. **Data Loading Failures**: Verify internet connection for Yahoo Finance
3. **Guard-Rail Violations**: Review strategy code for banned operations
4. **Backtest Errors**: Ensure strategy defines 'signals' variable correctly
5. **Knowledge Base Issues**: Regenerate embeddings if search returns empty

## Contributing

When adding new modules:

1. Follow existing code structure and documentation patterns
2. Add comprehensive unit tests
3. Update this README with new component documentation
4. Ensure guard-rail constraints are maintained
5. Test integration with full pipeline

---

*Last Updated: August 1, 2025*
*Atlas Version: 1.0.0 (MVP)*