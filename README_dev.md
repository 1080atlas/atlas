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
  - JSON metrics storage with test_sharpe filtering
  - Embedding storage as binary blobs
  - Status-based strategy filtering ('candidate' for completed strategies)
- **Main Methods**:
  - `get_top_k()`: Best candidate strategies by test Sharpe ratio
  - `get_children()`: Strategy evolution tree traversal
  - `store_strategy()`: Complete strategy persistence
  - `update_strategy_metrics()`: Update performance metrics
  - `update_strategy_analysis()`: Store analysis reports
  - `update_strategy_status()`: Manage strategy lifecycle

#### 3. Guard-Rail System (`src/guard_rail.py`)
- **Purpose**: AST-based security and constraint validation
- **Checks Enforced**:
  - Library restrictions (pandas, numpy, vectorbt, math, datetime only)
  - No forward-looking operations (negative shifts)
  - Leverage ≤ 2×, position size ≤ 5% ADV
  - No network calls (requests, urllib, aiohttp, socket) or dangerous functions
  - File operations restricted to /tmp
  - No eval, exec, or dynamic imports
- **Return Format**: `{passed: bool, errors: [str]}`
- **Main Methods**:
  - `check_strategy()`: Comprehensive strategy validation
  - `validate_signals()`: Runtime signal constraint checking
  - `_check_network_operations()`: Network operation detection

#### 4. Backtester (`src/backtester.py`)
- **Purpose**: Walk-forward validation with realistic cost modeling
- **Configuration**:
  - Train: 36 months, Validate: 6 months, Test: 1 month
  - Monthly rolling windows with configurable purge days
  - 10 bps bid-ask spread + 10 bps slippage
- **Metrics Calculated**:
  - Sharpe ratio, maximum drawdown, turnover, beta
  - Win rate, profit factor, number of trades
  - Instability detection (test folds with Sharpe < 0.3)
- **Main Methods**:
  - `run_walk_forward_backtest()`: Complete validation with instability detection
  - `execute_strategy()`: Safe strategy code execution
  - `calculate_transaction_costs()`: Realistic cost modeling
  - `_check_sharpe_instability()`: Detects strategies with poor stability

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
  - `retrieve_top_n()`: Returns `{filepath, text}` format for consistent interface
  - `update_knowledge_base()`: Rebuild embeddings
  - `load_snippet_text()`: Load raw markdown text for citations
  - `_format_knowledge_snippets()`: Format snippets for AI consumption

### AI Agents

#### 6. Planner (`src/planner.py`)
- **Purpose**: GPT-4 powered strategy evolution
- **Input**: Parent strategy, analyzer feedback, knowledge context
- **Output**: `{code, motivation}` - Modified strategy code and explanation
- **Constraints**: Must generate valid Python with 'signals' variable
- **Main Methods**:
  - `plan_strategy()`: Core planning with database integration and Next Action feedback
  - `generate_seed_strategy()`: Bootstrap with MA crossover
  - `validate_strategy_format()`: Code structure verification
  - `_format_knowledge_snippets()`: Format knowledge for prompt inclusion

#### 7. Analyzer (`src/analyzer.py`)
- **Purpose**: Generate structured analysis reports and improvement suggestions using GPT-4
- **Features**:
  - GPT-4 powered analysis with knowledge base integration
  - Structured report generation following CLAUDE.md template
  - Instability detection integration (Sharpe < 0.3 threshold)
  - Dual save locations (file and database)
- **Report Structure**:
  - Summary paragraph
  - Metrics table (train/validation/test)
  - Stability analysis (instability flag passed from backtester)
  - Strengths & weaknesses
  - Next action for planner (feeds back into strategy evolution)
- **Main Methods**:
  - `analyze_backtest_results()`: Complete GPT-4 report generation with unstable flag
  - `_generate_ai_report()`: Core AI analysis with knowledge context
  - `get_performance_summary()`: Key metrics extraction for database
  - `_format_knowledge_snippets()`: Knowledge formatting for analysis context

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
  - Real `get_top_k()` integration with candidate strategy selection
  - Analyzer feedback extraction and planner integration
  - Instability flag passing from backtester to analyzer
  - Comprehensive error handling and logging
  - Performance tracking and statistics
  - Strategy status management (candidate/failed/analyzing)
- **Usage**:
  ```bash
  python pipeline_runner.py --openai-key YOUR_KEY --iterations 5
  python pipeline_runner.py --seed  # Generate initial strategy only
  ```

## Testing

The test suite covers all major components with pytest:

- `test_guard_rail.py`: Security and constraint validation (13 tests)
  - Updated for new `{passed, errors}` return format
  - Tests aiohttp network detection and all guard-rail features
- `test_backtester.py`: Walk-forward backtesting logic (10 tests)
  - Includes instability detection testing
  - Tests purged walk-forward validation
  - 5 years of synthetic data for comprehensive testing
- `test_database.py`: Storage and retrieval operations (14 tests)
  - Tests candidate status filtering and lineage tracking
  - Knowledge base storage and retrieval
- `test_knowledge_base.py`: Knowledge retrieval functionality
  - Comprehensive embedding and search testing (may skip in CI due to model dependencies)

Run tests with:
```bash
pip install -r requirements.txt
pytest tests/ -v  # Runs 37 core tests
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
- **Security**: No network calls (requests, urllib, aiohttp, socket), file operations only in /tmp
- **Execution**: All code runs in sandboxed environment with AST analysis
- **Return Format**: `{passed: bool, errors: [str]}` for consistent error handling

## Performance Metrics

Strategies are evaluated on:

- **Risk-Adjusted Returns**: Sharpe ratio (target ≥ 0.7)
- **Drawdown Control**: Maximum drawdown (target ≤ 15%)
- **Stability**: Test fold Sharpe ≥ 0.3 (strategies marked unstable if violated)
- **Transaction Costs**: Realistic bid-ask and slippage modeling with purged walk-forward
- **Consistency**: Performance persistence across multiple time windows

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

## Recent Updates (Roadmap Implementation)

### Database Layer Improvements
- Fixed `get_top_k()` to filter by `status='candidate'` instead of 'completed'
- Added comprehensive strategy lifecycle management
- Enhanced lineage tracking and parent-child relationships

### Knowledge Base Enhancements  
- Standardized `retrieve_top_n()` return format to `{filepath, text}`
- Added `load_snippet_text()` helper for citation loading
- Improved knowledge formatting for AI consumption

### Planner Integration
- Updated `plan_strategy()` to use database integration and analyzer feedback
- Changed return format to `{code, motivation}` for consistency
- Enhanced knowledge snippet formatting for prompt inclusion

### Guard-Rail Security Improvements
- Added aiohttp network operation detection
- Expanded AST checks for network calls and position sizing
- Updated return format to `{passed: bool, errors: [str]}`
- Enhanced pattern detection for security violations

### Backtester Stability Detection
- Implemented purged walk-forward validation with configurable purge days
- Added Sharpe instability detection (test fold Sharpe < 0.3)
- Enhanced metrics calculation with stability flags
- Added `_check_sharpe_instability()` method

### Analyzer Report Structure
- Implemented dual save locations (file + database)
- Added instability flag integration from backtester  
- Enhanced knowledge base integration with `retrieve_top_n()`
- Structured report generation following CLAUDE.md template

### Pipeline Runner Integration  
- Real `get_top_k()` calls with candidate strategy selection
- Analyzer feedback extraction and planner integration
- Instability flag passing from backtester to analyzer
- Enhanced strategy status management

### Testing Suite Expansion
- Updated 37 core tests for new interfaces and return formats
- Added comprehensive instability detection testing
- Enhanced guard-rail testing with aiohttp detection
- Updated database tests for candidate status filtering

---

*Last Updated: January 2025*
*Atlas Version: 1.1.0 (Roadmap Implementation Complete)*