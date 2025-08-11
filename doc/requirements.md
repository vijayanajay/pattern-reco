# Requirements Document

## Introduction

This feature implements a CLI-only anomaly and pattern detection system specifically for Indian stocks (both large cap and small cap NSE-listed equities), following Kailash Nadh's philosophy of minimalism, testable increments, and walk-forward validation. The system identifies statistically grounded price patterns in NSE-listed equities that historically yield higher-than-market returns within ≤22 trading days, emphasizing parsimony, reproducibility, and out-of-sample survival.

## Kailash Nadh's 20 Hard Rules

1. **No complexity without proof**: Every feature must prove its worth in out-of-sample testing before inclusion
2. **Minimal viable everything**: Start with the smallest working slice, measure, then iterate or kill
3. **No lookahead bias ever**: Future data cannot influence past decisions under any circumstances
4. **Deterministic reproducibility**: Same config + same data + same seed = identical results every time
5. **Walk-forward validation only**: No cherry-picking time periods; rolling validation mimics reality
6. **Parameter parsimony**: Maximum 3 parameters per detector; complexity is the enemy
7. **Out-of-sample degradation tracking**: OOS/IS ratio must be ≥0.5 or the detector dies
8. **Transaction costs are reality**: Model fees, slippage, and execution constraints or results are fantasy
9. **Survivorship across multiple stocks**: Patterns must work on ≥5 liquid names or they're noise
10. **Final holdout is sacred**: Last 2 years untouched until final sign-off; no peeking allowed
11. **Frozen data snapshots**: Never refetch during backtests; snapshot once, diff on refresh
12. **No technical indicators**: Pure OHLCV statistics only; no RSI, MACD, or other derived nonsense
13. **Daily frequency minimum**: No intraday complexity; daily or higher timeframes only
14. **Single position per stock**: No pyramiding, no averaging down; one bet per name at a time
15. **Circuit breaker respect**: If market says no fill, accept it; no VWAP fallbacks or wishful thinking
16. **Median over mean always**: Robust statistics resist outliers; means lie, medians tell truth
17. **Pre-registered acceptance criteria**: Define success metrics before running; no post-hoc rationalization
18. **Configuration drives everything**: Single YAML file controls entire pipeline; no hidden CLI flags
19. **Audit trail mandatory**: Every run produces manifest with hashes, params, and git SHA
20. **Kill failing detectors fast**: Two consecutive OOS failures = permanent retirement

## Requirements

### Requirement 1: Configuration-Driven Pipeline

**User Story:** As a researcher, I want to define a single YAML configuration file that controls all aspects of data, universe selection, detector parameters, walk-forward splits, execution assumptions, portfolio rules, and reporting outputs, so that every run is completely reproducible with minimal CLI options and no hidden flags.

#### Acceptance Criteria

1. WHEN a user provides a valid config.yaml file THEN the system SHALL execute the complete pipeline using only parameters from that file
2. WHEN the config file contains run metadata (name, seed, output_dir) THEN the system SHALL use these values for deterministic execution and output organization
3. WHEN the config specifies data parameters (source, interval, date ranges) THEN the system SHALL fetch and process data according to these specifications
4. WHEN the config defines universe selection criteria THEN the system SHALL select stocks deterministically based on these rules
5. WHEN the config contains detector parameters THEN the system SHALL apply these parameters with per-stock overrides if specified
6. WHEN the config specifies walk-forward parameters THEN the system SHALL create splits according to these rules
7. WHEN the config defines execution and portfolio rules THEN the system SHALL simulate trading according to these constraints

### Requirement 2: Data Management and Integrity

**User Story:** As a data pipeline, I want to snapshot yfinance data to parquet files with complete metadata and produce diff reports on refresh, so that historical backtests are never silently altered by upstream data changes.

#### Acceptance Criteria

1. WHEN fetching data from yfinance THEN the system SHALL save raw data to parquet files with timestamp, timezone, interval, and source metadata
2. WHEN a data refresh is requested THEN the system SHALL compare new data with existing snapshots and produce a detailed diff report
3. WHEN data diffs exceed epsilon thresholds THEN the system SHALL freeze the old dataset for historical reproducibility
4. WHEN loading data for backtests THEN the system SHALL use only frozen snapshots and never refetch during execution
5. WHEN data integrity checks run THEN the system SHALL validate trading days against NSE calendar, check for splits/adjustments, and verify timezone alignment
6. WHEN data is missing or corrupted THEN the system SHALL fail hard with clear error messages rather than proceeding with bad data

### Requirement 3: Walk-Forward Validation Framework

**User Story:** As a backtester, I want to perform deterministic rolling walk-forward splits with 3-year in-sample periods, 1-year out-of-sample periods, and a final 2-year holdout, so that validation is honest and prevents any form of lookahead bias.

#### Acceptance Criteria

1. WHEN creating walk-forward splits THEN the system SHALL use 3-year in-sample and 1-year out-of-sample periods with annual rolling
2. WHEN fitting detector parameters THEN the system SHALL use only in-sample data and freeze parameters for out-of-sample testing
3. WHEN the final 2-year holdout period is defined THEN the system SHALL never use this data for any fitting or parameter selection
4. WHEN calendar alignment is enabled THEN the system SHALL align split boundaries with trading calendar dates
5. WHEN walk-forward validation completes THEN the system SHALL track OOS/IS performance ratios for each detector and stock combination
6. WHEN a detector shows OOS/IS ratio < 0.5 THEN the system SHALL flag it for potential retirement

### Requirement 4: Execution Simulation with Realistic Constraints

**User Story:** As an executor, I want to simulate next-open fills with circuit guards and parameterized slippage models while logging all unfilled signals, so that backtest results reflect real-world tradeability constraints.

#### Acceptance Criteria

1. WHEN a signal is generated THEN the system SHALL attempt to fill at the next trading day's open price
2. WHEN the next open price is outside the circuit guard range (±10% of previous close) THEN the system SHALL mark the trade as unfilled and log the reason
3. WHEN calculating slippage THEN the system SHALL apply the gap percentile model with turnover adjustments and minimum floors
4. WHEN applying transaction costs THEN the system SHALL add both fees (default 10 bps per side) and calculated slippage to each trade
5. WHEN a signal cannot be filled THEN the system SHALL log the unfilled signal with timestamp, reason, and attempted price
6. WHEN re-entry is attempted THEN the system SHALL respect lockout rules and not allow new positions until current position is closed

### Requirement 5: Portfolio Management and Position Sizing

**User Story:** As a portfolio allocator, I want to enforce maximum concurrent positions with deterministic selection rules based on signal extremity, turnover, and ticker alphabetical order, so that capacity constraints and tie-breaking are explicit and reproducible.

#### Acceptance Criteria

1. WHEN multiple signals are generated simultaneously THEN the system SHALL select positions using the deterministic ordering: signal extremity descending, turnover descending, ticker ascending
2. WHEN the maximum concurrent position limit is reached THEN the system SHALL reject additional signals and log them as capacity-constrained
3. WHEN weighting positions THEN the system SHALL use equal weighting (1/N) across all concurrent positions
4. WHEN a position exits THEN the system SHALL immediately make capacity available for new signals
5. WHEN re-entry lockout is enabled THEN the system SHALL prevent new positions in the same stock until the lockout period expires
6. WHEN position limits change THEN the system SHALL apply new limits only to future signals, not existing positions

### Requirement 6: Gap-Z Detector Implementation

**User Story:** As a detector, I want to compute Gap-Z signals with maximum 3 parameters (window, threshold, max_hold) and per-stock parameter fitting on in-sample data only, so that complexity is minimized and out-of-sample evaluation is clean.

#### Acceptance Criteria

1. WHEN calculating gap percentages THEN the system SHALL compute (Open_t - Close_{t-1}) / Close_{t-1} for each trading day
2. WHEN computing z-scores THEN the system SHALL use rolling mean and standard deviation over the specified window parameter
3. WHEN the z-score falls below the k_low threshold THEN the system SHALL generate a long entry signal
4. WHEN parameter fitting on in-sample data THEN the system SHALL test window values [20, 60], k_low values [-1.0, -1.5, -2.0], and max_hold ≤ 22 days
5. WHEN selecting optimal parameters THEN the system SHALL choose the combination that maximizes median trade return post-costs with minimum hit-rate requirements
6. WHEN parameters are fitted THEN the system SHALL freeze them for all out-of-sample and holdout testing

### Requirement 7: Universe Selection and Management

**User Story:** As a universe selector, I want to deterministically select the top N stocks by median daily turnover at backtest start (t0) and freeze this list for the entire backtest, so that survivorship bias is controlled and selection is transparent.

#### Acceptance Criteria

1. WHEN selecting universe at t0 THEN the system SHALL compute median daily turnover (Close * Volume in INR) over the trailing 2 years for all available NSE tickers including both large cap and small cap stocks
2. WHEN ranking stocks THEN the system SHALL select the top N stocks (default 10) by median turnover value from the combined large cap and small cap universe
3. WHEN the universe is selected THEN the system SHALL freeze this list for the entire backtest period and document it in the run manifest
4. WHEN exclusion lists are provided THEN the system SHALL remove specified tickers (e.g., ASM/GSM) from consideration
5. WHEN survivorship limitations exist THEN the system SHALL document the inability to include delisted stocks where yfinance data is unavailable
6. WHEN universe selection completes THEN the system SHALL log the exact list with turnover statistics and selection criteria

### Requirement 8: Comprehensive Reporting and Audit Trail

**User Story:** As a reporter, I want to emit trade ledgers, unfilled signal logs, summary JSON/MD reports, and a complete run manifest with hashes and configuration, so that every result is fully auditable and reproducible.

#### Acceptance Criteria

1. WHEN trades are executed THEN the system SHALL log timestamps, entry/exit prices, fees, slippage, notional amounts, PnL, and holding duration to a CSV trade ledger
2. WHEN signals cannot be filled THEN the system SHALL log unfilled signals with timestamps, reasons, and attempted prices to a separate CSV file
3. WHEN generating summary reports THEN the system SHALL create both JSON and Markdown formats with per-detector, per-stock IS vs OOS metrics
4. WHEN creating the run manifest THEN the system SHALL include data file hashes, universe list, walk-forward split dates, parameter grids, cost models, random seed, and git SHA
5. WHEN optional plots are requested THEN the system SHALL generate equity curves, drawdown charts, PnL histograms, and calibration plots to the plots directory
6. WHEN the run completes THEN the system SHALL organize all outputs under the specified output_dir with consistent naming conventions

### Requirement 9: Performance Metrics and Evaluation

**User Story:** As a reviewer, I want to see per-stock out-of-sample medians, lower-tail returns, hit rates, OOS/IS degradation ratios, and portfolio metrics against benchmarks in the generated report, so that I can make informed go/no-go decisions quickly.

#### Acceptance Criteria

1. WHEN calculating per-stock metrics THEN the system SHALL compute median trade return post-costs, 5th percentile trade return, and hit rate for each stock and time period
2. WHEN evaluating detector performance THEN the system SHALL track Sharpe ratios, maximum drawdown, and OOS/IS performance ratios
3. WHEN aggregating portfolio metrics THEN the system SHALL compute equal-weighted returns across concurrent positions with maximum concurrent position limits
4. WHEN comparing to benchmarks THEN the system SHALL calculate performance against stock buy-and-hold and NIFTY baseline over the same time windows
5. WHEN assessing robustness THEN the system SHALL verify that signals persist across ≥5 liquid names and ≥3 walk-forward rolls
6. WHEN generating final evaluation THEN the system SHALL clearly indicate pass/fail status against pre-registered acceptance thresholds

### Requirement 10: CLI Interface and Command Structure

**User Story:** As a user, I want a minimal CLI with only two commands - run and refresh-data - that accept only a config file path with no additional options, so that the interface is extremely simple and all complexity is contained within the configuration file.

#### Acceptance Criteria

1. WHEN running the main pipeline THEN the system SHALL accept only `python cli.py run --config path/to/config.yaml` with absolutely no additional CLI flags or options
2. WHEN refreshing data THEN the system SHALL accept only `python cli.py refresh-data --config path/to/config.yaml` for optional data updates with no additional CLI options
3. WHEN invalid commands are provided THEN the system SHALL display clear usage instructions and exit with appropriate error codes
4. WHEN config file is missing or invalid THEN the system SHALL provide specific error messages about what is wrong
5. WHEN commands execute successfully THEN the system SHALL provide progress updates and final status messages
6. WHEN errors occur during execution THEN the system SHALL log detailed error information and exit gracefully