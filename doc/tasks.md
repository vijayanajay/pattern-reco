# Implementation Plan

- [x] 1. Set up project structure and configuration system





  - Create directory structure with src/, tests/, runs/, data/snapshots/
  - Implement YAML config loading with validation for all required fields
  - Create config schema validation for dates, parameters, and paths
  - Add error handling for missing/invalid config files
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

- [x] 2. Implement data fetching and snapshot management





  - Create yfinance data fetcher with error handling and rate limiting
  - Implement parquet snapshot saving with complete metadata (SHA256, timestamps, git hash)
  - Build snapshot loading functionality with integrity checks
  - Add data diff generation for refresh operations
  - Implement NSE trading calendar integration for date validation
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [ ] 3. Build universe selection system




  - Implement turnover calculation using Close * Volume in INR
  - Create deterministic stock ranking by median daily turnover
  - Add filtering for penny stocks and illiquid names
  - Implement universe freezing at t0 with documentation
  - Handle exclusion lists and survivorship bias documentation
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 4. Create feature engineering components
  - Implement daily returns calculation from adjusted close prices
  - Build gap percentage calculation: (Open - PrevClose) / PrevClose
  - Create rolling z-score computation with mean and standard deviation
  - Add robust statistics using median/MAD for outlier resistance
  - Implement proper handling of missing data and stock splits
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 5. Implement Gap-Z detector with parameter fitting
  - Create Gap-Z signal generation logic with configurable window and threshold
  - Build parameter grid search for window [20, 60] and k_low [-1.0, -2.0]
  - Implement in-sample parameter fitting using median trade return objective
  - Add parameter persistence and loading for out-of-sample testing
  - Create signal scoring system for position selection
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 6. Build walk-forward validation framework
  - Create time-series splitting with 3-year IS and 1-year OOS periods
  - Implement annual rolling with proper calendar alignment
  - Add final holdout period protection (never touched during fitting)
  - Build validation to prevent data leakage between splits
  - Create split boundary validation against NSE trading calendar
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 7. Implement execution simulation engine
  - Create next-day open fill simulation with realistic timing
  - Build circuit guard validation (±10% from previous close)
  - Implement gap-based slippage model with percentile thresholds
  - Add transaction fee calculation (10 bps per side)
  - Create unfilled signal logging with reasons and attempted prices
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 8. Build portfolio management system
  - Implement deterministic position selection using signal extremity, turnover, ticker order
  - Create capacity constraint enforcement (max 5 concurrent positions)
  - Add equal weight position sizing (₹1L per position)
  - Implement re-entry lockout rules until position closure
  - Build position tracking and availability management
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ] 9. Create performance metrics and evaluation system
  - Implement per-stock trade metrics: median return, 5th percentile, hit rate
  - Build OOS/IS performance ratio tracking for detector validation
  - Create portfolio aggregation with equal weighting across positions
  - Add benchmark comparison against buy-and-hold and NIFTY
  - Implement robustness checks across multiple stocks and time periods
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [ ] 10. Build comprehensive reporting system
  - Create CSV trade ledger with all transaction details and costs
  - Implement unfilled signal logging with timestamps and reasons
  - Build JSON and Markdown summary report generation
  - Create run manifest with data hashes, parameters, and git SHA
  - Add optional plot generation for equity curves and performance analysis
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [ ] 11. Implement CLI interface
  - Create main CLI entry point with run and refresh-data commands
  - Add config file path validation and error handling
  - Implement progress reporting and status messages
  - Build graceful error handling with detailed logging
  - Add command validation to prevent invalid usage
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [ ] 12. Create comprehensive test suite
  - Write unit tests for gap calculation, z-score computation, and split logic
  - Build integration test with end-to-end pipeline using synthetic data
  - Create property tests for determinism and no-lookahead validation
  - Add circuit guard and position selection determinism tests
  - Implement reproducibility tests with fixed seeds
  - _Requirements: All requirements validation through testing_

- [ ] 13. Add logging and error handling
  - Implement structured logging with timestamps and context
  - Create hard failure handling for critical errors (missing data, invalid config)
  - Add soft warning system for non-critical issues
  - Build error categorization and appropriate exit codes
  - Create audit trail logging for all major operations
  - _Requirements: Error handling across all requirements_

- [ ] 14. Integrate and test complete pipeline
  - Wire all components together in main execution flow
  - Test complete pipeline with real NSE data subset
  - Validate walk-forward execution with parameter fitting and OOS testing
  - Verify all outputs are generated correctly with proper naming
  - Test reproducibility with same config and seed
  - _Requirements: Integration of all requirements_

- [ ] 15. Create example configuration and documentation
  - Build example YAML config file with all required parameters
  - Create README with installation and usage instructions
  - Add configuration parameter documentation
  - Include example output interpretation guide
  - Document known limitations and assumptions
  - _Requirements: 1.1, 10.1, 10.2_