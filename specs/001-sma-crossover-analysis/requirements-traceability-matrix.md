## Requirements Traceability Matrix (RTM)

This document maps each formal requirement in `specs/001-sma-crossover-analysis` to concrete implementation tasks, proposed code artifacts, tests, and expected outputs. It is written from an architect's perspective with conservative, auditable mappings and explicit assumptions where the repo does not yet contain implementation files.

Runbook: file produced automatically to capture traceability. Use this file during implementation, code reviews, and QA.

---

### How to read this RTM

- Requirement ID: unique short id (R-xxx). Where the spec used FR-### or NFR-### we map to R-### for traceability.
- Source: original spec file(s) where the requirement is defined.
- Acceptance Criteria: short, testable acceptance.
- Tasks: recommended granular tasks (design, implement, tests, docs). Each task has a Task ID (T-xxx) for tracking.
- Code artifact(s): proposed file(s) and function/class names where the required behaviour will be implemented. If files do not exist yet, these are proposed locations under `src/sma_analysis/` consistent with `plan.md`.
- Test artifact(s): proposed pytest test files and test names covering the acceptance criteria (happy path + edge cases).
- Outputs & artifacts: files produced by a successful implementation (CSV, PNG, markdown, manifest, logs).
- Notes & assumptions: explicit assumptions and any follow-ups.

---

Summary of sources scanned

- `specs/001-sma-crossover-analysis/spec.md`
- `specs/001-sma-crossover-analysis/plan.md`
- `specs/001-sma-crossover-analysis/data-model.md`
- `specs/001-sma-crossover-analysis/quickstart.md`
- `specs/001-sma-crossover-analysis/research.md`
- `specs/001-sma-crossover-analysis/checklists/requirements.md`
- `specs/001-sma-crossover-analysis/contracts/cli.md`
- `specs/001-sma-crossover-analysis/contracts/csv-schema.md`

All FR-* and NFR-* items in the spec were used as the authoritative source for functional and non-functional requirements.

---

SECTION A — Functional Requirements

R-001 (FR-001) — Data Acquisition & Preprocessing
 - Source: `spec.md` (FR-001 Data Acquisition & Preprocessing)
 - Text (short): Given tickers + date range, download daily Adj Close, forward-fill sporadic NaNs, produce per-ticker cleaned series and polars DataFrame; log NaN fill counts and warnings.
 - Acceptance Criteria:
   - For provided tickers and range, function returns a `polars.DataFrame` (date index) with tickers as columns and no NaNs after forward-fill.
   - Per-ticker `na_filled_count` and `percent_missing_before_fill` logged into `outputs/logs/na_fill_warnings_{RUN_TS}.csv`.
   - Download retries: exponential backoff, 3 attempts. Failures recorded in `outputs/logs/download_failures_{RUN_TS}.csv` and `run_manifest`.
 - Tasks:
   - T-001.1: Design download/adapter interface (DataProvider) supporting `yfinance` and future providers.
   - T-001.2: Implement downloader with retry/backoff and per-ticker diagnostics.
   - T-001.3: Implement cleaning pipeline (forward-fill, percent-missing check, warmup counting).
   - T-001.4: Add unit tests + integration smoke test for download + clean.
 - Code artifacts (proposed):
   - `src/sma_analysis/data.py`
     - function: `download_tickers(tickers: list[str], start: str, end: str, provider: str = "yfinance") -> dict[str, polars.DataFrame]`
     - function: `clean_timeseries(ts_df: polars.DataFrame, na_threshold: float) -> polars.DataFrame, diagnostics: dict`
     - constants/config helper: `DEFAULT_DOWNLOAD_ATTEMPTS = 3`, `BACKOFF_BASE_SEC = 2`
 - Test artifacts:
   - `tests/unit/test_data_download.py`
     - test: `test_download_with_retries_success_and_failure()` (mock yfinance to simulate temporary failures)
     - test: `test_forward_fill_and_na_threshold_flagging()` (create synthetic series with NaNs)
 - Outputs & artifacts:
   - `outputs/logs/na_fill_warnings_{RUN_TS}.csv`
   - `outputs/logs/download_failures_{RUN_TS}.csv`
   - diagnostics written to `outputs/sma_crossover_analysis_diagnostics_{RUN_TS}.csv`
 - Notes & assumptions:
   - Assumption: `polars` is available in runtime; if not, provide shim to pandas.
   - When creating API, use `polars.DataFrame` as canonical in-memory type per NFR-002.

R-002 (FR-002) — Time Period Segmentation (cohorts)
 - Source: `spec.md` (FR-002 Time Period Segmentation)
 - Text (short): Slice 2004-01-01 to 2023-12-31 into four inclusive 5-year cohorts; assign signals to cohort by T (signal date). Forward returns may cross cohort boundaries.
 - Acceptance Criteria:
   - Cohort boundaries returned by period generator match spec (exact dates).
   - Signal assignment function assigns each signal to the appropriate cohort label.
 - Tasks:
   - T-002.1: Implement deterministic cohort generator.
   - T-002.2: Tests validating signal assignment to cohorts and forward-return behavior across boundaries.
 - Code artifacts (proposed):
   - `src/sma_analysis/utils.py`
     - function: `generate_cohorts(start_date, end_date, cohort_length_years=5) -> list[tuple[str,str,str]]` (returns list of (label,start,end))
     - function: `assign_signal_to_period(signal_date, cohorts) -> period_label`
 - Test artifacts:
   - `tests/unit/test_periods.py`
     - test: `test_generate_canonical_cohorts()`
     - test: `test_assign_signal_near_boundary()`
 - Outputs & artifacts:
   - cohort list used in run manifest and `analysis_summary.md`
 - Notes & assumptions:
   - The canonical cohorts are hard-coded in the default config; generator supports custom ranges for testing.

R-003 (FR-003) — Signal Generation
 - Source: `spec.md` (FR-003 Signal Generation)
 - Text (short): Compute SMAs for canonical 10 pairs and return buy signals where SMA_short[T] > SMA_long[T] AND SMA_short[T-1] <= SMA_long[T-1]. Support overlapping and non-overlapping modes.
 - Acceptance Criteria:
   - For a synthetic test series, signals match expected indices for both overlapping and non_overlapping modes.
   - SMA-pairs validated (Short < Long, unique); duplicate/invalid input raises a validation error and causes CLI to exit with code 2.
 - Tasks:
   - T-003.1: Implement SMA rolling with full-window requirement (min_periods = window).
   - T-003.2: Implement signal detector using T & T-1 comparisons.
   - T-003.3: Implement overlap-mode handling (`overlapping` vs `non_overlapping`).
   - T-003.4: Unit tests for SMA calculation and signal detection edge cases.
 - Code artifacts (proposed):
   - `src/sma_analysis/signals.py`
     - function: `compute_sma(series: polars.Series, window: int) -> polars.Series`
     - function: `detect_signals(df: polars.DataFrame, sma_pairs: list[tuple[int,int]], overlap_mode: str = "overlapping") -> list[SignalRecord]`
 - Test artifacts:
   - `tests/unit/test_signals.py`
     - test: `test_signal_rule_basic_case()`
     - test: `test_sma_warmup_rows_are_excluded()`
     - test: `test_non_overlapping_mode_suppresses_overlaps()`
 - Outputs & artifacts:
   - `signals` intermediate structure serialized if requested for debugging: `outputs/signals_{RUN_TS}.parquet` (optional)
 - Notes & assumptions:
   - Implementation must ensure SMA requires full-window; i.e., rows before the long window produce no SMA and are excluded from candidate signals.

R-004 (FR-004) — Post-Signal Performance (forward returns & drawdown)
 - Source: `spec.md` (FR-004 Post-Signal Performance)
 - Text (short): For each signal compute forward Return_N and Max_Drawdown_N for N in {10,15,20} trading days; discard signals without sufficient forward rows for a given N.
 - Acceptance Criteria:
   - Return_N computed as Price_{T+N}/Price_T - 1.
   - Max_Drawdown_N computed as max_{t in [T,T+N]} (1 - Price_t/Price_T).
   - Signals without sufficient forward data are flagged and discarded for that N.
 - Tasks:
   - T-004.1: Implement `compute_forward_returns_and_drawdown(signal_records, holding_periods)`.
   - T-004.2: Tests for boundary discards and drawdown correctness.
 - Code artifacts (proposed):
   - `src/sma_analysis/metrics.py`
     - function: `compute_forward_outcomes(series: polars.Series, signal_dates: list[date], holding_periods: list[int]) -> DataFrame[SignalRecord]`
     - function: `max_drawdown_from_entry(entry_price, window_prices) -> float`
 - Test artifacts:
   - `tests/unit/test_metrics.py`
     - test: `test_forward_returns_and_drawdown_basic()`
     - test: `test_discard_insufficient_forward_data()`
 - Outputs & artifacts:
   - per-signal table used for aggregation, optional intermediate write: `outputs/signals_with_outcomes_{RUN_TS}.parquet`
 - Notes & assumptions:
   - Trading days measured as N rows forward in the cleaned per-ticker series (per spec).

R-005 (FR-005) — Aggregation & Stats
 - Source: `spec.md` (FR-005 Aggregation & Stats)
 - Text (short): Aggregate per (Stock, Period_Label, Short_SMA, Long_SMA, Holding_Period) compute Num_Signals, Num_Signals_Used, Mean_Return, Std_Dev_Return, Mean_Max_Drawdown, Win_Rate, P_Value via t-test; compute median, skew, kurtosis and bootstrap 95% CI; perform BH-FDR per holding period.
 - Acceptance Criteria:
   - Aggregation rows computed for every combination; sentinel rows emitted for zero-signal groups per `data-model.md`.
   - P_Value computed for `Num_Signals_Used >= 5`, otherwise `P_Value = NA` and `Low_Power_Flag = TRUE`.
   - BH-FDR applied per holding period among rows with `Num_Signals_Used >= 5` and stored in `Adj_P_Value_BH`.
  - P_Value is computed using `scipy.stats.ttest_1samp` (two-sided by default); the aggregation output or companion diagnostics MUST also record the t-statistic and degrees-of-freedom (df) for each aggregated row to aid auditability and reproducibility.
  - Output requirement: The canonical aggregated CSV MUST include two additional audit columns for each aggregated row: `T_Stat` (float or `NA`) and `DF` (int or `NA`). The aggregator and CSV writer MUST populate these fields (or emit `NA` for sentinel rows). Add unit tests to verify values are present and match the aggregator diagnostics.
 - Tasks:
   - T-005.1: Implement aggregator that returns canonical aggregated rows (including sentinel rows).
   - T-005.2: Implement p-value calculation and bootstrap CI (1000 resamples, seed=42).
   - T-005.3: Implement BH-FDR wrapper using `statsmodels` if available, fall back to builtin deterministic implementation.
   - T-005.4: Tests for aggregation sentinel rows, small-sample logic, BH-FDR parity tests.
 - Code artifacts (proposed):
   - `src/sma_analysis/aggregate.py`
     - function: `aggregate_signals(signal_outcomes_df: polars.DataFrame, holding_periods: list[int], bh_backend: str) -> polars.DataFrame`
     - function: `bootstrap_mean_ci(values: np.ndarray, resamples: int, seed: int) -> tuple[float,float]`
     - function: `apply_bh_fdr(pvals: np.ndarray, backend: str = 'statsmodels') -> np.ndarray`
 - Test artifacts:
   - `tests/unit/test_aggregate.py`
     - test: `test_aggregation_sentinels_present()`
     - test: `test_bootstrap_ci_reproducible_given_seed()`
     - test: `test_bh_fdr_matches_statsmodels_if_available()`
    - test: `test_bootstrap_reproducible_seeded()`
    - test: `test_p_value_and_tstat_recorded()`
    - test: `test_csv_includes_tstat_and_df_columns()`
 - Outputs & artifacts:
   - `outputs/sma_crossover_analysis_results_{RUN_TS}.csv`
   - `outputs/sma_crossover_analysis_diagnostics_{RUN_TS}.csv`
 - Notes & assumptions:
   - Use `scipy.stats.ttest_1samp` for P_Value (two-sided default).
   - BH-FDR backend preference stored in `run_manifest` as `bh_fdr_backend`.

R-006 (FR-006) — Buy-and-Hold Benchmark
 - Source: `spec.md` (FR-006 Buy-and-Hold Benchmark)
 - Text (short): For each stock and period compute buy-and-hold return = (Last - First)/First and include in final aggregation.
 - Acceptance Criteria:
   - Column `Buy_and_Hold_Return` present in aggregated CSV and numeric where period has data.
 - Tasks:
   - T-006.1: Add buy-and-hold computation into aggregation pipeline.
   - T-006.2: Test buy-and-hold correctness on sample series.
 - Code artifacts:
   - `src/sma_analysis/aggregate.py` (function: `compute_buy_and_hold(series, start, end) -> float`)
 - Test artifacts:
   - `tests/unit/test_benchmark.py` (test: `test_buy_and_hold_basic`)
 - Outputs & artifacts:
   - included as column in `outputs/sma_crossover_analysis_results_{RUN_TS}.csv`

R-007 (FR-007) — CSV Output (schema)
 - Source: `contracts/csv-schema.md`, `spec.md` (FR-007 CSV Output)
 - Text (short): Write timestamped canonical CSV containing specified columns (Stock, Period_Label, Period_Start, Period_End, Short_SMA, Long_SMA, Holding_Period, Num_Signals, Num_Signals_Used, Low_Power_Flag, Mean_Return, Std_Dev_Return, Median_Return, Mean_CI_Lower, Mean_CI_Upper, Mean_Max_Drawdown, Win_Rate, P_Value, Adj_P_Value_BH, Skew, Kurtosis, Overlap_Mode, Entry_Mode, Notes, Timestamp)
 - Acceptance Criteria:
   - CSV header and sentinel semantics exactly match `contracts/csv-schema.md` during write.
   - CSV present at `outputs/sma_crossover_analysis_results_{RUN_TS}.csv`.
 - Tasks:
   - T-007.1: Implement CSV writer honoring column order, sentinel rules for numeric NaNs and textual `NA` sentinel when appropriate.
   - T-007.2: Add unit/integration test ensuring header parity with `contracts/csv-schema.md`.
 - Code artifacts:
   - `src/sma_analysis/io.py`
     - function: `write_canonical_csv(df: polars.DataFrame, path: str, schema_contract_path: str) -> None`
 - Test artifacts:
   - `tests/unit/test_io.py` (test: `test_csv_header_and_sentinels_match_contract`)
  - Add test: `test_csv_numeric_missing_is_empty_cell()` to assert numeric NA cells are empty and textual sentinel values use exact literals (e.g., `no_signals`).
 - Outputs & artifacts:
   - `outputs/sma_crossover_analysis_results_{RUN_TS}.csv`
 - Notes & assumptions:
   - Implementation must emit numeric missing values as empty CSV fields (not literal "NA").

R-008 (FR-008) — Heatmap Visualization
 - Source: `spec.md` (FR-008 Heatmap Visualization)
 - Text (short): Create `heatmap_avg_mean_return.png` per holding period, divergent colormap centered at 0, masking untested cells, 300 DPI.
 - Acceptance Criteria:
   - PNG files saved for each holding period `outputs/heatmap_avg_mean_return_{holding_period}d_{RUN_TS}.png` and weighted variant.
   - Axes labeled as Short_SMA/Long_SMA and colorbar centered at 0.
 - Tasks:
   - T-008.1: Implement heatmap generator with masking for missing combos and weighted/unweighted options.
   - T-008.2: Tests verifying files created and basic numeric checks on heatmap matrix.
 - Code artifacts:
   - `src/sma_analysis/viz.py`
     - function: `plot_heatmap(aggregated_df, holding_period, weighted=False, out_path: str)`
 - Test artifacts:
   - `tests/unit/test_viz.py` (test: `test_heatmap_file_created_and_axes_labeled`)
 - Outputs & artifacts:
   - `outputs/heatmap_avg_mean_return_{holding_period}d_{RUN_TS}.png`
   - `outputs/heatmap_weighted_by_signals_{holding_period}d_{RUN_TS}.png`

R-009 (FR-009) — Markdown Summary Report
 - Source: `spec.md` (FR-009 Markdown Summary Report)
 - Text (short): Write `analysis_summary.md` that includes Executive Summary, Heatmap reference, Consistency Analysis, Top Performers, Fallen Angel case for YESBANK.NS, Benchmark Comparison, Signal Frequency Analysis, reproducibility header.
 - Acceptance Criteria:
   - `outputs/analysis_summary_{RUN_TS}.md` exists and contains the listed sections and references to generated artifacts (heatmap PNG, CSV), and reproducibility header (library versions, seed, git commit, branch).
 - Tasks:
   - T-009.1: Implement report writer assembling sections from run artifacts.
   - T-009.2: Test that summary includes required subsections and references.
 - Code artifacts:
   - `src/sma_analysis/report.py`
     - function: `write_analysis_summary(aggregated_csv_path, heatmap_paths, diagnostics_path, out_md_path, manifest)`
 - Test artifacts:
   - `tests/unit/test_report.py` (test: `test_summary_includes_repro_header_and_yesbank_section`)
 - Outputs & artifacts:
   - `outputs/analysis_summary_{RUN_TS}.md`
   - `outputs/yesbank_signals_{RUN_TS}.png` (if YESBANK.NS present)

SECTION B — Non-Functional Requirements (NFRs)

R-010 (NFR-001) — Performance Target
 - Source: `spec.md` (NFR-001 Performance)
 - Text (short): Canonical run should complete within 30 minutes on 4-core/16GB laptop for canonical dataset — prefer vectorized polars ops and optional parallelism.
 - Acceptance Criteria:
   - On CI or developer machine with similar resources, run time reasonably approaches target; tests verify function-level performance bounds for critical functions (e.g., SMA computation on 10-year series completes within X sec).
 - Tasks:
   - T-010.1: Add optional batching/workers to CLI (config keys `batch_size`, `workers`).
   - T-010.2: Profile critical paths and add `polars.set_num_threads()` guidance for worker processes.
 - Code artifacts:
   - `src/sma_analysis/cli.py` (worker orchestration and config parsing)
 - Test artifacts:
   - `tests/integration/test_performance_smoke.py` (time-bounded smoke run on small dataset with asserts on upper bound)
 - Notes & assumptions:
   - Exact performance will depend on machine; provide guidance in README and run_manifest.

R-011 (NFR-002) — Data Engine (polars)
 - Source: `spec.md` (NFR-002 Data Engine)
 - Text (short): Use polars as primary DataFrame engine; record engine and version in manifest.
 - Acceptance Criteria:
   - `run_manifest` includes `polars` version and python version.
 - Tasks:
   - T-011.1: Add runtime environment capture utility used by manifest writer.
 - Code artifacts:
   - `src/sma_analysis/manifest.py` (function: `write_run_manifest(config, out_path)`)
 - Test artifacts:
   - `tests/unit/test_manifest.py` (test: `test_manifest_contains_versions_and_git_provenance`)

R-012 (NFR-003) — Version Pinning & Reproducibility Policy
 - Source: `spec.md` (NFR-003)
 - Text (short): Critical numeric libraries pinned in requirements; run produces `requirements_exact_{RUN_TS}.txt` and `run_manifest` includes exact versions and git commit/branch.
 - Acceptance Criteria:
   - `outputs/requirements_exact_{RUN_TS}.txt` present; manifest contains `requirements_file` path.
 - Tasks:
   - T-012.1: Add step in CLI run to call `pip freeze` (or equivalent) and write exact requirements. On Windows use `python -m pip freeze | sort > outputs\requirements_exact_{RUN_TS}.txt`.
   - T-012.2: Add manifest writer to record `requirements_file` and `requirements_exact_file`.
 - Code artifacts:
   - `src/sma_analysis/manifest.py` (integrates with CLI)
 - Test artifacts:
   - `tests/unit/test_requirements_export.py` (test: `test_requirements_file_written_when_env_available`)

 - Manifest required fields & fallback behavior (add to run_manifest schema):
   - `run_ts` (string, UTC compact ISO `YYYYMMDDTHHMMSSZ`)
   - `python_version`, `python_executable`
   - `packages` (map of package -> version including `polars`, `numpy`, `scipy`, `statsmodels` if used)
   - `git_commit`, `git_branch`, `git_provenance` (use `"unknown"` and `"unavailable"` when git metadata cannot be obtained)
   - `requirements_file` (path to repository `requirements.txt` used)
   - `requirements_exact_file` (path to `outputs/requirements_exact_{RUN_TS}.txt` written during the run)
   - `bh_fdr_backend` and `bh_fdr_backend_version` (or `builtin` when using in-repo fallback)
   - `overlap_mode` and `entry_mode` (recorded verbatim)
  - `date_label_timezone` (string, e.g., `Asia/Kolkata` or `IST`)  # recommended field recording timezone used for labeled outputs

R-013 (NFR-004) — Scalability & Processing Strategy
 - Source: `spec.md` (NFR-004)
 - Text (short): Default in-memory polars; provide chunking/sharding and `--batch-size`/`--workers` CLI options; document scaling notes.
 - Acceptance Criteria:
   - CLI supports `batch_size` and `workers` config keys and runs on chunked batches producing intermediate files which are merged into final CSV.
 - Tasks:
   - T-013.1: Implement batch processing orchestration in `cli.py` with worker pool.
   - T-013.2: Add unit/integration test for batch merge correctness.
 - Code artifacts:
   - `src/sma_analysis/cli.py` (orchestrator)
 - Test artifacts:
   - `tests/integration/test_batch_processing.py`

SECTION C — CLI Contract & Exit Codes

R-014 (CLI Contract)
 - Source: `contracts/cli.md`
 - Text (short): The CLI is configuration-first; `python -m sma_analysis.cli run` with `config.yaml` present or `SMA_CONFIG` env var. Exit codes 0,1,2,3,4 defined.
 - Acceptance Criteria:
   - CLI returns prescribed exit codes for errors: 2 for invalid input, 3 for fatal data acquisition error (all tickers failed), 4 for smoke check failure.
 - Tasks:
   - T-014.1: Implement CLI entrypoint, config validation, exception-to-exit-code mapping.
   - T-014.2: Tests for exit codes using subprocess calls or function-level wrappers.
 - Code artifacts:
   - `src/sma_analysis/cli.py`
     - function: `main()` and `run_analysis(config)` (programmatic API)
 - Test artifacts:
   - `tests/unit/test_cli_exit_codes.py`
  - Add integration test: `test_cli_exits_with_4_and_logs_smoke_failures()` that intentionally fails a smoke check and asserts exit code 4 and presence of `logs/smoke_checks_{RUN_TS}.log`.

SECTION D — Tests, Quality Gates, and Smoke Checks

Global test plan (recommended):
 - Unit tests for small functions (SMA calc, signal detection, forward returns, drawdown, aggregation, BH-FDR, bootstrap CI).
 - Integration smoke test: run a tiny config (single ticker, single SMA pair, holding period 10, short date range) to assert files produced, manifest contains fields, and smoke checks pass.
 - CI: run unit tests and quick integration smoke test; integration full run optional in nightly pipeline.

Explicit smoke-checks (must be implemented and tested)

 - Existence: All required output files exist in the configured `output_dir` after a run (canonical CSV, diagnostics CSV, heatmaps, analysis_summary, run_manifest). Test: `test_smoke_check_files_exist()`.
 - CSV row coverage: The canonical CSV `sma_crossover_analysis_results_{RUN_TS}.csv` contains rows for every requested combination of (Stock, Period_Label, Short_SMA, Long_SMA, Holding_Period) — rows may be sentinel rows when `Num_Signals = 0`. Test: `test_smoke_check_csv_row_coverage()`.
 - Numeric validity: For rows with `Num_Signals_Used >= 5`, required numeric columns (`Mean_Return`, `Std_Dev_Return`, `P_Value`) must not be NaN and `P_Value` must be in [0,1]. For rows with `Num_Signals_Used < 5`, `P_Value = NA` and `Low_Power_Flag = TRUE`. Test: `test_smoke_check_numeric_validity()`.
 - Diagnostics present: `sma_crossover_analysis_diagnostics_{RUN_TS}.csv` must exist and contain any `Insufficient_History` flags discovered. Test: `test_smoke_check_diagnostics_present()`.

Note: smoke-check failures must be written to `logs/smoke_checks_{RUN_TS}.log` and cause the CLI to exit with code `4` per `contracts/cli.md`.

Suggested test files (full list):
- `tests/unit/test_data_download.py`
- `tests/unit/test_sma.py` (or `test_signals.py`)
- `tests/unit/test_metrics.py`
- `tests/unit/test_aggregate.py`
- `tests/unit/test_io.py`
- `tests/unit/test_viz.py`
- `tests/unit/test_report.py`
- `tests/unit/test_manifest.py`
- `tests/unit/test_cli_exit_codes.py`
- `tests/integration/test_smoke_run.py` (small config)

CSV header parity & casing guidance

 - The `contracts/csv-schema.md` file is the authoritative, case-sensitive source of truth for CSV headers and ordering. The RTM and tests MUST refer to the exact header names from that contract. Implementers should provide a small mapping between in-memory field names (snake_case) and CSV header names (case-sensitive) in the CSV writer to avoid casing mismatches. Add test: `test_csv_header_parity_with_contract()` that reads `contracts/csv-schema.md` and asserts exact header match.

Canonical required config keys (for validation tests)

 - Required keys to validate in `config.yaml` (and unit tests): `tickers`, `start_date`, `end_date`, `sma_pairs`, `holding_periods`, `output_dir`. Add tests: `test_cli_missing_required_keys_exits_2()` and `test_cli_invalid_sma_pair_exits_2()`.

Manifest & timezone note

 - The run manifest MUST include a `date_label_timezone` field (recommended value `Asia/Kolkata`) or an explicit note that date labels are interpreted as IST. This helps with reproducibility of labeled outputs and plots. Add to manifest schema under R-012: `date_label_timezone`.

RUN_TS canonical format

 - The canonical RUN_TS format is the UTC compact ISO `YYYYMMDDTHHMMSSZ` and MUST be used for primary outputs and the `run_manifest`. Implementations MAY accept legacy `YYYYMMDD_HHMMSS` for convenience, but the canonical format must be used when writing manifests and canonical filenames. Add test: `test_run_ts_format_in_manifest()`.

Quality gates mapping
 - Build: Not applicable (Python), but ensure `pip install -r requirements.txt` and `python -m pytest` pass.
 - Lint/Typecheck: Add `ruff` / `mypy` guidance in README; not implemented here.
 - Tests: Unit tests described above.

SECTION E — Traceability Matrix (Condensed view)

Below is a condensed matrix mapping requirement IDs to primary tasks, code files, and tests. Use this as quick lookup.

- R-001 → T-001.* → `src/sma_analysis/data.py` → `tests/unit/test_data_download.py`
- R-002 → T-002.* → `src/sma_analysis/utils.py` → `tests/unit/test_periods.py`
- R-003 → T-003.* → `src/sma_analysis/signals.py` → `tests/unit/test_signals.py`
- R-004 → T-004.* → `src/sma_analysis/metrics.py` → `tests/unit/test_metrics.py`
- R-005 → T-005.* → `src/sma_analysis/aggregate.py` → `tests/unit/test_aggregate.py`
- R-006 → T-006.* → `src/sma_analysis/aggregate.py` → `tests/unit/test_benchmark.py`
- R-007 → T-007.* → `src/sma_analysis/io.py` → `tests/unit/test_io.py`
- R-008 → T-008.* → `src/sma_analysis/viz.py` → `tests/unit/test_viz.py`
- R-009 → T-009.* → `src/sma_analysis/report.py` → `tests/unit/test_report.py`
- R-010 → T-010.* → `src/sma_analysis/cli.py` → `tests/integration/test_performance_smoke.py`
- R-011 → T-011.* → `src/sma_analysis/manifest.py` → `tests/unit/test_manifest.py`
- R-012 → T-012.* → `src/sma_analysis/manifest.py` + `cli.py` → `tests/unit/test_requirements_export.py`
- R-013 → T-013.* → `src/sma_analysis/cli.py` → `tests/integration/test_batch_processing.py`
- R-014 → T-014.* → `src/sma_analysis/cli.py` → `tests/unit/test_cli_exit_codes.py`

SECTION F — Implementation notes, risks and next steps

1) File skeletons & small scaffolding: create the `src/sma_analysis/` package with the modules listed above. Keep each module focused and small (single responsibility). This RTM assumes those files will be added during implementation.

2) Tests first: for core numeric logic (SMA, signals, forward returns, drawdown, bootstrap) implement unit tests first to lock down numerical behaviour and sentinel semantics.

3) BH-FDR backend: prefer `statsmodels`; include a deterministic builtin fallback covered by unit tests to ensure reproducibility.

4) Manifest & requirements export: on Windows the `pip freeze` command differs; CLI should run the python-invoked `python -m pip freeze` and write to `outputs/requirements_exact_{RUN_TS}.txt` to meet the spec.

5) Logging & diagnostics: ensure all diagnostics are written to `outputs/logs/` as CSVs and the `sma_crossover_analysis_diagnostics_{RUN_TS}.csv` summarizes issues for quick QA.

6) Small extras to implement now (low-risk):
   - Add `src/sma_analysis/__init__.py` exposing `run_analysis(config: dict)` to support programmatic calls from tests.
   - Add a script `scripts/smoke_run_example.py` with a minimal config for local verification.

SECTION G — Verification checklist (Before merging implementation)

- [ ] Unit tests for each module pass.
- [ ] Integration smoke run produces canonical CSV, heatmaps, analysis_summary, run_manifest, and `requirements_exact` file.
- [ ] Smoke checks pass and CLI exits with code 0 on success.
- [ ] CSV header and sentinel rules verified against `contracts/csv-schema.md`.
- [ ] BH-FDR backend recorded in manifest and parity test with statsmodels passes (if statsmodels used).

---

Appendix: Example task breakdown for R-005 (Aggregation & Stats)

- T-005.1 (design): Define aggregated row schema (column types, sentinel behaviours) — owner: architect — 0.5 day.
- T-005.2 (implement): `aggregate_signals()` — implement grouping, basic stats — owner: developer — 1.5 days.
- T-005.3 (implement): Bootstrap CI and p-value rules, small-sample handling, and sentinel emission — owner: developer — 1.0 day.
- T-005.4 (implement): BH-FDR module and integration with aggregator — owner: developer — 0.5 day.
- T-005.5 (tests & docs): Unit tests for all above, and update `contracts/csv-schema.md` if needed — owner: QA/Dev — 1.0 day.

---

End of RTM
