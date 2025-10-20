# Tasks: SMA Crossover Signal Efficacy Analysis

This file is generated from `plan.md` and `spec.md` and organized by phase. Each task is independently actionable and follows the strict checklist format required by the project.

## Phase 1 — Setup

- [ ] T001 Create Python virtual environment and record activation instructions in `README.md` (repo root) — `README.md`
- [ ] T002 Install pinned dependencies from `requirements.txt` into the environment and verify imports for `polars`, `numpy`, and `scipy` (document results in `outputs/setup/check_install.txt`) — `requirements.txt`
- [ ] T003 Create `outputs/` and `outputs/logs/` directories with correct .gitignore entries and a sample `.gitkeep` to preserve dirs — `outputs/` and `outputs/logs/`
- [ ] T004 Add a sample `config.yaml` with canonical SMA pairs, holding periods, date range, and default `output_dir` (template) — `specs/001-sma-crossover-analysis/config.template.yaml`
- [ ] T005 Add CLI run instruction to `specs/001-sma-crossover-analysis/quickstart.md` and verify it references `config.template.yaml` — `specs/001-sma-crossover-analysis/quickstart.md`

## Phase 2 — Foundational (blocking prerequisites)

- [ ] T006 Create package structure `src/sma_analysis/__init__.py` (module init) — `src/sma_analysis/__init__.py`
- [ ] T007 Implement data downloader & cleaner skeleton `src/sma_analysis/data.py` with placeholder functions and docstrings (download, forward-fill, retry wrapper) — `src/sma_analysis/data.py`
- [ ] T008 Implement SMA & signal utilities skeleton `src/sma_analysis/signals.py` with functions to compute rolling SMAs and detect crossover candidates (requires full-window check) — `src/sma_analysis/signals.py`
- [ ] T009 Implement metrics skeleton `src/sma_analysis/metrics.py` with placeholders for forward returns, max drawdown, aggregation, bootstrap CI, and BH-FDR API (docstring-only for now) — `src/sma_analysis/metrics.py`
- [ ] T010 Implement IO utilities `src/sma_analysis/io.py` for CSV/PNG writers and run_manifest writer (skeleton) — `src/sma_analysis/io.py`
- [ ] T011 Add CLI entrypoint `src/sma_analysis/cli.py` that reads `config.yaml` (env override `SMA_CONFIG`) and exposes `run()` function (skeleton) — `src/sma_analysis/cli.py`
- [ ] T012 Create tests directory `tests/unit/` and add `tests/unit/test_smoke.py` that imports top-level modules to validate importability — `tests/unit/test_smoke.py`
- [ ] T013 Add `pyproject.toml` or minimal metadata to allow `python -m sma_analysis.cli` invocation (if missing) — `pyproject.toml`
- [ ] T036 Implement config and SMA-pair validation (Short < Long, uniqueness) and map validation errors to CLI exit code 2 with clear error messages — `src/sma_analysis/cli.py`

## Phase 3 — [US1] Researcher cohort analysis (Priority: P1)

Goal: Implement the core analysis pipeline that produces the canonical CSV and diagnostics for a single stock and single SMA pair (independently testable increment).

Independent test criteria: Running the CLI with `config.template.yaml` modified for a single ticker produces `outputs/sma_crossover_analysis_results_{RUN_TS}.csv` with required columns and at least one non-empty row.

- [ ] T014 [US1] Implement cohort slicing utility and canonical cohort definitions (2004-01-01..2023-12-31 split into four 5-year cohorts) — `src/sma_analysis/data.py`
- [ ] T015 [US1] Implement SMA computation function that enforces full-window min_periods and returns warmed `polars` frames — `src/sma_analysis/signals.py`
- [ ] T016 [US1] Implement crossover signal detection rule (SMA_short[T] > SMA_long[T] AND SMA_short[T-1] <= SMA_long[T-1]) with warmup checks — `src/sma_analysis/signals.py`
- [ ] T017 [US1] Implement forward return and max drawdown computation for holding periods [10,15,20] with discard behavior when insufficient forward rows exist — `src/sma_analysis/metrics.py`
- [ ] T018 [US1] Implement per-signal `SignalRecord` emitter and diagnostics notes (including `discarded_insufficient_forward_data`) — `src/sma_analysis/metrics.py`
- [ ] T019 [US1] Implement aggregation to `AggregatedMetric` rows (Num_Signals, Num_Signals_Used, Mean_Return, Std_Dev_Return, Win_Rate, Median_Return, Skew, Kurtosis) and write canonical CSV writer — `src/sma_analysis/metrics.py` and `src/sma_analysis/io.py`
- [ ] T020 [US1] Implement percentile bootstrap (1000 resamples, RNG seed from config, default 42) for Mean_CI_lower and Mean_CI_upper — `src/sma_analysis/metrics.py`
- [ ] T021 [US1] Implement BH-FDR adjustment function (use `statsmodels` when available, else builtin) and ensure `Adj_P_Value_BH` per holding period is computed — `src/sma_analysis/metrics.py`
- [ ] T022 [US1] Write run manifest (versions, git commit/branch fallback to `unknown`, RNG seed, inputs) and save `outputs/run_manifest_{RUN_TS}.json` — `src/sma_analysis/io.py`
- [ ] T023 [US1] Add unit tests for SMA correctness, signal detection, forward returns, and drawdown (happy path + edge: insufficient forward data) — `tests/unit/test_sma_signals.py`
- [ ] T024 [P] [US1] Add integration smoke test that runs CLI for a single ticker and verifies CSV + diagnostics exist and contain expected columns — `tests/integration/test_single_ticker_run.py`
- [ ] T037 [US1] Implement `overlap_mode` handling (`overlapping` and `non_overlapping`) in signal acceptance logic and document behavior in `analysis_summary.md` and manifest — `src/sma_analysis/signals.py`
- [ ] T038 [US1] Ensure aggregator emits audit columns `T_Stat` and `DF` for each aggregated row and enforces sentinel row semantics (Num_Signals=0 etc.) per `data-model.md` and `contracts/csv-schema.md` — `src/sma_analysis/metrics.py` and `src/sma_analysis/io.py`
- [ ] T039 [US1] Add buy-and-hold benchmark computation into aggregation pipeline and include `Buy_and_Hold_Return` column in canonical CSV — `src/sma_analysis/aggregate.py`
 - [ ] T046 [US1] Add unit tests for metrics: `tests/unit/test_metrics.py` covering forward returns, max drawdown, and bootstrap CI reproducibility — `tests/unit/test_metrics.py`
 - [ ] T047 [US1] Add unit tests for aggregation: `tests/unit/test_aggregate.py` covering sentinel rows, p-value/T_Stat/DF recording, and BH-FDR application parity — `tests/unit/test_aggregate.py`

## Phase 4 — [US2] Heatmap overview (Priority: P2)

Goal: Produce per-holding-period heatmaps (unweighted and weighted-by-signals) saved at 300 DPI and masked for untested cells.

Independent test criteria: Running the report writer against aggregate CSV produces `outputs/heatmap_avg_mean_return_10d_{RUN_TS}.png` and `outputs/heatmap_weighted_by_signals_10d_{RUN_TS}.png` with axes labeled and a caption.

- [ ] T025 [US2] Implement visualization module `src/sma_analysis/viz.py` with functions to build seaborn heatmaps and mask non-tested SMA pairs — `src/sma_analysis/viz.py`
- [ ] T026 [US2] Implement heatmap writer that consumes aggregated CSV and writes PNGs at 300 DPI per holding period — `src/sma_analysis/viz.py` and `src/sma_analysis/io.py`
- [ ] T027 [P] [US2] Add unit/integration test verifying heatmap files are created and non-empty given a small synthetic aggregated CSV — `tests/unit/test_viz.py`

## Phase 5 — [US3] Analysis summary report (Priority: P3)

Goal: Produce `analysis_summary.md` synthesizing executive summary, heatmap references, Fallen Angel (`YESBANK.NS`) subsection, and top performers table.

Independent test criteria: Running the report writer after generating CSV and PNG artifacts produces `outputs/analysis_summary_{RUN_TS}.md` that references existing PNG/CSV files and contains the reproducibility header.

- [ ] T028 [US3] Implement report writer `src/sma_analysis/report.py` that consumes CSV and PNGs to assemble `analysis_summary.md` with required sections — `src/sma_analysis/report.py`
- [ ] T029 [US3] Implement focal ticker plot function and `YESBANK.NS` signals visualization writer `src/sma_analysis/viz.py` (or separate file) that annotates signals on time-series PNG — `src/sma_analysis/viz.py`
- [ ] T030 [US3] Add test that verifies `analysis_summary_{RUN_TS}.md` is generated and contains reproducibility header fields (python_version, packages, git_commit) when given run_manifest and artifacts — `tests/unit/test_report.py`

## Final Phase — Polish & Cross-cutting concerns

- [ ] T031 Update `requirements.txt` with exact pins for numeric libs and add `requirements_exact_template.txt` generation instruction in `src/sma_analysis/io.py` manifest writer — `requirements.txt`
- [ ] T032 Add logging and `outputs/logs/na_fill_warnings_{RUN_TS}.csv` and `warmup_drop_counts_{RUN_TS}.csv` writers and ensure diagnostics CSV `outputs/sma_crossover_analysis_diagnostics_{RUN_TS}.csv` is produced — `src/sma_analysis/data.py` and `src/sma_analysis/io.py`
- [ ] T033 Add CI job configuration snippet (GitHub Actions) to run `pytest` for unit and integration tests and to run the smoke CLI test (example workflow in `.github/workflows/sma-analysis-ci.yml`) — `.github/workflows/sma-analysis-ci.yml`
- [ ] T034 Add developer docs section in `specs/001-sma-crossover-analysis/README.md` describing how to run full canonical analysis and how to reproduce exact env (include `python -m pip freeze | sort > outputs\requirements_exact_{RUN_TS}.txt` for cmd.exe) — `specs/001-sma-crossover-analysis/README.md`
- [ ] T035 [P] Implement small BH-FDR parity test comparing builtin implementation to `statsmodels` (if present) and add unit test `tests/unit/test_bh_fdr_parity.py` — `tests/unit/test_bh_fdr_parity.py`
- [ ] T040 Implement CSV header parity test that reads `contracts/csv-schema.md` and asserts exact header and sentinel semantics during CSV write (`tests/unit/test_io.py`) — `tests/unit/test_io.py`
- [ ] T041 Add unit tests for data download and retry behavior (`tests/unit/test_data_download.py`) including mock yfinance failures and `outputs/logs/download_failures_{RUN_TS}.csv` creation — `tests/unit/test_data_download.py`
- [ ] T042 Implement CLI exit-code mapping (invalid input -> 2, fatal data acquisition -> 3, smoke checks fail -> 4) and coverage via tests `tests/unit/test_cli_exit_codes.py` — `src/sma_analysis/cli.py` and `tests/unit/test_cli_exit_codes.py`
- [ ] T043 Add `batch_size` and `workers` CLI/config support and orchestration in `src/sma_analysis/cli.py` (implement chunking + worker pool) — `src/sma_analysis/cli.py`
- [ ] T044 Add integration test for batch processing and merge correctness (`tests/integration/test_batch_processing.py`) — `tests/integration/test_batch_processing.py`
- [ ] T045 Ensure run manifest writer exports exact requirements file `outputs/requirements_exact_{RUN_TS}.txt` (use `python -m pip freeze`) and add test `tests/unit/test_requirements_export.py` — `src/sma_analysis/manifest.py` and `tests/unit/test_requirements_export.py`
 - [ ] T048 Add unit test for run manifest: `tests/unit/test_manifest.py` verifying manifest fields, versions, git fallback, and `date_label_timezone` — `tests/unit/test_manifest.py`
 - [ ] T049 Add integration performance smoke test: `tests/integration/test_performance_smoke.py` (time-bounded run on small dataset asserting upper-bound performance for SMA/aggregation) — `tests/integration/test_performance_smoke.py`

## Dependencies (story completion order)

1. Foundational phase (T006..T013) MUST complete before story phases (T014..T024, T025..T027, T028..T030).
2. [US1] (T014..T024) should be delivered first (MVP). [US2] and [US3] depend on the canonical aggregated CSV produced by [US1].
3. BH-FDR (T021) must run before final aggregation outputs and `Adj_P_Value_BH` population used by reports (T019, T022, T028).

## Parallel execution examples

- Example A (parallelizable): T025 (viz) and T028 (report assembly) can run in parallel once aggregated CSV and PNGs exist. Marked tasks include `[P]` where safe.
- Example B (parallelizable): Unit tests T023, T027, T030, and parity test T035 can run in parallel with each other and with non-blocking IO tasks.

## Implementation strategy (MVP first)

- MVP scope: Implement Phase 1, Phase 2, and Phase 3 [US1] tasks (T001..T024). This produces a canonical CSV and basic diagnostics and satisfies the primary research goal.
- Iterative delivery: After MVP, implement visualizations (Phase 4) and report writer (Phase 5). Polish tasks (Phase Final) follow.

## Format validation checklist

- All tasks in this file start with `- [ ]` and contain a Task ID (T###).  
- Story-labeled tasks include `[US1]`, `[US2]`, or `[US3]` as required.  
- Parallelizable tasks include `[P]` when they operate on different files and don't depend on incomplete tasks.

---

Generated on: 2025-10-20
