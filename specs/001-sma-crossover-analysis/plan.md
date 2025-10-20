# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Deliver a reproducible SMA crossover signal efficacy analysis that computes signals for 10 canonical SMA pairs across a 20-year window sliced into four 5-year cohorts. The implementation will use vectorized Polars pipelines to download, clean, compute SMAs, generate signals (overlapping and non-overlapping modes), compute forward returns and drawdowns for 10/15/20 trading-day holding periods, aggregate statistics (including t-tests and bootstrap CIs), produce heatmaps, and emit a human-readable `analysis_summary.md`. Outputs are timestamped and include diagnostic logs and a run manifest recording environment and git provenance.

## Technical Context

**Language/Version**: Python 3.11 (need to record exact minor version at run time).
**Primary Dependencies**: polars (primary DataFrame engine), numpy, scipy, matplotlib, seaborn, yfinance (or a small download shim), tqdm (optional) — exact versions pinned in `requirements.txt` and recorded in run manifest.
**Storage**: Filesystem-based outputs under an overridable `output_dir` (default: `outputs/`). No DB required.
**Testing**: `pytest` for unit and integration tests; include small numerical tolerance tests for SMA/returns/drawdown.
**Target Platform**: Developer laptop (Windows and Linux supported). The run manifest will record OS; default CI target is Linux but development and quick-runs on Windows/cmd are supported.
**Project Type**: Single Python analysis package with a small CLI entrypoint `src/sma_analysis/cli.py`. The CLI is configuration-first: it reads runtime inputs from a `config.yaml` (or JSON) file and is executed without command-line arguments. An environment variable `SMA_CONFIG` may override the config file path.
**Performance Goals**: A canonical run (~15 tickers, 10 SMA pairs, 3 holding periods) should complete within ~30 minutes on a 4-core, 16GB laptop when using vectorized polars operations and optional batching/workers. `batch_size` and `workers` are configured via `config.yaml`.
**Constraints**: Must accept explicit input/output paths; deterministic bootstrap seed; no hidden defaults — configuration declared in `config.yaml` (or via the `SMA_CONFIG` environment variable) and recorded in `run_manifest`.
**Scale/Scope**: Designed for up to ~200 tickers with batch processing; default UX targets 10-20 tickers for fast iteration.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

This feature must follow the repository constitution (`.specify/memory/constitution.md`). Key gates to evaluate now:

- Gate 1 (Minimalism): Implementation must avoid extra features beyond the described analysis. -> Compliant (plan focuses on core SMA analysis, heatmaps, and summary).
- Gate 2 (One responsibility): Each module (data, signals, aggregation, viz, reports) will be single-responsibility. -> Planned.
- Gate 3 (Test-First): Unit tests will be added for SMA, signal detection, forward returns/drawdown, aggregation, and CSV/PNG outputs. -> Planned (tests/ unit/ files).
-- Gate 4 (Reproducible paths): CLI reads explicit input/output paths from `config.yaml`; defaults provided but overridable via `SMA_CONFIG`. -> Planned.
- Gate 5 (Deterministic): Bootstrap RNG seed default 42; recorded in manifest. -> Planned.
- Gate 6 (Explicit Data Hygiene): All NaN fills, warmup drops, and warnings will be logged to `outputs/logs/` and included in diagnostics CSV. -> Planned.
- Gate 7 (Fail Fast on Ambiguity): Signals without N trading days will be discarded with clear reason. -> Planned.
- Gate 8 (No Hidden Defaults): All defaults in `config.yaml` and documented in `analysis_summary.md`. -> Planned.

Decision: BH-FDR backend and dependency policy

We will prefer `statsmodels.stats.multitest.multipletests(method='fdr_bh')` as the BH-FDR implementation. For reproducibility, `statsmodels` should be added to `requirements.txt` and pinned for canonical runs. If `statsmodels` is unavailable in a runtime, the code will fall back to a small in-repo BH-FDR implementation that produces the same results for identical inputs; the backend used (`statsmodels` or `builtin`) and its version MUST be recorded in the `run_manifest` under `bh_fdr_backend`.

Canonical RUN_TS format

All canonical filenames produced by runs MUST use UTC compact ISO format: `YYYYMMDDTHHMMSSZ` (for example, `20251018T153045Z`). Helper utilities may support `YYYYMMDD_HHMMSS` for legacy scripts, but the canonical format is the ISO compact form and is used in `run_manifest` and all primary outputs.

Test-first & Smoke checks (expanded)

Unit and integration tests will be created before or alongside feature implementation for the following behaviors (non-exhaustive):

- SMA calculation and full-window warmup (verify that SMA values with `min_periods=window` are only produced after the warmup and that warmup rows are counted and logged).
- Signal detection rule including T-1 availability checks (SMA_short[T], SMA_long[T], SMA_short[T-1], and SMA_long[T-1] must exist to consider a candidate signal).
- Overlap modes: `overlapping` and `non_overlapping` behaviors with exact definition: `non_overlapping` suppresses any candidate signal that occurs within the holding period (N trading days) of a previously accepted signal for the same (Ticker, SMA pair, Period).
- Forward returns and drawdown calculation and discarding signals lacking sufficient forward data.
- Aggregation correctness and sentinel row behavior for zero-signal groups.
- BH-FDR behavior using `statsmodels` (or builtin) on synthetic p-values.
- Bootstrap percentile CI reproducibility given seed and resample count.

Smoke checks (run-time): the CLI will run a small battery of validations after artifacts are written and before exit. These checks are listed in `contracts/cli.md` and include:

- existence of canonical outputs
- deterministic row counts in the canonical CSV
- numeric validity checks for rows with `Num_Signals_Used > 0` (no NaNs in essential columns)

Note: Failures in smoke checks should surface with a non-zero exit code (exit code 4) and be logged to `outputs/logs/smoke_checks_{RUN_TS}.log` for debugging.

No gate violations detected at plan-time. Any future deviations must be justified in Complexity Tracking.
## Project Structure

### Documentation (this feature)

```
specs/001-sma-crossover-analysis/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
└── tasks.md
```

### Source Code (repository root)

```
src/
├── sma_analysis/
│   ├── data.py           # downloading & cleaning time series
│   ├── signals.py        # SMA computation & signal rules
│   ├── metrics.py        # forward returns, drawdown, aggregation
│   ├── viz.py            # heatmaps & focal ticker plots
│   ├── cli.py            # CLI entrypoint
│   └── io.py             # CSV/manifest writers
tests/
├── unit/
│   ├── test_sma.py
│   ├── test_signals.py
│   ├── test_metrics.py
│   ├── test_cli_exit_codes.py        # test mapping of errors -> exit codes and smoke checks
│   └── test_run_manifest.py         # verify manifest keys and fallback behavior
```

**Structure Decision**: Single Python analysis package under `src/sma_analysis/` keeps code compact and focused per the Constitution's Minimalism principle.
**Structure Decision**: [Document the selected structure and reference the real
directories captured above]

## Complexity Tracking

No constitution violations requiring justification were identified at plan-time. If we later add distributed processing or external databases, we will add entries here explaining the trade-offs and why simpler in-repo batch processing would be insufficient.

