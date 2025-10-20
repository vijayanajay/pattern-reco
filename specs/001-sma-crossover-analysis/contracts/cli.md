# Contract: CLI & Programmatic API

This document describes the CLI contract and the minimal programmatic API for running the SMA crossover analysis. It defines exact CLI arguments, outputs, exit codes, the programmatic `run` signature, and the `run_manifest` schema used for reproducibility.

Canonical entrypoint

`python -m sma_analysis.cli run`

Notes

- All canonical outputs MUST use the compact UTC ISO timestamp format `RUN_TS = YYYYMMDDTHHMMSSZ` (e.g., `20251018T153045Z`) in filenames.
- CSV column names are case-sensitive and defined in `contracts/csv-schema.md` (this file is the source of truth for output column names, ordering, types, and sentinel values).

Configuration-driven CLI (no-arg execution)

The CLI is intentionally configuration-first. The canonical CLI invocation is:

```
python -m sma_analysis.cli run
```

The CLI MUST be executed without command-line arguments. All runtime inputs are supplied via a configuration file in YAML or JSON format. By default the CLI looks for a `config.yaml` file in the current working directory. The path may be overridden by setting the environment variable `SMA_CONFIG` to a full path to a config file (this is the only supported runtime override; the CLI must function with no arguments).

Configuration file schema (keys)

The config file must contain the same keys that were previously available as CLI flags. Example `config.yaml` structure (YAML):

```
tickers: ["YESBANK.NS", "RELIANCE.NS"]
start_date: "2004-01-01"
end_date: "2023-12-31"
sma_pairs: [[5,20],[5,50],[10,50]]
holding_periods: [10,15,20]
overlap_mode: "overlapping"
entry_mode: "entry_on_T"  # allowed values: "entry_on_T" (default), "entry_next_open"
seed: 42
output_dir: "outputs/"
batch_size: 50
workers: 4
bootstrap_resamples: 1000
na_threshold: 10.0
write_latest: false
bh_fdr_backend: "statsmodels"
verbose: true
```

Config keys (required vs optional)

The config file must include the following required keys:

- `tickers` (list[str]) — required
- `start_date` (YYYY-MM-DD) — required
- `end_date` (YYYY-MM-DD) — required
- `sma_pairs` (list[list[int,int]]) — required (each pair must satisfy Short < Long)
- `holding_periods` (list[int]) — required
- `output_dir` (str) — required

Optional keys (with defaults if omitted):

- `overlap_mode` (str) = "overlapping"
- `entry_mode` (str) = "entry_on_T"  # default: use cleaned Adj Close on T
- `seed` (int) = 42
- `batch_size` (int) = 50
- `workers` (int) = 4
- `bootstrap_resamples` (int) = 1000
- `na_threshold` (float) = 10.0
- `write_latest` (bool) = false
- `bh_fdr_backend` (str) = "statsmodels" (fallback to builtin allowed)
- `verbose` (bool) = false

If any required key is missing or malformed, the CLI should treat this as a configuration validation error and exit with code `2` after writing a helpful log message.

Note on na_threshold units

Note: `na_threshold` is expressed as a percentage (for example, `10.0` = 10%). If your configuration uses fractions instead (e.g., `0.10` for 10%), provide `na_threshold_fraction` and the implementation should support reading either key (preferring `na_threshold` percent when both are present) to preserve backward compatibility.

Notes:

- `config.yaml` is the canonical configuration location. The CLI must succeed when executed with no arguments and a valid `config.yaml` is present in the working directory.
- The environment variable `SMA_CONFIG` may be set to a path to an alternate config file if callers need to override the default location. This is optional and primarily intended for CI or containerized runs; the CLI should still work without this variable.
- Validation errors in the config (missing required keys, invalid SMA pairs, invalid date ranges) are considered input validation failures and should cause the process to exit with code `2` and write a helpful message to the logs.


Exit codes (process exit codes)

- `0`: Success and smoke checks passed.
- `1`: Uncaught exception / runtime error.
- `2`: Invalid input arguments (e.g., invalid SMA pair format, invalid dates, Short >= Long).
- `3`: Fatal data acquisition error (all tickers failed to download or no usable data for any requested period).
- `4`: Smoke checks failed (files missing, NaNs in required columns for rows with `Num_Signals_Used > 0`, p-values outside [0,1], or required outputs missing).

Exception mapping (recommended)

- Input validation (e.g., missing required keys, invalid SMA pairs, date range errors): raise `ValidationError` -> CLI exit code `2`.
- Data acquisition failures (no usable data for any ticker / all tickers failed): raise `DataAcquisitionError` -> CLI exit code `3`.
- Smoke check failures after artifact writes: raise `SmokeCheckFailure` (include log path) -> CLI exit code `4`.
- Uncaught exceptions: exit code `1`.

Outputs (files written to `--output-dir`; all timestamped with `RUN_TS`)

- `sma_crossover_analysis_results_{RUN_TS}.csv` — canonical aggregated results (columns and order defined in `contracts/csv-schema.md`).
- `sma_crossover_analysis_diagnostics_{RUN_TS}.csv` — per-ticker diagnostics and warnings (Insufficient_History flags, percent_missing_before_fill, na_filled_count, warmup_drop_counts).
- `heatmap_avg_mean_return_{RUN_TS}.png` — visual heatmap of average Mean_Return across SMA pairs (300 DPI).
- `analysis_summary_{RUN_TS}.md` — markdown summary containing Executive Summary, Heatmap reference, Consistency Analysis, Top Performers, Fallen Angel case (YESBANK.NS), Benchmark Comparison, Signal Frequency Analysis.
- `run_manifest_{RUN_TS}.json` — reproducibility manifest (see `Run manifest schema` below).
- `logs/na_fill_warnings_{RUN_TS}.csv`
- `logs/warmup_drop_counts_{RUN_TS}.csv`
- `logs/smoke_checks_{RUN_TS}.log` — detailed smoke-check output when smoke checks fail.

Programmatic API (minimal contract)

To support programmatic usage and testing, implement and export a function with the following signature in `src/sma_analysis/cli.py` (or package module):

```
def run_analysis(config: dict) -> dict:
	"""
	Run the SMA crossover analysis with the provided configuration.

	Args:
		config: dict with keys matching the CLI flags (see required keys below).

	Required config keys (example shape):
		{
			"tickers": ["YESBANK.NS", "RELIANCE.NS"],
			"start_date": "2004-01-01",
			"end_date": "2023-12-31",
			"sma_pairs": [[5,20],[5,50]],
			"holding_periods": [10,15,20],
			"overlap_mode": "overlapping",
			"entry_mode": "entry_on_T",
			"seed": 42,
			"output_dir": "outputs/",
			"bootstrap_resamples": 1000,
			"na_threshold": 10.0,
			"bh_fdr_backend": "statsmodels",
			"workers": 4,
			"batch_size": 50,
		}

	Returns:
		dict: {"results_csv": <path>, "diagnostics_csv": <path>, "heatmap_png": <path>, "analysis_summary": <path>, "run_manifest": <path>, "logs": { ... }, "stats": { ... }}

	On fatal error, raise RuntimeError with context and let the CLI wrapper set the appropriate exit code.
	"""


```

Run manifest schema

The `run_manifest_{RUN_TS}.json` is required for reproducibility and MUST contain at least the following keys (example values):

```
{
  "run_ts": "20251018T153045Z",
  "python_version": "3.11.4",
  "polars_version": "0.19.0",
  "numpy_version": "2.2.0",
  "scipy_version": "1.11.0",
  "bh_fdr_backend": "statsmodels",
  "bh_fdr_backend_version": "0.21.0",
  "git_commit": "abcdef1234567890",
  "git_branch": "001-sma-crossover-analysis",
  "rng_seed": 42,
  "os": "Windows-10-10.0.19045",
  "cli_args": {
	"tickers": ["YESBANK.NS"],
	"sma_pairs": [[5,20]],
	"holding_periods": [10,15,20]
  }
}
```

The concrete list of keys and package versions MUST be produced at runtime and written to the manifest.

Smoke checks (performed after artifacts are written)

Smoke checks MUST run automatically and cause exit code `4` if any fail. Required smoke checks:

1. Existence: All required output files exist in `--output-dir`.
2. CSV rows: The canonical CSV `sma_crossover_analysis_results_{RUN_TS}.csv` contains rows for every requested combination of (Stock, Period_Label, Short_SMA, Long_SMA, Holding_Period) — rows may be sentinel rows when Num_Signals = 0.
3. Numeric validity: For rows with `Num_Signals_Used >= 5`, required numeric columns (`Mean_Return`, `Std_Dev_Return`, `P_Value`) must not be NaN and `P_Value` must be in [0,1]. For rows with `Num_Signals_Used < 5`, the implementation MUST set `P_Value = NA` and `Low_Power_Flag = TRUE` (see `research.md`).
4. Diagnostics present: `sma_crossover_analysis_diagnostics_{RUN_TS}.csv` must exist and contain any `Insufficient_History` flags discovered.

If smoke checks fail, write `logs/smoke_checks_{RUN_TS}.log` with detailed failures and exit with code 4.

Notes about errors and partial failures

- If some tickers fail to download but at least one ticker yields usable data, the run should continue; per-ticker failures must be recorded in diagnostics and logs. Only if all tickers fail should the CLI exit with code `3`.
- Input validation errors (invalid SMA pair, wrong dates, or empty tickers) should be reported and return exit code `2`.

