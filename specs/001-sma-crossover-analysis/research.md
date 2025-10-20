# Research: SMA Crossover Analysis - Decisions & Rationale

This document resolves open choices from the spec and plan and records decisions, rationale, and alternatives.

---

Decision: Use Python 3.11 as runtime; record exact minor version (e.g., 3.11.x) in run manifest.

Rationale: Python 3.11 offers performance improvements and wide package support. The spec's NFRs require pinned numeric libs; recording the exact Python minor version in the manifest ensures reproducibility.

Alternatives considered: Use Python 3.10 for wider default CI images; rejected because 3.11 performance and future-proofing are preferred.

---

Decision: Use `polars` as the primary DataFrame engine for cleaning, SMA rolling, and aggregations. Use `numpy` and `scipy` for numerical/stats operations. Implement bootstrapping with `numpy` array operations.

Rationale: `polars` is fast, memory-efficient, and aligns with the project's vectorized processing goals (NFR-001). It also enables straightforward per-column rolling operations and quick CSV writes.

Alternatives considered: `pandas` (more familiar) but `polars` preferred for performance and memory; provide a small compatibility shim if downstream tools expect `pandas` (export to pandas when needed).

---

Decision: Use `yfinance` for initial prototype downloads with a small retry/backoff wrapper in `src/sma_analysis/data.py`.

Rationale: `yfinance` is simple to use, widely available, and sufficient for historical Adjusted Close series. The wrapper will implement the 3-attempt exponential backoff and produce per-ticker failure logs.

Alternatives considered: Use a paid/enterprise historical data API (AlphaVantage, Tiingo) — rejected for prototype due to API keys and rate limits. Provide an adapter interface (DataProvider) so swapping is straightforward.

---

Decision: For bootstrap 95% CI, implement percentile bootstrap (1000 resamples, RNG seed=42) using `numpy.random.default_rng(seed).choice` on the per-signal returns array and compute mean distribution percentiles (2.5, 97.5).

Rationale: The spec requests percentile bootstrap with fixed seed; using numpy RNG is deterministic and fast for 1000 resamples.

Alternatives considered: BCa intervals (more accurate) — document as follow-up.

---

Decision: Use `scipy.stats.ttest_1samp` (two-sided by default) for P_Value computation and record t-statistic and df in the aggregation CSV. For Num_Signals_Used < 5, set P_Value=NA and Low_Power_Flag=True.

Rationale: Matches spec and Constitution's requirement to use standard tests and report test stats.

Alternatives considered: non-parametric sign test — include as an exploratory check if normality assumptions appear violated.

---

Decision: Visualization via `matplotlib` + `seaborn` heatmap (divergent colormap centered at 0). Save at 300 DPI and annotate masked/un-tested cells as grey.

Rationale: Stable, well-known libraries; seaborn heatmap supports masking and annotations.

Alternatives considered: plotly for interactive visuals — deferred as out-of-scope (non-goal: no extra UI features).

---

Decision: File naming timestamp format: RUN_TS = UTC compact ISO `YYYYMMDDTHHMMSSZ` (e.g., 20251018T153045Z) for canonical outputs and run_manifest. Provide optional stable `_latest` copies.

Rationale: Matches spec explicit requirement for compact UTC timestamp.

Alternatives considered: `YYYYMMDD_HHMMSS` — support both in tooling to satisfy existing notes; canonical default is the ISO compact format.

---

Decision: CLI entrypoint `python -m sma_analysis.cli run` executed without command-line arguments and driven entirely by a configuration file (default `config.yaml`) that contains keys for tickers, date range, SMA pairs, holding periods, overlap/entry modes, RNG seed, output_dir, batch_size and workers. The environment variable `SMA_CONFIG` may be used to override the config file path for CI or containerized runs.

Rationale: Clear reproducible command; aligns with Run contract.

Alternatives considered: purely programmatic API — still supported, but CLI ensures reproducibility and easy automation.

---

Decision: Implement `--workers` parallelism at per-batch level using Python multiprocessing (process pool) to keep GIL interference minimal. Default `--workers=4`, `--batch-size=50`.

Rationale: Polars is fast but downloads and per-ticker operations can benefit from multiprocessing for IO-bound steps.

Alternatives considered: ThreadPool for downloads — rejected due to GIL and potential CPU-bound steps; prefer process-based parallelism.

---

Decision: Tests: write pytest unit tests for SMA calculation, signal detection rules, forward-return/discard behavior, drawdown computation, aggregation including BH-FDR implementation, and bootstrap CI correctness (using small synthetic arrays for deterministic checks).

Rationale: Constitution requires test-first for core computations.

Alternatives considered: property-based tests — nice but deferred to follow-up.

---

Decision: BH-FDR adjustment: use `statsmodels.stats.multitest.multipletests(method='fdr_bh')` when available; if `statsmodels` is not desired as a hard dependency, implement a simple BH-FDR function in `metrics.py` and pin `statsmodels` as optional.

Rationale: `statsmodels` provides a robust, tested implementation; adding it as a dependency increases reproducibility and clarity.

Alternatives considered: custom implementation (simple to implement) — prefer `statsmodels` if acceptable; otherwise include a small in-repo BH-FDR implementation.

BH-FDR fallback and determinism

If `statsmodels` is not available at runtime the repository will include a deterministic in-repo Benjamini-Hochberg implementation in `src/sma_analysis/metrics.py`. This builtin implementation MUST follow the standard BH algorithm (sort p-values, compute adjusted p-values by scaling and taking cumulative minima, etc.) and be covered by a unit test that asserts parity with `statsmodels.stats.multitest.multipletests(method='fdr_bh')` for representative p-value vectors. The run manifest MUST record `bh_fdr_backend` and `bh_fdr_backend_version` (or `builtin`) so consumers know which backend was used.

---

Decision: Diagnostics and logs: write `outputs/logs/na_fill_warnings_{RUN_TS}.csv` and `warmup_drop_counts_{RUN_TS}.csv`. Also create `outputs/sma_crossover_analysis_diagnostics_{RUN_TS}.csv` that aggregates per-ticker warnings.

Polars and multiprocessing guidance

When using multiprocessing alongside Polars, be careful to avoid CPU oversubscription which can degrade performance and make runtime behavior non-deterministic. Recommended practices:

- Set Polars thread count inside worker processes (for example, call `polars.set_num_threads(1)` at worker process start) or set the environment variable `POLARS_MAX_THREADS=1` for worker processes.
- Alternatively, control threading centrally with `polars.set_num_threads()` in the main process and choose a conservative `--workers` value so that total threads across processes does not exceed available cores.
- Document and expose a small config option or environment guidance so CI and users can reproduce performance: e.g., `POLARS_MAX_THREADS` and `WORKER_PROCESSES` settings.

Rationale: Matches spec and Constitution.

---

Decision: Report reproducibility header: include package versions resolved at runtime (from importlib.metadata), python executable path, git commit hash (from `git rev-parse HEAD`) and branch (from environment or git), RNG seed, OS, and `requirements.txt` path.

Rationale: Required by spec and Constitution.

Git provenance fallback

When git metadata is not available (for example, in a packaged release or CI environment without an attached `.git` directory), the manifest writer should not fail. Implementations SHOULD attempt to retrieve git metadata using `git` commands inside a try/except block; if retrieval fails, set `git_commit` and `git_branch` to the literal string `unknown` and add a manifest flag `git_provenance = "unavailable"` so consumers can detect the absence of repository provenance.

---

Decision: Default SMA pair validation: enforce Short < Long and uniqueness; raise a CLI error if invalid.

Rationale: Protects against invalid inputs and matches spec.

---

Next steps (Phase 1): build `data-model.md` describing entities, produce OpenAPI-like `contracts/` entries for the CLI and programmatic API, and write `quickstart.md` with the minimal reproducible command and smoke checks.
