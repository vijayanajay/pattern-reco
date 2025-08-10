# MVP Architecture — Pattern Detection on Indian Stocks (Kailash-style)

Scope
- Daily bars only. yfinance only. No intraday, no indicators, no fundamentals.

Principles
- Minimal surface: one CLI, one YAML config, one pipeline.
- Deterministic: frozen snapshots, exact splits, seeds, SHA256 everywhere.
- Testable: thin end-to-end slice first; iterate.
- Small knobs: detectors ≤ 3 params; IS-only fitting; strict OOS and final holdout.

System
- Config: single YAML drives data, universe, WF, detector params, costs, outputs.
- Data: fetch to Parquet with metadata; runs use only frozen snapshots; refresh emits diffs.
- Features: returns, gaps, minimal rolling stats; vectorized; explicit NaN handling.
- Detector: gap_z (MVP). Pluggable registry.
- Rules/Exec: next-open fills, circuit guard, fixed exit by max_hold; explicit unfilled handling.
- WF: rolling IS=3y, OOS=1y; final 2y holdout unused until sign-off.
- Backtest: vectorbt wrapper; single long per stock; max hold; costs; concurrency cap; deterministic selection.
- Reporting: ledgers, summary JSON/MD, manifest with hashes/params to output_dir.

Directory Structure (proposed)
repo_root/
  cli.py
  config/
    schema.py
    loader.py
  data/
    fetch.py
    snapshot.py
    diff.py
    integrity.py
    calendar.py
    universe.py
  features/
    returns.py
    rolling.py
    robust.py
  detectors/
    base.py
    gap_z.py
    registry.py
  wf/
    splitter.py
    params.py
  exec/
    slippage.py
    rules.py
  bt/
    engine.py
    selection.py
  eval/
    metrics.py
    portfolio.py
  io/
    manifest.py
    reporting.py
    hashing.py
  tests/
    unit/
  runs/
  data/
    snapshots/

Config (conceptual)
- run: name, seed, t0, holdout_years, output_dir
- data: source='yfinance', interval in {'1d','1wk','1mo'}, start, end, snapshot_dir, calendar='NSE', refresh
- universe: top_turnover, lookback_years, size, freeze_at_t0, exclude
- wf: is_years=3, oos_years=1, roll='1y', calendar_aligned
- exec: next_open, circuit_guard_pct, fees_bps_side, slippage
- portfolio: max_concurrent, weighting='equal', selection_order, reentry_lockout
- detectors: [{name, params (≤3), per_stock_overrides}]
- reporting: {ledgers, json, md, plots}

Validation
- end ≥ start; t0 ≥ start; t0 + holdout_years ≤ end
- final holdout excluded from WF
- slippage bins/bps length match
- detector param bounds; ≤ 3 knobs

Data
- Snapshot path: data/snapshots/{source}_{interval}/{SYMBOL}.parquet
- Metadata: fetch_ts, tz, start/end, interval, calendar, source_version, code_git_sha, schema, symbol, exchange, row_count, sha256
- Refresh: refetch → diff (index/row/cell) → data_diff.csv; version old if drift > epsilon
- Integrity: trading days vs NSE, adj close continuity, TZ normalization, holiday alignment

Universe
- At t0: median daily turnover over lookback; rank desc; top N; apply excludes; freeze list; record ranks/stats.
- Note: survivorship limits of yfinance documented.

Features (minimal)
- Returns: daily, overnight, intraday
- Gaps: absolute and pct
- Rolling: mean/std, median/MAD, quantiles
- Helpers: zscore, robust_z with eps floors

Detectors
- Base: validate_params, param_space, optional fit(IS), signal(frame, params) -> Series[bool/float]
- gap_z:
  - params: W ∈ [20,60], k_low ∈ {-1.0,-1.5,-2.0}, max_hold ≤ 22
  - logic: z(gap_pct, W); entry if z < k_low; score = -z
- Param search: per-stock small grid on IS; objective = IS median trade return post-costs with hit-rate floor.
- Persist: runs/<run>/params/{TICKER}_gap_z.json; freeze for OOS/holdout.

Orders/Exits
- Entry: if signal on t and flat → pending for open t+1
- Exit: time-based ≤ max_hold; optional early exits off by default
- Lockout: until flat
- Unfilled: if circuit guard fails, log once, no retry

WF
- Rolls: IS=3y, OOS=1y, step=1y, calendar_aligned
- Each roll: fit on IS (frozen) → run OOS with frozen params → collect metrics

Execution/Costs
- Fill: next open
- Circuit guard: Open_{t+1} within [PrevClose * (1 ± guard)]
- Slippage model: percentile(|gap_pct|) → bps; turnover scaling; min floor; fees per side

Backtest Engine
- Single long per stock
- Max holding days ≤ 22
- Apply costs at fills
- Concurrency cap across universe
- Deterministic selection when oversubscribed:
  1) |z| desc
  2) turnover desc
  3) ticker asc
- Logs: trade ledger, unfilled with reasons; seeds wired to run.seed

Evaluation/Artifacts
- Metrics per stock/split: median trade ret (post-cost), 5th pct, hit rate, Sharpe, max DD, OOS/IS
- Portfolio: equal-weight of concurrent positions; MVP single detector
- Artifacts:
  - CSV: ledgers/trade_ledger.csv, ledgers/unfilled_log.csv, data_diff.csv (if refresh)
  - JSON: summary.json
  - MD: report.md (tables for OOS medians, 5th pct, hit rate, OOS/IS, MAR/DD vs benchmarks)
  - Manifest: run, seed, git sha; config hash+embed; snapshot hashes; universe; splits; per-stock params; cost/slippage config
  - Plots (optional): equity, drawdown, pnl_hist, calibration

CLI
- run:    python cli.py run --config path/to/config.yaml
- refresh python cli.py refresh-data --config path/to/config.yaml

Determinism
- No network during run
- All paths under output_dir
- Exchange calendar; explicit reindexing; no silent ffill
- Fixed grids; record git sha and SHA256s

Testing
- Unit: features, detectors, WF splitter, exec (guards/costs), bt (hold/selection)
- Property: reproducibility w/ seed; no lookahead; calendar alignment
- Regression: frozen dataset replays; weekly canary thresholds
- Lint/Type: ruff, black, mypy

Performance
- Vectorized pandas; precompute/cache rolling stats; Parquet(snappy); small N (top 10) for MVP

Errors/Logging
- Log: runs/<run>/logs/run.log (config, universe, splits, warnings, rejections, summary)
- Hard fail: missing snapshots when refresh=false; invalid config; empty universe; insufficient history
- Soft warn: minor diffs within epsilon; explained calendar gaps

Security
- Network only in refresh; no creds; run is offline

Acceptance Gates
- Record: OOS/IS per stock and roll; survival across ≥5 names and ≥3 rolls; portfolio MAR/DD
- Report pass/fail vs pre-registered thresholds.

Rationale
- Small surface, frozen data, explicit manifests.
- IS-only per-stock fitting; strict OOS and holdout.
- Deterministic capacity allocation.
- Focus on medians and tail after costs; avoid mean-chasing.
