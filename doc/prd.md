# Anomaly & Pattern Detection for Indian Stocks
CLI-only personal project • Python stats + yfinance • vectorbt backtesting • No indicators, no fundamentals • Daily or higher only
Mindset: minimalism, testable increments, walk-forward validation (Kailash Nadh style)

## 1) Purpose
Identify simple, statistically grounded price-patterns/anomalies in Indian equities that historically yield higher-than-market returns within ≤ 22 trading days. Emphasize parsimony, reproducibility, and out-of-sample (OOS) survival.

## 2) Constraints & Principles
- Data: yfinance only; intervals: 1d/1wk/1mo. Use Adjusted Close for returns. Snapshot to parquet with timestamp; never refetch for backtests unless explicit refresh with diff logs.
- Universe: NSE-listed stocks with deterministic, liquidity-based selection (see Universe). Optional NIFTY/NIFTYBEES for dispersion baseline only.
- Interface: CLI only. Outputs to console + CSV/JSON/MD; optional plots saved to disk.
- Methods: Pure statistics from OHLC(V). No technical indicators (RSI, MACD, etc.), no fundamentals, no intraday.
- Backtesting: vectorbt; per-stock unique parameterization (fit IS, test OOS). Rolling walk-forward. No lookahead. Final 2y holdout untouched for sign-off.
- Costs: model transaction fees (bps/side) and parameterized slippage by gap percentile/turnover. Execution realism (see Execution Assumptions).
- Philosophy: Smallest working slice → measure → iterate or kill. Keep parameter count minimal. Pre-register acceptance thresholds, enforce family-wise control via cross-name and multi-roll survival.

## 3) Success Metrics
- Primary per-stock, per-detector OOS (≤ 22D holds): median trade return (post-costs), 5th percentile trade return, hit-rate. Secondary: Sharpe, max drawdown.
- Portfolio aggregation: equal-weight across stocks (and optionally across detectors) with max concurrent cap; deterministic selection.
- Benchmarks: stock buy-and-hold and NIFTY baseline over same windows.
- Robustness: OOS not collapsing vs IS; signal persists across ≥ 5 liquid names and ≥ 3 WF rolls; OOS/IS ratio ≥ 0.5.

## 4) Data & Storage
- Source: yfinance `Ticker.history` with daily bars (Open, High, Low, Close, Adj Close, Volume).
- Snapshot: Write raw fetch to Parquet per symbol with timestamp, tz, start/end, interval metadata. No refetch for backtests unless explicit refresh.
- Diffing: On refresh, produce `data_diff.csv` (ticker, date, changed fields, counts). If diffs > epsilon, freeze old dataset for historical runs.
- Sanity checks: missing trading days, splits/adjustments, timezone alignment, NSE holiday calendar alignment.
- Universe selection: see Universe section; freeze list and store in manifest.

## 5) System Architecture (Config-first)
1. config: Single run configuration file (YAML/TOML) declares data sources, universe, detectors, parameters, WF splits, execution, costs, and outputs.
2. data: Fetch, snapshot, load; align calendars; diff detection on refresh; all paths/params from config.
3. features: Returns, gaps, rolling stats (mean/var, median/MAD, IQR), quantiles, run-lengths, density estimates.
4. detectors: Pluggable, stats-only modules producing per-day signals/scores; params from config; per-stock overrides supported.
5. signal: Turn detector outputs into entries/exits with per-stock thresholds and rules; defined in config.
6. wf: Deterministic rolling walk-forward split; fit IS only; apply to OOS; final 2y holdout untouched; boundaries defined in config.
7. execution: Next-open fill with circuit guard and slippage model; log unfilled signals; one position per stock; model params from config.
8. backtest: vectorbt wrapper; max holding days; costs; concurrency cap; deterministic selection; all driven by config.
9. eval: Per-stock and portfolio metrics; IS vs OOS; parameter persistence per stock; OOS/IS ratio.
10. io/report: Trade ledger, unfilled log, summary JSON/MD, plots; run manifest with hashes and config; artifact paths from config.

## 6) Run Configuration (single file)
- All runs are driven by a single config file (YAML/TOML). Example (YAML):
  run:
    name: gap_z_mvp_2025-08-01
    seed: 42
    t0: 2012-01-02
    holdout_final_years: 2
    output_dir: runs/gap_z_mvp_2025-08-01

  data:
    source: yfinance
    interval: 1d
    start: 2010-01-01
    end: 2025-07-31
    snapshot_dir: data/snapshots
    holiday_calendar: NSE
    refresh: false

  universe:
    selection: top_turnover
    lookback_years: 2
    size: 10
    freeze_at_t0: true
    exclude_lists: []  # e.g., ASM/GSM if available

  walk_forward:
    is_years: 3
    oos_years: 1
    roll: 1y
    calendar_aligned: true

  execution:
    next_open: true
    circuit_guard_pct: 0.10
    fees_bps_side: 10
    slippage_model:
      type: gap_percentile_turnover
      gap_percentile_bins: [0.5, 0.8, 0.95]
      slippage_bps: [5, 10, 15, 25]
      turnover_scaling: 0.5
      min_slippage_bps: 5
      percentile_lookback_years: 2

  portfolio:
    max_concurrent: 5
    weighting: equal
    selection_order: [signal_extremity_desc, turnover_desc, ticker_asc]
    reentry_lockout: true

  detectors:
    - name: gap_z
      params:
        window: [20, 60]
        k_low: [-1.0, -1.5, -2.0]
        max_hold: 22
      per_stock_overrides: {}

  reporting:
    emit_trade_ledger: true
    emit_unfilled_log: true
    emit_summary_json: true
    emit_md_report: true
    plots: [equity, drawdown, pnl_hist, calibration]

- Only two CLI entry points:
  1) `python cli.py run --config path/to/config.yaml`
  2) `python cli.py refresh-data --config path/to/config.yaml` (optional data refresh + diff)

## 7) Universe
- Deterministic, liquidity-based selection at backtest start t0:
  - Compute median daily turnover (Close * Volume, INR) over trailing 2y up to t0 for all NSE tickers available.
  - Select top N (default 10) by median turnover.
  - Freeze this list for the entire backtest and document in the run manifest. If opting to re-freeze per WF roll instead, log the exact list per roll; MVP uses fixed-at-t0.
- Survivorship guard: acknowledge inability to include delisted where data unavailable via yfinance; document limitation in manifest.
- Optional exclusions: tickers on ASM/GSM lists if a reliable source is available; otherwise proceed and log.

## 8) Execution Assumptions
- Entry/exit timing: at the next official open after signal day (no lookahead).
- Circuit guard: if open not within [prev_close * (1 ± 10%)] or yfinance indicates limit condition, mark trade unfilled. Log to unfilled ledger; no VWAP fallback in MVP.
- Slippage model (parameterized):
  - Base by gap percentile:
    - |gap| < P50: 5 bps
    - P50–P80: 10 bps
    - P80–P95: 15 bps
    - > P95: 25 bps
  - Turnover adjustment: for upper turnover quartiles, scale slippage by 0.5x with floor at 5 bps.
- Fees: default 10 bps/side. Total cost = fees + slippage.
- Re-entry: unfilled signals do not enter; follow re-entry lockout rules as if no position existed.

## 9) CLI Surface (minimal)
- Single-run: `python cli.py run --config path/to/config.yaml`
- Optional data refresh: `python cli.py refresh-data --config path/to/config.yaml`
- All outputs (paths, formats) are specified in the config; no other CLI flags are required or supported.

## 10) Backtest Specification
- Walk-forward: 3y IS fit, 1y OOS test; roll annually; no peeking across boundary. Final 2y holdout untouched for sign-off.
- Entries: at next open after signal day (avoid lookahead bias) with execution assumptions above.
- Exits: time-based (≤ 22 trading days) and/or upon median or prior-close reversion; stop-based optional (acknowledge gap risk).
- Costs: fees default 10 bps/side; slippage per slippage model; parameterizable via CLI.
- Risk/Portfolio: long-only initially; one concurrent position per stock; re-entry lockout until exit; max concurrent positions (default 5); deterministic selection by signal extremity then liquidity.

## 11) Tests & Integrity
- Unit tests: feature computations, detector outputs on synthetic data, walk-forward splitter, concurrency selection determinism.
- Integrity: fee + slippage application, entry/exit alignment, unfilled accounting, no future data leakage, circuit guard behavior.
- Property tests: reproducibility with fixed seed; per-stock parameter persistence and loading; calendar alignment with NSE holidays.
- Regression: frozen dataset replays to detect yfinance drift; weekly canary run diff with epsilon thresholds.

## 12) MVP (Milestone 1)
- Implement fetch/snapshot, returns, z/robust-z utils, deterministic walk-forward splitter.
- Execution realism: next-open fills, circuit guard, slippage model.
- Detector 1: Open-Close Negative Gap Z (mean-reversion).
- Backtest via vectorbt with ≤ 22-day holds, fees + slippage, max concurrent = 5, deterministic selection.
- Universe: top 10 by median 90D turnover at t0; frozen and logged.
- Produce OOS report across ≥5 liquid names; include trade ledger, unfilled log, run manifest; go/no-go decision.

### 12.1 MVP User Stories (high level)
- As a researcher, I can define a single config file with data, universe, detector, walk-forward, execution, portfolio, and reporting settings, so that a full run is reproducible without extra CLI flags.
- As a data pipeline, I can snapshot yfinance data to parquet with metadata and produce a diff report on refresh, so historical backtests are not silently altered by upstream changes.
- As the backtester, I can perform deterministic 3y IS / 1y OOS rolling splits with a final 2y holdout, so that leakage is prevented and validation is honest.
- As the executor, I can simulate next-open fills with a circuit guard and a parameterized slippage model, and I can log unfilled signals, so results reflect tradeability constraints.
- As the portfolio allocator, I can enforce max concurrent positions with deterministic selection (signal extremity → turnover → ticker), so capacity and tie-breaking are explicit.
- As the detector, I can compute Gap-Z signals with ≤3 parameters and per-stock fitting on IS only, so complexity is capped and OOS is evaluated cleanly.
- As the reporter, I can emit a trade ledger, unfilled log, summary JSON/MD, and a run manifest with hashes and config, so every result is auditable and reproducible.
- As the reviewer, I can open the generated MD report and quickly see per-stock OOS medians, lower-tail, hit-rate, OOS/IS degradation, and portfolio MAR/DD against benchmarks, so I can make a go/no-go decision fast.

## 13) Detector/Pattern Ideas (Stats-only, daily+; ≤ 3 params each; per-stock fit)
Gaps / Tails / Mean-Reversion
1. Open-Close Gap Z
   gap_t = (Open_t - Close_{t-1}) / Close_{t-1}; rolling z over W. Enter long if z < k_low.

2. Overnight vs Intraday Split
   r_overnight = Open_t/Close_{t-1}-1; r_intraday = Close_t/Open_t-1. Rare large negative overnight → positive drift next 1–5 days.

3. Lower-tail Quantile Reversion
   If r_t below q_low (1–5%), enter long; exit on median reversion or ≤ 22 days.

Robust Outliers / Dispersion
4. Robust Z (Median/MAD)
   robust_z = (r_t - median)/MAD. Enter if robust_z < -k.

5. IQR Outlier (Tukey fences)
   If r_t < Q1 - c*IQR; enter; exit median/T days.

6. Surprisal (density-based)
   Estimate p(r_t) via KDE/Histogram; surprisal s_t = -log p(r_t). Enter on extreme negative surprisal if historically followed by drift.

Variance / Volatility Structure
7. Variance Spike Reversion
   Spike vs rolling median/MAD; enter once spike abates X% and day is negative tail.

8. Range Compression then Shock
   Low-range streak followed by large negative day; enter for bounce.

Changepoints / Regimes
9. Mean/Variance Changepoint (ruptures)
   Break at boundary with negative jump; partial reversion.

10. HMM (2-state) on returns
   Panic/volatile vs calm. Negative tail near transition → reversion.

11. Autocorr Break
   Short-lag autocorr sign flip with negative tail day → regression to mean.

Run-length / Sequences
12. Down Run-Length Extreme
   Consecutive down days > 95th pct; last day largest drop; enter.

13. Diminishing Ladder
   Diminishing negative days followed by a larger spike; enter.

14. KNN Sequence Match
   k-day return vector; nearest historical sequences; if neighbors’ next-22d mean > 0, enter.

Drawdown Context
15. Drawdown-Phase Tail
   In top-decile drawdown; if bottom 2% daily return within this regime, enter.

16. Winsorized Drawdown Extreme
   Rare drawdown depth percentile breach → bounce.

Temporal Conditioning (Price-only)
17. Day-of-Week Shock
   Stock-specific weekday extremes; use only if IS and OOS both support.

18. Month-End/Start
   Large negative at month-end normalizes in early next month; validate OOS.

Event Windows by Price-only
19. Price Event Clusters
   Clusters of |r| > threshold define events; post-event drift/reversion N days.

Noise-Model Residuals
20. AR(1) Residual Shock
   Fit AR(1) IS; residual < -k*sigma_res → enter; exit T days or mean.

21. Gaussian Diffusion Tail
   Fit sigma; extreme negative tail → enter.

Shapes
22. Quadratic U-Shape (k=5–10)
   Fit y ~ a t^2 + b t + c on recent returns; concave-up with large negative final residual → bounce.

Microstructure-like (no intraday)
23. Down-Gap Absorption
   Large negative gap with close≈open (small net day); next-day long.

Cross-Sectional (Price-only)
24. Dispersion vs NIFTY
   diff_t = r_stock - r_nifty; extreme underperformance day → revert toward peer.

Regime-aware Thresholding
25. Volatility Regime Split
   Separate thresholds for low/high variance regimes to stabilize signals.

## 14) Parameterization & Per-Stock Uniqueness
- Each detector ≤ 3 parameters: window W, threshold k, holding H.
- Per stock: fit on IS via small grid/Bayes; persist best params (YAML/JSON).
- Apply frozen params in OOS; log metrics; prune detectors failing OOS across ≥ 3 years and ≥ 5 stocks; require OOS/IS ≥ 0.5.

## 15) Risk & Execution
- Long-only initially; one position per stock; re-entry lockout until exit.
- Max hold 22 trading days. Optional early exit on median or prior-close reversion.
- Costs 10 bps/side by default; slippage per model; parameterizable.
- Portfolio: max N concurrent (default 5); equal-weight 1/N; deterministic selection by signal extremity then liquidity; no pyramiding.

## 16) Reporting
- Trade ledger CSV: timestamps, prices, fees, slippage, notional, PnL, holding duration; unfilled signals log.
- JSON/MD summaries: per-detector, per-stock IS vs OOS metrics, params; portfolio metrics; OOS/IS degradation ratios.
- Manifest (run.yaml): data file hashes, universe list, WF split dates, param grid, costs, slippage model, seed, git sha, CLI args.
- Optional plots to `./plots`: equity curves, drawdowns, holding PnL hist, regime-wise returns, parameter sensitivity, calibration by signal decile.

## 17) Iteration Plan
- Milestone 1 (1–2 days): fetch/snapshot, returns & z/robust-z utils, WF splitter, Gap-Z detector, execution realism, vbt backtest with ≤ 22-day holds + fees + slippage + concurrency cap, OOS report with manifest and ledgers.
- Milestone 2: add Robust Z (MAD), Tail Quantile Reversion, Dispersion vs NIFTY, Volatility-regime thresholds; per-stock grid; aggregate equal-weight; prune non-generalizers via gates.
- Milestone 3: add Variance Spike Reversion; consider AR(1) residual only if Milestone 2 passes holdout; weekday conditionality only if IS+OOS agree across ≥3 rolls.
- Stretch: KNN sequence, HMM, KDE surprisal, U-shape; only if they clear strict OOS gates.

## 18) Guardrails Against Overfitting
- Minimal params; fixed walk-forward; no leakage; final 2y holdout untouched.
- Prefer robust stats (median/MAD). Prioritize medians and lower-tail metrics over means.
- Multiple testing control: require survival across ≥ 5 names, ≥ 3 WF rolls, and final holdout.
- Acceptance thresholds pre-registered (see Success Metrics, Failure Conditions).
- Track edge decay; deprecate decaying detectors.

## 19) Non-Goals
- No intraday/options/leverage initially.
- No TA indicators; no fundamentals.
- No deep learning; no heavy feature engineering.
- No external event calendars.

## 20) Failure Conditions
- Deprecate any detector whose OOS/IS performance ratio < 0.5 in two consecutive WF rolls.
- Deprecate any detector whose median trade return turns negative after costs in final 2y holdout.
- Portfolio-level rejection if holdout MAR < 0.2 or max drawdown breaches pre-set limit vs stock basket.

## 21) Stretch
- Regime meta-switch across detectors.
- Simple ensemble voting across detectors while preserving per-stock uniqueness.
- Conservative Kelly sizing if OOS stability proven and drawdown controlled.

— End of PRD —
