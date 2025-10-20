# Feature Specification: SMA Crossover Signal Efficacy Analysis

**Feature Branch**: `001-sma-crossover-analysis`  
**Created**: 2025-10-18  
**Status**: Draft  
**Input**: User description from `doc/requirements.md` (SMA crossover analysis; emulate Geoffrey Hinton & Kailash Nadh mindset)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Researcher asks for cohort signal analysis (Priority: P1)

As a quantitative researcher, I want to run the SMA crossover analysis over a 20-year dataset broken into four 5-year cohorts so that I can identify whether short-term SMA crossover signals show consistent, statistically significant post-signal returns across multiple market regimes.

**Why this priority**: Core research question — validates the hypothesis and drives all other outputs.

**Independent Test**: Run the analysis for a single stock and single SMA pair; verify CSV output and summary metrics contain expected columns and non-empty values.

**Acceptance Scenarios**:

1. **Given** cleaned `Adj Close` prices for a stock and period, **When** the analysis runs for SMA pair (X,Y) and holding period N, **Then** the system outputs signal dates and calculated returns/drawdowns for each valid signal.
2. **Given** multiple stocks and periods, **When** aggregation runs, **Then** a timestamped canonical CSV `outputs/sma_crossover_analysis_results_{RUN_TS}.csv` is produced (one row per Stock, Period_Label, Short_SMA, Long_SMA, Holding_Period). For convenience the run MAY also write a stable copy `outputs/sma_crossover_analysis_latest.csv` pointing to the most recent canonical file. The CSV must contain Num_Signals, Num_Signals_Used, Mean_Return, Std_Dev_Return, Mean_Max_Drawdown, Win_Rate, P_Value and the expanded diagnostics columns described below.

---

### User Story 2 - Analyst wants heatmap overview (Priority: P2)

As a market analyst, I want a heatmap showing average mean returns across all stocks and periods for the tested SMA pairs, so I can visually identify promising SMA parameter regions.

**Why this priority**: Provides a compact, visual summary for quick decision-making.

**Independent Test**: Generate `heatmap_avg_mean_return.png`; visually confirm cells corresponding to the 10 SMA pairs are present and caption indicates aggregation method.

**Acceptance Scenarios**:

1. **Given** aggregated metrics, **When** the heatmap is created, **Then** the PNG file exists and axes are labeled with Short and Long SMA periods.

---

### User Story 3 - Product/PM needs written summary (Priority: P3)

As a product stakeholder, I want `analysis_summary.md` that synthesizes executive summary, heatmap reference, consistency analysis, top performers, Fallen Angel case (YESBANK.NS), and benchmark comparisons so stakeholders can understand results without running code.

**Why this priority**: Required for communicating findings to non-technical stakeholders.

**Independent Test**: `analysis_summary.md` exists with the required sections and refers to the generated PNG and CSV.

**Acceptance Scenarios**:

1. **Given** output artifacts, **When** the report writer runs, **Then** the markdown file contains Executive Summary, Heatmap reference, Consistency Analysis, Top Performers table, Fallen Angel analysis, Benchmark Comparison, and Signal Frequency Analysis.

---

### Edge Cases

- Signals that occur fewer than the required forward days from the period end must be discarded for that holding period calculation.
- Stocks with large contiguous NaNs: forward-fill is applied and a warning logged; extremely sparse series (e.g., >10% NaNs) should be flagged in the report.
- Tickers delisted or with truncated history (e.g., YESBANK.NS during collapse): include analysis but call out limited data and interpret significance carefully.

### Parameter list (explicit)

The analysis will test the following 10 SMA pairs (Short_SMA, Long_SMA). These are canonical; Short < Long for all pairs:

- (5, 20)
- (5, 50)
- (10, 50)
- (10, 100)
- (20, 50)
- (20, 100)
- (50, 100)
- (8, 21)
- (13, 34)
- (3, 10)

If you want a different set, list them explicitly; code will validate Short < Long and uniqueness.

## Requirements *(mandatory)*

Derived and formalized from `doc/requirements.md`.

### Functional Requirements

- **FR-001 (Data Acquisition & Preprocessing)**: Given a list of tickers and date range, system MUST download daily `Adj Close` prices, forward-fill sporadic NaNs, and return a pandas DataFrame indexed by date with tickers as columns. The system MUST log a warning per ticker with the count of NaNs filled.

	- Implementation details and guardrails:
		- Forward-fill is applied per-ticker; after forward-fill, if more than 10% of dates for a ticker in a period remain missing (or were NaN before forward-fill), that ticker-period pair MUST be flagged in outputs and may be excluded from some summaries (see `Low_Power_Flag`).
		- SMA computation requires a full rolling window: use rolling(window=period, min_periods=period) so rows before the long SMA window exists are dropped for signal-generation purposes and logged. Log counts of dropped/WARM-UP rows per ticker.
		- All warnings and per-ticker diagnostic summaries MUST be written to `outputs/logs/na_fill_warnings.csv` and `outputs/logs/warmup_drop_counts.csv`.

- **FR-002 (Time Period Segmentation)**: System MUST slice the 2004-01-01 to 2023-12-31 range into four labeled periods: `2004-2009 (Pre-Crisis Bull Market & Crash)`, `2009-2014 (Post-Crisis Recovery)`, `2014-2019 (Modi-Era Bull Run)`, `2019-2024 (COVID Volatility & New Highs)`. All downstream analysis MUST iterate over these slices.
 - **FR-002 (Time Period Segmentation)**: System MUST slice the 2004-01-01 to 2023-12-31 range into four non-overlapping 5-year cohorts using inclusive start/end dates. The cohorts are:

	- `2004-01-01` to `2008-12-31` (Pre-Crisis Bull Market & Crash)
	- `2009-01-01` to `2013-12-31` (Post-Crisis Recovery)
	- `2014-01-01` to `2018-12-31` (Modi-Era Bull Run)
	- `2019-01-01` to `2023-12-31` (COVID Volatility & New Highs)

	Signals are assigned to the period in which the signal date T occurs (inclusive). Forward returns used to evaluate a signal MAY use data outside the period (i.e., crossing the period boundary) as long as the required N trading days exist in the ticker's cleaned timeline. This preserves realism: a signal at period end is assigned to that period but its forward outcome is the true future outcome.

- **FR-003 (Signal Generation)**: For each stock and period, system MUST compute SMAs for the 10 specified pairs and record a buy signal on day T when SMA_X[T] > SMA_Y[T] AND SMA_X[T-1] <= SMA_Y[T-1]. The system MUST return signal dates per (Stock, Period, Short_SMA, Long_SMA).

	- Additional rules and options:
		- SMA computation: require full window for both short and long SMAs; do not use partial-window values to avoid biased signals.
		- Overlapping signals policy: by default, the system SHALL allow overlapping signals (count each signal independently). Optionally, a mode `non_overlapping` may be specified to suppress signals that occur within the holding period of a prior accepted signal for the same (Stock, SMA pair, Period). Both modes MUST be supported and the mode used MUST be recorded in outputs (`Overlap_Mode`).
		- Validate SMA pair list at runtime and error if duplicates or Short >= Long are provided.

- **FR-004 (Post-Signal Performance)**: For each signal, system MUST compute forward returns and max drawdown for holding periods of 10, 15, and 20 trading days using the entry price on T. Signals without sufficient forward data for a holding period MUST be discarded for that holding period.

	- Precise definitions and computation rules:
		- "N trading days" is defined as N rows forward on the ticker's cleaned Adj Close series (i.e., per-ticker trading calendar). Do not translate to calendar days.
		- EntryPrice: default EntryPrice = cleaned `Adj Close` on date T (backtest-style). An alternative mode `entry_next_open` may be supported later; if used, it must be documented and the mode recorded in `analysis_summary.md`.
		- Return_N = Price_{T+N} / Price_T - 1. If Price_{T+N} does not exist for that ticker, the signal is discarded for that N.
		- Max_Drawdown_N (for the holding window [T, T+N]): compute as max_{t in [T, T+N]} (1 - Price_t / Price_T). Report as a non-negative decimal (e.g., 0.12 for 12%). Example: if EntryPrice=100 and prices drop to 88 at worst, Max_Drawdown=0.12.
		- Signal assignment: a signal belongs to the period where T occurs; its forward computation can use data beyond the period boundary as described in FR-002.

- **FR-005 (Aggregation & Stats)**: For each unique (Stock, Period_Label, Short_SMA, Long_SMA, Holding_Period) group, system MUST compute: Num_Signals, Mean_Return, Std_Dev_Return, Mean_Max_Drawdown, Win_Rate (returns > 0), and P_Value (one-sample t-test vs 0 using scipy.stats.ttest_1samp).

	- Additional statistical rules:

		- Raw p-values are computed via a one-sample t-test (by default two-sided) against a null mean of 0. For robustness, also compute summary statistics: `Median_Return`, `Skew`, `Kurtosis`, and a bootstrap 95% CI for the mean (percentile bootstrap, 1000 resamples, RNG seed=42). If `Num_Signals_Used >= 30` the t-test is treated as the primary inferential statistic; if `5 <= Num_Signals_Used < 30` present both the t-test and the bootstrap CI and add a caution about small-sample inference; if `Num_Signals_Used < 5` set `Low_Power_Flag = TRUE` and exclude the row from BH-FDR (but retain it in the CSV for inspection).

		- Definitions and usage:

			- `Num_Signals` = raw count of detected signals before discarding for insufficient forward data.
			- `Num_Signals_Used` = count of signals remaining after discarding signals that lack sufficient forward data for the given holding period.

			- All per-holding-period aggregations, p-values, and the BH-FDR procedure MUST be computed using `Num_Signals_Used`.

		- BH-FDR (Benjamini-Hochberg) adjustment:

			- For each holding period separately (10, 15, 20 days), collect the flattened set of aggregated rows (Stock, Period_Label, Short_SMA, Long_SMA) with `Num_Signals_Used >= 5` and compute BH-FDR adjusted p-values. Record the result in the `Adj_P_Value_BH` column. Rows with `Num_Signals_Used < 5` must have `Adj_P_Value_BH = NA`.

- **FR-006 (Buy-and-Hold Benchmark)**: For each stock and period, system MUST compute buy-and-hold return = (Last - First)/First and include this in the final report aggregation.


- **FR-007 (CSV Output)**: System MUST write a timestamped canonical CSV `outputs/sma_crossover_analysis_results_{RUN_TS}.csv` containing at minimum the following columns:

	- Stock, Period_Label, Period_Start, Period_End, Short_SMA, Long_SMA, Holding_Period,
	- Num_Signals, Num_Signals_Used, Low_Power_Flag (boolean), Mean_Return, Std_Dev_Return, Median_Return,
	- Mean_CI_lower, Mean_CI_upper (bootstrap 95% CI), Mean_Max_Drawdown, Win_Rate, P_Value, Adj_P_Value_BH,
	- Skew, Kurtosis, Overlap_Mode, Entry_Mode, Notes (free-text for warnings e.g., "insufficient history"), and Timestamp/Version.

	- The run MUST also write a companion diagnostics file `outputs/sma_crossover_analysis_diagnostics_{RUN_TS}.csv` containing per-ticker warnings and warmup counts. For convenience the run MAY also write stable copies named `outputs/sma_crossover_analysis_latest.csv` and `outputs/sma_crossover_analysis_diagnostics_latest.csv` that point to the most recent canonical files.

- **FR-008 (Heatmap Visualization)**: System MUST create `heatmap_avg_mean_return.png` that visualizes the average Mean_Return across all stocks and periods for each tested SMA pair using a divergent colormap and clearly labeled axes.

	- Aggregation and output rules:
	- Aggregation and output rules:
		- Heatmap primary cell value = unweighted mean of `Mean_Return` across all (Stock, Period_Label) aggregated rows for that SMA pair and holding period. Also produce a weighted heatmap where each cell is weighted by `Num_Signals_Used` to reflect sample size.
		- Heatmaps MUST be produced per holding period (10, 15, 20) by default. A separate aggregated heatmap across holding periods may be generated on demand; such aggregation must be explicitly captioned.
		- Heatmap axes: x-axis = Short_SMA, y-axis = Long_SMA. Because the tested SMA pairs form an explicit list (not necessarily a full Cartesian grid), cells for Short/Long combinations that were not tested must be masked (greyed out) and annotated in the caption as "not tested". Each heatmap must use a divergent colormap centered at 0% mean return and include a colorbar and cell annotations when feasible.
		- Save heatmaps at 300 DPI to `outputs/heatmap_avg_mean_return_{holding_period}d.png` and `outputs/heatmap_weighted_by_signals_{holding_period}d.png` with color scale centered at 0% mean return.

- **FR-009 (Markdown Summary Report)**: System MUST write `analysis_summary.md` containing Executive Summary, Heatmap reference, Consistency Analysis (highlight parameter regions with positive mean returns and p-value < 0.05), Top Performers (top 5 significant rows), Fallen Angel analysis for `YESBANK.NS`, Benchmark Comparison, and Signal Frequency Analysis.

	- The report MUST include:
		- A reproducibility header listing versions of key libraries (pandas, numpy, scipy, matplotlib, seaborn, python) and the RNG seed used.
		- The Overlap_Mode and Entry_Mode used for the run.
		- A section on multiple-testing correction describing the BH-FDR procedure and which rows were included/excluded.
		- Fallen Angel (`YESBANK.NS`) subsection: include a time-series PNG (`outputs/yesbank_signals.png`) with signals annotated, a small table of each signal (SignalDate, EntryPrice, Return_10/15/20, Max_Drawdown_10/15/20), a note on data truncation and number of usable signals, and whether any p-values remain significant after FDR correction.
		- A reproducibility header listing versions of key libraries (pandas, numpy, scipy, matplotlib, seaborn, python), the RNG seed used, and the git commit hash and branch used for the run.
		- The Overlap_Mode and Entry_Mode used for the run. Defaults: `Overlap_Mode` = `overlapping` (count signals independently) and `Entry_Mode` = `entry_on_T` (use cleaned Adj Close on T). These defaults must be written to the run manifest.
		- A section on multiple-testing correction describing the BH-FDR procedure and which rows were included/excluded.
		- Fallen Angel (`YESBANK.NS`) subsection: include a time-series PNG (`outputs/yesbank_signals.png`) with signals annotated, a small table of each signal (SignalDate, EntryPrice, Return_10/15/20, Max_Drawdown_10/15/20), a note on data truncation and number of usable signals, and whether any p-values remain significant after FDR correction.

### Key Entities

- **TickerTimeSeries**: Daily adjusted close prices keyed by date for a ticker; cleaned via forward-fill and annotated with NaN fill counts.
- **SignalRecord**: {Ticker, Period_Label, Short_SMA, Long_SMA, SignalDate, EntryPrice, Returns_{10,15,20}, MaxDrawdowns_{10,15,20}}
- **AggregatedMetric**: {Stock, Period_Label, Short_SMA, Long_SMA, Holding_Period, Num_Signals, Mean_Return, Std_Dev_Return, Mean_Max_Drawdown, Win_Rate, P_Value}

 - Diagnostics and logging: the run MUST write `outputs/logs/na_fill_warnings.csv` (per-ticker counts of pre-forward-fill NaNs and percent missing), and `outputs/logs/warmup_drop_counts.csv` (per-ticker counts of rows dropped due to SMA warmup). When more than 10% of dates in a ticker-period are missing before forward-fill, that ticker-period MUST be flagged in diagnostics as `Insufficient_History` and included in `outputs/sma_crossover_analysis_diagnostics.csv` with a `Low_Power_Flag` note. Forward-fill is still applied but the flag warns downstream users.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Data function returns a cleaned DataFrame for all 15 tickers with no NaNs remaining after forward-fill and per-ticker NaN fill warnings printed.
- **SC-002**: For each of the 10 SMA pairs and 3 holding periods, aggregated CSV contains rows for each stock-period combination; no required row is missing unless there were zero signals.
- **SC-003**: Heatmap PNG is produced at 300 DPI resolution and shows cells for all 10 SMA pairs; the color scale is centered at 0% mean return.
- **SC-004**: The t-test P_Value is computed for every aggregated row; rows with Num_Signals < 5 should be flagged in the CSV as low-power (inspectable).
- **SC-005**: `analysis_summary.md` contains the listed sections and includes a clear call-out for `YESBANK.NS` describing data limitations and observed strategy performance.

Additionally:

- The CSV MUST include `Adj_P_Value_BH` and `Low_Power_Flag` columns as described.

- The run MUST write the `outputs/logs` folder with timestamped log files named `na_fill_warnings_{RUN_TS}.csv` and `warmup_drop_counts_{RUN_TS}.csv` and include a timestamped `run_manifest_{RUN_TS}.json` listing parameters (SMA pairs, date range, Overlap_Mode, Entry_Mode, RNG seed, library versions, git commit, git branch, OS). Use RUN_TS = UTC compact ISO format `YYYYMMDDTHHMMSSZ` for canonical filenames. The run MAY also write stable convenience copies using `_latest` suffix.











- The run MUST write the `outputs/logs` folder with `na_fill_warnings.csv` and `warmup_drop_counts.csv` and include a timestamped `run_manifest_{YYYYMMDD_HHMMSS}.json` listing parameters (SMA pairs, date range, Overlap_Mode, Entry_Mode, RNG seed, library versions, git commit, git branch, OS). The default filenames for main outputs must include a timestamp (e.g., `outputs/sma_crossover_analysis_results_{YYYYMMDD_HHMMSS}.csv`).

## Assumptions

## Run contract (API / CLI schema)

This project exposes a simple run contract for implementers. The run may be invoked via a Python API or a CLI script that accepts the inputs below and writes the outputs described.

- Inputs:

	- `tickers`: List[str] or path to CSV (column `Ticker`).
	- `start_date`: string, YYYY-MM-DD (inclusive).
	- `end_date`: string, YYYY-MM-DD (inclusive).
	- `sma_pairs`: List[tuple[int,int]]; default = the 10 canonical pairs in this spec.
	- `holding_periods`: List[int]; default = [10, 15, 20].
	- `overlap_mode`: one of `overlapping` (default) or `non_overlapping`.
	- `entry_mode`: one of `entry_on_T` (default) or `entry_next_open`.
	- `seed`: int RNG seed for bootstrap (default 42).
	- `output_dir`: path where `outputs/` will be written (default: repository `outputs/` folder).

- Outputs (canonical):

	- `outputs/sma_crossover_analysis_results_{RUN_TS}.csv` (canonical aggregated CSV).
	- `outputs/sma_crossover_analysis_diagnostics_{RUN_TS}.csv` (per-ticker diagnostics).
	- `outputs/logs/na_fill_warnings_{RUN_TS}.csv`, `outputs/logs/warmup_drop_counts_{RUN_TS}.csv`.
	- `outputs/heatmap_avg_mean_return_{holding_period}d_{RUN_TS}.png` and weighted variant.
	- `outputs/analysis_summary_{RUN_TS}.md` and any focal ticker PNGs (e.g., `outputs/yesbank_signals_{RUN_TS}.png`).
	- `outputs/run_manifest_{RUN_TS}.json` containing inputs, environment (library versions), git commit/branch, OS, and timestamps.


- The Yahoo Finance `Adj Close` series accurately reflects corporate actions for these tickers over the period 2004-01-01 to 2023-12-31.
- Trading days are calendar business days present in the downloaded series; "N trading days" is interpreted as N rows forward in the cleaned series.
- Signals are evaluated and aggregated independently per period slice (no carry-over of signals between periods).
- Significance threshold for p-values is 0.05 unless otherwise specified in follow-ups.

Notes:

- Analysis treats each (Stock, Period) unit independently when aggregating; heatmap aggregation method (unweighted vs weighted) is explicitly controlled as above.

- There is no plan for intraday processing; all timestamps are treated as date-only and the timezone for interpretation and any required labeling is Indian Standard Time (IST).

Notes:

- Analysis treats each (Stock, Period) unit independently when aggregating; heatmap aggregation method (unweighted vs weighted) is explicitly controlled as above.

## Non-Goals / Out of Scope

- Transaction costs, slippage, taxes, or execution latency are explicitly excluded.
- Any position sizing, stop-loss mechanics, sell signals, or portfolio-level risk management are out of scope.
- No exhaustive parameter grid search beyond the specified 10 SMA pairs.

## Deliverables

- `specs/001-sma-crossover-analysis/spec.md` (this file)
- `sma_crossover_analysis_results.csv` (aggregated results)
- `heatmap_avg_mean_return.png` (visual summary)
- `analysis_summary.md` (human-readable report)

## Testing & Validation Notes

- Unit tests should cover: SMA signal generator for edge cases, forward-return and drawdown computation including boundary discard behavior, aggregation correctness, and CSV/PNG writer presence.
- Validate that Num_Signals >= 1 for rows included; flag low-sample rows (Num_Signals < 5).

- Bootstrap 95% CI: use percentile bootstrap with 1000 resamples and RNG seed=42. Report `Mean_CI_lower` and `Mean_CI_upper` computed from the bootstrap distribution (percentile method). If BCa intervals are required, document and implement them in a follow-up.

