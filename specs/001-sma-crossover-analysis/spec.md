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
2. **Given** multiple stocks and periods, **When** aggregation runs, **Then** a CSV `sma_crossover_analysis_results.csv` is produced with one row per (Stock, Period_Label, Short_SMA, Long_SMA, Holding_Period) containing Num_Signals, Mean_Return, Std_Dev_Return, Mean_Max_Drawdown, Win_Rate, P_Value.

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

## Requirements *(mandatory)*

Derived and formalized from `doc/requirements.md`.

### Functional Requirements

- **FR-001 (Data Acquisition & Preprocessing)**: Given a list of tickers and date range, system MUST download daily `Adj Close` prices, forward-fill sporadic NaNs, and return a pandas DataFrame indexed by date with tickers as columns. The system MUST log a warning per ticker with the count of NaNs filled.

- **FR-002 (Time Period Segmentation)**: System MUST slice the 2004-01-01 to 2023-12-31 range into four labeled periods: `2004-2009 (Pre-Crisis Bull Market & Crash)`, `2009-2014 (Post-Crisis Recovery)`, `2014-2019 (Modi-Era Bull Run)`, `2019-2024 (COVID Volatility & New Highs)`. All downstream analysis MUST iterate over these slices.

- **FR-003 (Signal Generation)**: For each stock and period, system MUST compute SMAs for the 10 specified pairs and record a buy signal on day T when SMA_X[T] > SMA_Y[T] AND SMA_X[T-1] <= SMA_Y[T-1]. The system MUST return signal dates per (Stock, Period, Short_SMA, Long_SMA).

- **FR-004 (Post-Signal Performance)**: For each signal, system MUST compute forward returns and max drawdown for holding periods of 10, 15, and 20 trading days using the entry price on T. Signals without sufficient forward data for a holding period MUST be discarded for that holding period.

- **FR-005 (Aggregation & Stats)**: For each unique (Stock, Period_Label, Short_SMA, Long_SMA, Holding_Period) group, system MUST compute: Num_Signals, Mean_Return, Std_Dev_Return, Mean_Max_Drawdown, Win_Rate (returns > 0), and P_Value (one-sample t-test vs 0 using scipy.stats.ttest_1samp).

- **FR-006 (Buy-and-Hold Benchmark)**: For each stock and period, system MUST compute buy-and-hold return = (Last - First)/First and include this in the final report aggregation.

- **FR-007 (CSV Output)**: System MUST write `sma_crossover_analysis_results.csv` containing columns: Stock, Period_Label, Short_SMA, Long_SMA, Holding_Period, Num_Signals, Mean_Return, Std_Dev_Return, Mean_Max_Drawdown, Win_Rate, P_Value.

- **FR-008 (Heatmap Visualization)**: System MUST create `heatmap_avg_mean_return.png` that visualizes the average Mean_Return across all stocks and periods for each tested SMA pair using a divergent colormap and clearly labeled axes.

- **FR-009 (Markdown Summary Report)**: System MUST write `analysis_summary.md` containing Executive Summary, Heatmap reference, Consistency Analysis (highlight parameter regions with positive mean returns and p-value < 0.05), Top Performers (top 5 significant rows), Fallen Angel analysis for `YESBANK.NS`, Benchmark Comparison, and Signal Frequency Analysis.

### Key Entities

- **TickerTimeSeries**: Daily adjusted close prices keyed by date for a ticker; cleaned via forward-fill and annotated with NaN fill counts.
- **SignalRecord**: {Ticker, Period_Label, Short_SMA, Long_SMA, SignalDate, EntryPrice, Returns_{10,15,20}, MaxDrawdowns_{10,15,20}}
- **AggregatedMetric**: {Stock, Period_Label, Short_SMA, Long_SMA, Holding_Period, Num_Signals, Mean_Return, Std_Dev_Return, Mean_Max_Drawdown, Win_Rate, P_Value}

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Data function returns a cleaned DataFrame for all 15 tickers with no NaNs remaining after forward-fill and per-ticker NaN fill warnings printed.
- **SC-002**: For each of the 10 SMA pairs and 3 holding periods, aggregated CSV contains rows for each stock-period combination; no required row is missing unless there were zero signals.
- **SC-003**: Heatmap PNG is produced at 300 DPI resolution and shows cells for all 10 SMA pairs; the color scale is centered at 0% mean return.
- **SC-004**: The t-test P_Value is computed for every aggregated row; rows with Num_Signals < 5 should be flagged in the CSV as low-power (inspectable).
- **SC-005**: `analysis_summary.md` contains the listed sections and includes a clear call-out for `YESBANK.NS` describing data limitations and observed strategy performance.

## Assumptions

- The Yahoo Finance `Adj Close` series accurately reflects corporate actions for these tickers over the period 2004-01-01 to 2023-12-31.
- Trading days are calendar business days present in the downloaded series; "N trading days" is interpreted as N rows forward in the cleaned series.
- Signals are evaluated and aggregated independently per period slice (no carry-over of signals between periods).
- Significance threshold for p-values is 0.05 unless otherwise specified in follow-ups.

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

