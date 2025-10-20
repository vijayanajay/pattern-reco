# Data Model: SMA Crossover Analysis

This document defines the canonical data entities, validation rules, and sentinel values used by the SMA crossover analysis. The model deliberately mirrors the CSV outputs so downstream pipelines can rely on deterministic column names and sentinel semantics.

Entities

1. TickerTimeSeries
   - Description: Daily adjusted close prices keyed by date for a single ticker; cleaned via forward-fill and annotated with NaN fill counts.
   - Fields:
     - ticker: str (e.g., "YESBANK.NS")
     - date: date (index)
     - adj_close: float (cleaned Adj Close after forward-fill)
     - na_filled_count: int (count of NaNs filled during preprocessing)
     - percent_missing_before_fill: float
   - Notes: Stored in-memory as a `polars.Series`/`polars.DataFrame` per ticker during processing. Persisted only as diagnostics (CSV) and per-run artifacts if requested.

2. SignalRecord
   - Description: A detected buy signal for a given SMA pair and date.
   - Fields:
     - ticker: str
     - period_label: str (e.g., "2004-01-01_to_2008-12-31")
     - short_sma: int
     - long_sma: int
     - signal_date: date
     - entry_price: float
     - return_10: float|NA
     - return_15: float|NA
     - return_20: float|NA
     - max_drawdown_10: float|NA
     - max_drawdown_15: float|NA
     - max_drawdown_20: float|NA
     - notes: str (warnings like "insufficient_history", "discarded_insufficient_forward_data")

3. AggregatedMetric
   - Description: Aggregated statistics per (Ticker, Period_Label, Short_SMA, Long_SMA, Holding_Period)
   - Fields:
     - stock: str
     - period_label: str
     - period_start: date
     - period_end: date
     - short_sma: int
     - long_sma: int
     - holding_period: int
     - num_signals: int
     - num_signals_used: int
     - low_power_flag: bool
     - mean_return: float|NA
     - std_dev_return: float|NA
     - median_return: float|NA
     - mean_ci_lower: float|NA
     - mean_ci_upper: float|NA
     - mean_max_drawdown: float|NA
     - win_rate: float|NA
     - p_value: float|NA
     - adj_p_value_bh: float|NA
     - skew: float|NA
     - kurtosis: float|NA
     - buy_and_hold_return: float|NA  # (Last - First) / First for the period
     - overlap_mode: str
     - entry_mode: str
     - notes: str
     - timestamp: str (RUN_TS)

Relationships

- Each `TickerTimeSeries` produces zero-or-more `SignalRecord` entries per SMA pair and period.
- Each `SignalRecord` maps to zero-or-one aggregated rows per holding period; aggregation groups by (Ticker, Period_Label, Short_SMA, Long_SMA, Holding_Period).

Validation rules

- SMA pair validation: Short < Long and pairs are unique.
- NaN threshold: default percent_missing_before_fill threshold = 10% (configurable). If percent_missing_before_fill > threshold for a ticker-period, flag `Insufficient_History` in diagnostics and set `Low_Power_Flag` where appropriate.
  
Note on na_threshold units

Note: the `na_threshold` referenced here is expressed as a percentage (for example, `10` or `10.0` = 10%). If a configuration uses a fraction instead (for example, `0.10`), the implementation SHOULD accept an alternate key `na_threshold_fraction` for backwards compatibility. When both `na_threshold` (percent) and `na_threshold_fraction` are present, prefer `na_threshold` (percent).
- Warmup: rows earlier than the long SMA window are excluded from signal generation and counted in `warmup_drop_counts` per ticker; signal detection requires full-window SMA values at both T and T-1 (i.e., SMA_short[T] and SMA_long[T] and SMA_short[T-1] and SMA_long[T-1] must all be present and built from full windows).

Sentinel row semantics (deterministic CSV rows)

Implementations MUST emit a row in the canonical CSV for every combination of (Stock, Period_Label, Short_SMA, Long_SMA, Holding_Period) in the requested grid even when zero signals were detected. For zero-signal or fully-discarded groups, the following sentinel values are required:

- `Num_Signals = 0`
- `Num_Signals_Used = 0`
- `Low_Power_Flag = TRUE`
- `P_Value = NA`
- `Adj_P_Value_BH = NA`
- `Mean_CI_lower = NA`, `Mean_CI_upper = NA`
- `Mean_Return = NA`, `Std_Dev_Return = NA`, `Median_Return = NA`
- `Mean_Max_Drawdown = NA`, `Win_Rate = NA`
- `Notes = "no_signals"`

CSV encoding and sentinel values

Implementations MUST follow a deterministic CSV encoding policy:

- Numeric missing values MUST be emitted as empty CSV fields (no quoted literal such as the string "NA").
- Textual sentinel values (for `Notes` and other text fields) SHOULD use the literal string `NA` when no message is present, and use the exact strings in `data-model.md` for sentinel notes (e.g., "no_signals", "discarded_insufficient_forward_data").
- The authoritative mapping of column names, order, types, and sentinel semantics MUST be kept in `contracts/csv-schema.md`. If that file differs from this document, `contracts/csv-schema.md` is authoritative for CSV producers and consumers.

Low-power rule (sample-size threshold)

As documented in `research.md`, if `Num_Signals_Used < 5` then the aggregator MUST set `P_Value = NA` and `Low_Power_Flag = TRUE`. This rule applies in addition to the `Insufficient_History` flag driven by percent_missing_before_fill.

State transitions

- Raw download -> cleaned `TickerTimeSeries` (forward-fill applied) -> SMA computation (full-window rolling) -> signal detection (requires SMA at T and T-1) -> forward-return/drawdown computation (per holding period; signals lacking forward data for an N-day holding period are discarded for that N and noted) -> per-holding-period aggregation -> BH-FDR adjustment (per holding period) -> outputs (CSV, PNG, markdown) and run_manifest.

Period / Cohort assignment

The canonical cohort generator used by the run is deterministic and inclusive: given `start_date` and `end_date` it will slice the range into contiguous, non-overlapping cohorts of equal length where possible. For the canonical run the cohorts are hard-coded as follows (inclusive start and end dates):

- `2004-01-01` to `2008-12-31`
- `2009-01-01` to `2013-12-31`
- `2014-01-01` to `2018-12-31`
- `2019-01-01` to `2023-12-31`

Signals are assigned to the cohort in which `signal_date` (T) occurs. Forward returns used to evaluate a signal MAY use data outside the cohort boundary as long as the required N trading days exist in the ticker's cleaned timeline; if insufficient forward data exists the signal is discarded for that holding period and a note `discarded_insufficient_forward_data` should be added to the SignalRecord.notes field.
