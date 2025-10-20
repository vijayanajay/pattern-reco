# Canonical CSV Schema: sma_crossover_analysis_results_{RUN_TS}.csv

This file defines the exact, case-sensitive CSV header, expected types, and sentinel values used for the canonical aggregated results CSV produced by the SMA crossover analysis. Use this schema as the single source of truth for CSV writers and downstream consumers.

Notes on serialization


Exact columns and order

1. `Stock` (string)
2. `Period_Label` (string)  # e.g., `2004-01-01_to_2008-12-31`
3. `Period_Start` (date, `YYYY-MM-DD`)
4. `Period_End` (date, `YYYY-MM-DD`)
5. `Short_SMA` (int)
6. `Long_SMA` (int)
7. `Holding_Period` (int)
8. `Num_Signals` (int)
9. `Num_Signals_Used` (int)
10. `Low_Power_Flag` (boolean: `TRUE`/`FALSE`)
11. `Mean_Return` (float or `NA`)
12. `Std_Dev_Return` (float or `NA`)
13. `Median_Return` (float or `NA`)
14. `Mean_CI_Lower` (float or `NA`)
15. `Mean_CI_Upper` (float or `NA`)
16. `Mean_Max_Drawdown` (float or `NA`)
17. `Win_Rate` (float in [0,1] or `NA`)
18. `P_Value` (float in [0,1] or `NA`)
19. `Adj_P_Value_BH` (float in [0,1] or `NA`)
20. `Skew` (float or `NA`)
21. `Kurtosis` (float or `NA`)
22. `Buy_and_Hold_Return` (float or `NA`)
23. `Overlap_Mode` (string)
24. `Entry_Mode` (string)
25. `Notes` (string)  # sentinel `no_signals` for groups with 0 signals
26. `Timestamp` (string RUN_TS)

Sentinel row example (zero signals)


Column naming & casing guidance


Diagnostics CSV: `sma_crossover_analysis_diagnostics_{RUN_TS}.csv`

Columns (recommended):

Logs


# contracts/csv-schema.md
This file is the canonical CSV schema for the SMA crossover analysis outputs. It must be kept in sync with `data-model.md`. Implementations MUST use these headers, types, and sentinel rules when writing CSV artifacts.
General CSV encoding rules
- Numeric missing values MUST be emitted as empty CSV fields (no quoted literal such as the string "NA").
- Textual sentinel values SHOULD use the literal string `NA` when no message is present, and use the exact strings defined in `data-model.md` for warning/notes values (for example, `no_signals`, `discarded_insufficient_forward_data`).
- Timestamps in filenames and `timestamp` fields MUST use the canonical RUN_TS format `YYYYMMDDTHHMMSSZ` (UTC compact ISO).
Files and canonical columns (minimal template — expand before production)
1) `sma_crossover_analysis_results_{RUN_TS}.csv` (one row per Stock, Period_Label, Short_SMA, Long_SMA, Holding_Period)
Columns (suggested names — MUST be finalized and ordered here before production):
- Stock
- Period_Label
- Period_Start
- Period_End
- Short_SMA
- Long_SMA
- Holding_Period
- Num_Signals
- Num_Signals_Used
- Low_Power_Flag
- Mean_Return
- Std_Dev_Return
- Median_Return
- Mean_CI_Lower
- Mean_CI_Upper
- Mean_Max_Drawdown
- Win_Rate
- P_Value
- Adj_P_Value_BH
- Skew
- Kurtosis
- Buy_and_Hold_Return
- Overlap_Mode
- Entry_Mode
- Notes
- Timestamp
2) `sma_crossover_analysis_diagnostics_{RUN_TS}.csv`
Suggested columns:
- Ticker
- Period_Label
- Percent_Missing_Before_Fill
- Na_Filled_Count
- Insufficient_History (boolean or flag)
- Warmup_Drop_Count
- Notes
3) Logs CSVs (e.g., `logs/na_fill_warnings_{RUN_TS}.csv`, `logs/warmup_drop_counts_{RUN_TS}.csv`)
Suggested columns: follow the diagnostics shape and include `Ticker`, `Date`, and the relevant metric.
IMPORTANT: This file is a template and must be expanded to include exact column order, exact column names (case-sensitive), types, and sentinel value rules before production runs. The implementation MUST read this file during code review or test creation to ensure parity.


