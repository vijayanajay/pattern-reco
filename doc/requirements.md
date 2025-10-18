## **Requirements Document: SMA Crossover Signal Efficacy Analysis**

**Version:** 1.0
**Date:** 2023-10-27
**Author:** System (emulating K. Nadh & G. Hinton's mindset)

### **1. Introduction & Vision**

#### **1.1. Project Goal**
To conduct a rigorous, empirical analysis of the short-term performance of Simple Moving Average (SMA) crossover signals on a curated list of Indian large-cap stocks over a 20-year period.

#### **1.2. Guiding Philosophy**
This analysis is an exploratory experiment, not a backtest of a complete trading system. The core philosophy is to:
*   **Embrace Simplicity (Nadh):** Use a basic, well-understood strategy (SMA crossover) and present findings in a clear, visually intuitive manner (heatmaps). Avoid unnecessary complexity.
*   **Demand Statistical Rigor (Hinton):** Do not accept results at face value. Every observation must be tested for statistical significance (t-test, p-value) and consistency across different market conditions (time periods) and assets (stocks). The goal is to identify robust patterns, not random flukes.

#### **1.3. Scope**
*   **In-Scope:** Data acquisition, signal generation for 10 pre-defined SMA pairs, analysis of post-signal returns and drawdowns over 3 fixed holding periods, statistical testing, benchmarking against buy-and-hold, and generation of specified outputs (CSV, Markdown report, PNG heatmaps).
*   **Out-of-Scope:**
    *   Developing a complete, tradable backtesting system (i.e., no sell signals, position sizing, or portfolio management).
    *   Inclusion of transaction costs, slippage, or taxes.
    *   Exhaustive parameter optimization (brute-forcing all possible X and Y values).
    *   Use of any machine learning, neural networks, or complex predictive models.
    *   Implementation of advanced risk management or market regime filters (e.g., NIFTY > 200d SMA).

---

### **2. System & Data Requirements**

#### **2.1. Environment**
*   **Language:** Python 3.x
*   **Core Libraries:**
    *   `yfinance`: For downloading historical stock data.
    *   `pandas`: For data manipulation and analysis.
    *   `numpy`: For numerical operations.
    *   `scipy`: Specifically `scipy.stats` for t-tests.
    *   `matplotlib` & `seaborn`: For generating heatmap visualizations.

#### **2.2. Data Source & Specifications**
*   **Source:** Yahoo Finance.
*   **Asset List (15 Tickers):**
    1.  `RELIANCE.NS`
    2.  `TCS.NS`
    3.  `HDFCBANK.NS`
    4.  `INFY.NS`
    5.  `HINDUNILVR.NS`
    6.  `ITC.NS`
    7.  `LT.NS`
    8.  `SBIN.NS`
    9.  `BHARTIARTL.NS`
    10. `SUNPHARMA.NS`
    11. `TATAMOTORS.NS`
    12. `TATASTEEL.NS`
    13. `ASIANPAINT.NS`
    14. `WIPRO.NS`
    15. `YESBANK.NS` (The "Fallen Angel")
*   **Time Period:** January 1, 2004, to December 31, 2023.
*   **Price Data:** `Adj Close` (Adjusted Close) to account for dividends and stock splits.

---

### **3. Functional Requirements (FR)**

**FR1: Data Acquisition and Preprocessing**
*   **Description:** The system must download historical daily price data for the specified list of 15 tickers for the defined 20-year period. It must handle missing data points gracefully.
*   **Details:**
    *   Use `yfinance` to download data for all tickers.
    *   Select only the `Adj Close` column for calculations.
    *   Check for missing values (NaNs) in the downloaded data for each stock.
    *   Apply a forward-fill (`ffill()`) method to populate sporadic NaNs.
    *   The system must log a warning for each stock indicating how many NaN values were filled. This highlights potential data quality issues.
*   **Definition of Done (DoD):** A function exists that, when given the list of tickers and date range, returns a single pandas DataFrame with tickers as columns and a `DatetimeIndex`, containing cleaned `Adj Close` prices. Warnings for filled NaNs are printed to the console.

**FR2: Time Period Segmentation**
*   **Description:** The system must segment the 20-year dataset into four distinct 5-year periods for cohort analysis.
*   **Details:** The data must be split into the following labeled periods:
    *   `2004-01-01` to `2008-12-31`: "2004-2009 (Pre-Crisis Bull Market & Crash)"
    *   `2009-01-01` to `2013-12-31`: "2009-2014 (Post-Crisis Recovery)"
    *   `2014-01-01` to `2018-12-31`: "2014-2019 (Modi-Era Bull Run)"
    *   `2019-01-01` to `2023-12-31`: "2019-2024 (COVID Volatility & New Highs)"
*   **DoD:** The main analysis loop iterates through these four distinct, labeled data slices. All subsequent calculations and reports are grouped by these periods.

**FR3: Strategy Signal Generation**
*   **Description:** For each stock and each time period, the system must calculate SMAs and identify "golden cross" buy signals.
*   **Details:**
    *   **SMA Pairs (X, Y):** The system will iterate through the following 10 pre-defined pairs: `(10, 20), (20, 50), (50, 100), (50, 150), (50, 200), (100, 200), (10, 50), (20, 100), (20, 200), (100, 150)`.
    *   **Signal Condition:** A buy signal is generated on day `T` if `SMA_X[T] > SMA_Y[T]` AND `SMA_X[T-1] <= SMA_Y[T-1]`.
    *   **Signal Uniqueness:** This condition inherently ensures only the first day of a crossover is registered as a signal, preventing repeated signals while the short SMA remains above the long SMA.
*   **DoD:** A function exists that takes a stock's price series and an (X, Y) pair, and returns a list of dates on which a buy signal occurred.

**FR4: Post-Signal Performance Calculation**
*   **Description:** For each generated signal, the system must calculate the forward returns and maximum drawdown for fixed holding periods without look-ahead bias.
*   **Details:**
    *   **Holding Periods:** 10, 15, and 20 trading days.
    *   **Entry Price:** The `Adj Close` price on the day of the signal (`T`).
    *   **Return Calculation:** For a holding period of `N` days, Return = `(Price[T+N] - Price[T]) / Price[T]`.
    *   **Max Drawdown Calculation:** For a holding period of `N` days, the drawdown is calculated *from the entry price*. Max Drawdown = `(Lowest Price in [T+1, T+N] - Price[T]) / Price[T]`. This will always be a negative number or zero.
    *   **Edge Case Handling:** If a signal occurs such that the holding period (e.g., 20 days) extends beyond the available data for that time segment, that signal is discarded for that specific holding period calculation.
*   **DoD:** For every signal, a record is created containing the signal date, entry price, and the calculated returns and max drawdowns for 10, 15, and 20-day holding periods.

**FR5: Aggregation and Statistical Analysis**
*   **Description:** The system must aggregate the results of all signals for each unique combination of (Stock, Time Period, SMA Pair, Holding Period) and compute a set of summary metrics.
*   **Details:** For each group, the following metrics must be calculated:
    1.  **Number of Signals:** Total count of valid signals.
    2.  **Mean Return:** Average of all calculated returns.
    3.  **Std. Dev. of Returns:** Standard deviation of all calculated returns.
    4.  **Mean Max Drawdown:** Average of all calculated max drawdowns.
    5.  **Win Rate:** Percentage of signals where the return was `> 0`.
    6.  **P-Value:** The p-value from a one-sample t-test (`scipy.stats.ttest_1samp`) comparing the distribution of returns against a population mean of 0.
*   **DoD:** A structured dataset (e.g., a pandas DataFrame) is created where each row represents a unique combination of the grouping keys and contains all the calculated metrics.

**FR6: Buy-and-Hold Benchmark Calculation**
*   **Description:** The system must calculate the simple buy-and-hold return for each stock within each of the four 5-year time periods to serve as a baseline for comparison.
*   **Details:** For each stock and each period, the return is calculated as `(Last Day's Price - First Day's Price) / First Day's Price`.
*   **DoD:** The buy-and-hold return for every stock and every 5-year period is calculated and stored for inclusion in the final report.

**FR7: Output Generation - CSV Files**
*   **Description:** The system must save the detailed aggregated results into a comprehensive CSV file.
*   **Details:**
    *   A single CSV file named `sma_crossover_analysis_results.csv` will be generated.
    *   **Columns:** `Stock`, `Period_Label`, `Short_SMA`, `Long_SMA`, `Holding_Period`, `Num_Signals`, `Mean_Return`, `Std_Dev_Return`, `Mean_Max_Drawdown`, `Win_Rate`, `P_Value`.
*   **DoD:** The specified CSV file is created in the output directory and is readable by standard spreadsheet software.

**FR8: Output Generation - Heatmap Visualization**
*   **Description:** The system must generate and save a heatmap visualizing the performance of the SMA pairs.
*   **Details:**
    *   A heatmap will be generated showing the **average Mean Return across ALL stocks and ALL time periods** for each of the 10 SMA pairs.
    *   **X-axis:** Short SMA period (X).
    *   **Y-axis:** Long SMA period (Y).
    *   **Color:** The color of each cell will represent the average Mean Return. A divergent colormap (e.g., `RdYlGn`) should be used.
    *   **Appearance:** The heatmap will be sparse, only showing cells for the 10 tested (X, Y) combinations. This is the intended behavior.
    *   **File:** The plot must be saved as a high-resolution PNG file named `heatmap_avg_mean_return.png`.
*   **DoD:** The specified PNG file is created in the output directory and clearly visualizes the performance landscape of the tested SMA pairs.

**FR9: Output Generation - Markdown Summary Report**
*   **Description:** The system must generate a single text/markdown file that synthesizes the findings in a human-readable format.
*   **Details:** The report, named `analysis_summary.md`, must contain:
    1.  **Executive Summary:** A brief overview of the project's goal and key findings.
    2.  **Overall Performance Heatmap:** The generated heatmap image embedded or referenced.
    3.  **Consistency Analysis:** A section discussing which, if any, (X, Y) parameter regions showed consistent profitability across multiple stocks and time periods. Highlight parameters with both positive mean returns and low p-values (< 0.05).
    4.  **Top Performers:** A small table showing the top 5 best-performing combinations (Stock, Period, SMA Pair, Holding Period) based on Mean Return, provided the p-value is significant.
    5.  **The "Fallen Angel" Case (YESBANK.NS):** A specific analysis of how the SMA strategies performed on Yes Bank, particularly during its period of collapse, to serve as a cautionary tale against strategy over-fitting.
    6.  **Benchmark Comparison:** A summary comparing the strategy's mean returns to the buy-and-hold returns for the corresponding periods.
    7.  **Signal Frequency Analysis:** A brief comment on which SMA pairs generated the most and fewest signals.
*   **DoD:** A well-formatted `analysis_summary.md` file is generated containing all the specified sections.

