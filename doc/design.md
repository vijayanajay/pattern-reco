# Design Document

## Overview

The system is a single Python pipeline that takes a YAML config and produces auditable results. Two modes: refresh data (optional), run backtest (primary). Everything else is noise.

Core principle: fit on in-sample, test on out-of-sample, never touch the final holdout. No complexity without proof. No parameters without justification. No results without costs.

## Architecture

### Core Flow

```
Config → Load Data → Select Universe → Split Time → Fit Params (IS) → Generate Signals (OOS) → Simulate Trades → Report Results
```

That's it. Everything else is implementation detail.

### Directory Structure

```
├── cli.py                    # Entry point
├── src/
│   ├── config.py            # Config loading & validation
│   ├── data.py              # Data fetch, snapshot, diff
│   ├── features.py          # Returns, gaps, rolling stats
│   ├── detectors.py         # Gap-Z detector + registry
│   ├── backtest.py          # Walk-forward + execution + portfolio
│   ├── metrics.py           # Performance evaluation
│   └── io.py                # Reporting + manifest
├── tests/
├── runs/                    # Output directory
└── data/snapshots/          # Parquet files
```

Fewer files = fewer bugs. Each file has a single responsibility.

## Critical Components

### Configuration

Single YAML file controls everything. No CLI flags, no environment variables, no hidden state.

```python
def load_config(path: str) -> dict:
    """Load and validate YAML config. Fail hard on errors."""
    
def validate_config(config: dict) -> None:
    """Check dates, parameters, paths. Fail fast."""
```

**Missing pieces that kill projects:**
- Config versioning (what happens when you change the schema?)
- Parameter bounds checking (prevent garbage in)
- Path validation (prevent directory traversal)
- Date validation (ensure t0 + holdout < end)

### Data Management

```python
def fetch_and_snapshot(symbols: List[str], config: dict) -> None:
    """Fetch from yfinance, save to parquet with metadata."""
    
def load_snapshots(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load existing snapshots. Fail if missing and refresh=False."""
    
def generate_diff_report(old_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """Compare datasets, return diff summary."""
```

**Critical missing pieces:**
- NSE holiday calendar (don't assume weekends = non-trading days)
- Timezone handling (yfinance returns UTC, NSE trades in IST)
- Corporate actions detection (stock splits will destroy your returns)
- Data quality checks (missing days, zero volume, price gaps > 50%)
- Backup strategy (what if yfinance is down?)

**Parquet metadata must include:**
- SHA256 hash of the data
- yfinance version (API changes break everything)
- Fetch timestamp (for debugging)
- Git commit hash (for reproducibility)

### Universe Selection

```python
def select_universe(all_symbols: List[str], t0: date, lookback_years: int, size: int) -> List[str]:
    """Select top N by median daily turnover. Freeze at t0."""
```

**Critical details:**
- Turnover = Close * Volume (in INR, not shares)
- Use median, not mean (outliers will fool you)
- Exclude penny stocks (< ₹10) and illiquid names (< ₹1cr daily turnover)
- Handle stock splits in turnover calculation
- Document survivorship bias (delisted stocks missing from yfinance)

### Features (Keep It Simple)

```python
def compute_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Daily returns from adjusted close. Handle splits/dividends."""
    
def compute_gaps(data: pd.DataFrame) -> pd.Series:
    """(Open - PrevClose) / PrevClose. The core signal."""
    
def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """(x - mean) / std. Use median/MAD for robustness."""
```

**What NOT to build:**
- Complex technical indicators (RSI, MACD, etc.)
- Intraday features (we're daily only)
- Fundamental ratios (P/E, etc.)
- Sentiment indicators
- Macro features

**What you MUST handle:**
- Stock splits (use adjusted close)
- Missing data (forward fill? drop? explicit choice)
- Outliers (winsorize at 99th percentile?)
- Minimum history requirements (need 252 days for 1-year stats)

### Gap-Z Detector (MVP Only)

```python
def gap_z_signal(data: pd.DataFrame, window: int, k_low: float) -> pd.Series:
    """
    gaps = (Open - PrevClose) / PrevClose
    z_score = (gaps - rolling_mean) / rolling_std
    signal = z_score < k_low  # Long on negative gaps
    """
```

**Parameter grid (keep tiny):**
- window: [20, 60] (2 values, not 20)
- k_low: [-1.0, -2.0] (2 values, not 10)
- max_hold: [22] (fixed)

**Parameter fitting:**
- Objective: median trade return post-costs
- Constraint: hit rate > 40%
- Method: exhaustive grid (4 combinations total)
- Fit on IS, freeze for OOS

**Don't build a "framework" yet. Build one detector that works.**

### Walk-Forward Validation

```python
def create_splits(start_date: date, end_date: date, holdout_years: int) -> List[dict]:
    """
    3-year IS, 1-year OOS, roll annually
    Final 2 years = sacred holdout
    """
```

**Example timeline:**
- Data: 2010-2025 (15 years)
- Holdout: 2023-2025 (2 years, never touched)
- Available: 2010-2023 (13 years)
- Splits:
  - Split 1: IS 2010-2013, OOS 2013-2014
  - Split 2: IS 2011-2014, OOS 2014-2015
  - ...
  - Split 9: IS 2018-2021, OOS 2021-2022

**Critical validation:**
- No gap between IS and OOS (data leakage)
- No overlap between splits (independence)
- Sufficient history for parameter fitting
- Calendar alignment (end on trading days)

### Execution Simulation

```python
def simulate_trade(signal_date: date, data: pd.DataFrame, max_hold: int) -> dict:
    """
    Entry: next day's open (if within circuit limits)
    Exit: max_hold days later or earlier if conditions met
    Costs: fees + slippage
    """
```

**Execution realism (this is where most backtests lie):**
- Entry: next open after signal (no same-day fills)
- Circuit guard: reject if open > ±10% from prev close
- Slippage model:
  - Gap < 2%: 5 bps
  - Gap 2-5%: 10 bps  
  - Gap > 5%: 20 bps
- Fees: 10 bps per side (conservative for retail)
- Position size: equal weight (₹1L per position)

**What kills backtests:**
- Assuming you can trade at close prices
- Ignoring bid-ask spreads
- Perfect fills on illiquid stocks
- No slippage on large orders

### Portfolio Rules

```python
def select_positions(signals: dict, current_positions: int, max_concurrent: int) -> List[str]:
    """
    Deterministic selection when oversubscribed:
    1. Signal strength (most negative z-score first)
    2. Liquidity (highest turnover first)  
    3. Ticker alphabetical (tie-breaker)
    """
```

**Portfolio constraints:**
- Max 5 concurrent positions (capacity limit)
- Equal weight (₹1L each, not 1/N of capital)
- No pyramiding (one position per stock)
- Re-entry lockout until position closes

**Why these rules matter:**
- Prevents overfitting to specific stocks
- Ensures tradeable position sizes
- Makes selection deterministic and auditable

## Data Models (Minimal)

```python
# Trade record
{
    "symbol": "RELIANCE.NS",
    "entry_date": "2023-01-15", 
    "entry_price": 2450.0,
    "exit_date": "2023-02-10",
    "exit_price": 2520.0,
    "hold_days": 18,
    "pnl": 70.0,
    "fees": 4.9,
    "slippage": 2.5,
    "net_pnl": 62.6
}

# Unfilled signal
{
    "symbol": "INFY.NS",
    "signal_date": "2023-01-20",
    "reason": "circuit_guard",  # or "capacity_limit"
    "attempted_price": 1650.0
}
```

Don't over-engineer data structures. Use dicts and DataFrames.

## Error Handling (Fail Fast)

**Hard failures (exit immediately):**
- Missing config file
- Invalid YAML syntax
- Missing data snapshots when refresh=False
- Empty universe after filtering
- Insufficient history for walk-forward splits

**Soft warnings (log and continue):**
- Minor data diffs within epsilon
- Unfilled trades due to circuit guards
- Missing data for specific dates

**No silent failures. Every error gets logged with timestamp and context.**

## Testing (Test What Matters)

**Unit tests:**
- Gap calculation with known prices
- Z-score computation with synthetic data  
- Walk-forward split boundaries
- Circuit guard logic
- Position selection determinism

**Integration test:**
- End-to-end with 2 stocks, 2 years of data
- Same config + seed = identical results

**Property tests:**
- No lookahead: future data never affects past signals
- Determinism: same inputs = same outputs
- Cost application: every trade has fees + slippage

**Don't test:**
- pandas/numpy internals
- yfinance API responses
- File I/O edge cases

**Test the logic, not the libraries.**

## Critical Implementation Details

**Performance:**
- Vectorize everything (pandas/numpy)
- Process one stock at a time (memory efficiency)
- Cache rolling calculations within each stock

**Determinism:**
- Set random seed from config
- Sort everything that could be non-deterministic
- SHA256 hash all inputs and outputs
- Use UTC timestamps, convert to IST for display

**What will break in production:**
- yfinance rate limits (add delays)
- Network timeouts (retry with backoff)
- Memory usage with 500+ stocks (process in batches)
- Timezone confusion (be explicit everywhere)

**Security:**
- No network calls during backtest execution
- Validate all file paths (prevent directory traversal)
- Sanitize config inputs

The design is deliberately minimal. Build the smallest thing that works, then iterate based on real usage.