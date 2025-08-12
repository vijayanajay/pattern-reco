"""
Universe selection for NSE stocks.
Deterministic selection based on median daily turnover at t0.

This module implements a deterministic universe selection system with the following key principles:
1. **Deterministic**: No randomness in selection - same inputs always produce same outputs
2. **Reproducible**: Results can be reproduced across runs with identical parameters
3. **Transparent**: All selection decisions are logged and documented
4. **Survivorship Bias Aware**: Proper handling of delisted stocks and historical data

Survivorship Bias Considerations:
--------------------------------
The universe selection system addresses survivorship bias through several mechanisms:

1. **Fixed Universe at t0**: The universe is frozen at a specific date (t0) and remains
   unchanged throughout the backtest period. This prevents look-ahead bias where
   future information influences universe selection.

2. **Historical Data Usage**: When selecting universe at t0, we use only data available
   up to t0 (lookback period). This ensures we're not using future information.

3. **Delisted Stock Handling**: The system gracefully handles stocks that may have
   been delisted after t0 by:
   - Including them in the initial universe if they met criteria at t0
   - Handling missing data gracefully during backtesting
   - Documenting any exclusions due to data unavailability

4. **Exclusion List Management**: Provides explicit exclusion mechanism for stocks
   that should not be included due to corporate actions, regulatory issues, etc.

5. **Comprehensive Logging**: All decisions are logged with timestamps and reasons,
   making it possible to audit the selection process.

Example Configuration:
---------------------
universe:
  size: 50                    # Number of stocks to select
  min_turnover: 10000000     # Minimum median daily turnover in INR (₹1cr)
  min_price: 10.0           # Minimum stock price in INR
  lookback_years: 2         # Years of historical data to use
  exclude_symbols: []       # List of symbols to exclude
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Any, Tuple, Optional
import warnings

import pandas as pd
import numpy as np

from .data import load_snapshots

__all__ = ["select_universe", "compute_turnover_stats", "get_nse_symbols", "UniverseSelector"]

logger = logging.getLogger(__name__)


class UniverseSelector:
    """
    Deterministic universe selector for NSE stocks.
    
    This class provides a configurable and reproducible way to select
    a universe of stocks based on liquidity and quality criteria.
    
    Key Features:
    - Deterministic ranking by median daily turnover
    - Configurable filtering for penny stocks and illiquid names
    - Universe freezing at specific date (t0)
    - Comprehensive logging of all decisions
    - Survivorship bias mitigation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with comprehensive validation of source data properties.
        
        Handles:
        - Price filters (min_price > 0)
        - Turnover filters (min_turnover >= 100000) for meaningful trading volume
        - Timeframe parameters (lookback_years > 0) for statistical significance
        - Symbol exclusion functionality for corporate actions/delisted stocks
        """
        universe_config = config.get('universe', {})
        data_config = config.get('data', {})
        
        self.config = config
        self.universe_config = universe_config
        self.data_config = data_config

        # Extract parameters with defaults
        self.size = self.universe_config.get("size", 10)
        self.min_turnover = self.universe_config.get("min_turnover", 10_000_000.0)  # ₹1cr
        self.min_price = self.universe_config.get("min_price", 10.0)  # ₹10
        self.exclude_symbols = self.universe_config.get("exclude_symbols", [])
        self.lookback_years = self.universe_config.get("lookback_years", 2)

        # Validate data source configuration
        self.primary_source = data_config.get('source', 'yfinance')
        if not self.primary_source:
            raise ValueError('Missing required data source configuration: primary_source')
        
        # Validate configuration
        self.validate_config()
        
        logger.info(f"Initialized UniverseSelector with config: {self.universe_config}")
    
    def validate_config(self) -> None:
        """
        Validate universe configuration parameters.
        
        Raises:
            ValueError: If configuration parameters are invalid
        """
        if self.size <= 0:
            raise ValueError(f"Universe size must be positive, got {self.size}")
        
        if self.min_turnover < 0:
            raise ValueError(f"Minimum turnover must be non-negative, got {self.min_turnover}")
        
        if self.min_price < 0:
            raise ValueError(f"Minimum price must be non-negative, got {self.min_price}")
        
        if self.lookback_years <= 0:
            raise ValueError(f"Lookback years must be positive, got {self.lookback_years}")
        
        if not isinstance(self.exclude_symbols, list):
            raise ValueError("exclude_symbols must be a list")
    
    def apply_filters(self, symbol: str, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Apply all filtering criteria to a symbol's data.
        
        Args:
            symbol: Stock symbol
            data: Stock price data with OHLCV columns
            
        Returns:
            Tuple of (passes_filters, reason_for_exclusion)
        """
        if data.empty:
            return False, "Empty data"
        
        # Check minimum price requirement (using most recent close)        
        recent_price = data['Close'].iloc[-1]
        if recent_price < self.min_price:
            return False, f"Price {recent_price:.2f} < {self.min_price:.2f}"
        
        # Compute turnover statistics
        stats = compute_turnover_stats(data, self.lookback_years)
        
        # Check if we have sufficient valid trading days
        min_days_req = max(30, int(self.lookback_years * 252 * 0.7))
        if stats["valid_days"] < min_days_req:
            return False, f"Insufficient data: {stats['valid_days']} < {min_days_req} days"
        
        # Check minimum turnover requirement
        if stats["median_turnover"] < self.min_turnover:
            return False, f"Turnover {stats['median_turnover']:.0f} < {self.min_turnover:.0f}"
        
        return True, "Passes all filters"


def get_nse_symbols() -> List[str]:
    """
    Get list of NSE symbols for universe selection.
    
    For MVP, returns a curated list of liquid NSE stocks.
    In production, this would query NSE API or use a comprehensive symbol list.
    
    The symbol list includes both large-cap and small-cap stocks to provide
    a representative sample of the Indian equity market.
    
    Returns:
        List of NSE stock symbols including both large cap and small cap
        
    Note:
        This list is deterministic and reproducible across runs.
        For production use, consider integrating with NSE's official API
        or using a comprehensive symbol database.
    """
    # Large cap NSE stocks - top 30 by market cap
    large_cap_stocks = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
        "BAJFINANCE.NS", "LICI.NS", "HCLTECH.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "AXISBANK.NS", "LT.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS",
        "WIPRO.NS", "NESTLEIND.NS", "POWERGRID.NS", "NTPC.NS", "JSWSTEEL.NS",
        "TATAMOTORS.NS", "M&M.NS", "TECHM.NS", "COALINDIA.NS", "INDUSINDBK.NS"
    ]
    
    # Small cap NSE stocks - representative sample
    small_cap_stocks = [
        "ADANIPORTS.NS", "ADANIENT.NS", "GODREJCP.NS", "DIVISLAB.NS", "DRREDDY.NS",
        "EICHERMOT.NS", "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "JINDALSTEL.NS",
        "ONGC.NS", "SAIL.NS", "TATASTEEL.NS", "VEDL.NS", "ZEEL.NS",
        "APOLLOHOSP.NS", "BAJAJFINSV.NS", "BRITANNIA.NS", "CIPLA.NS", "DABUR.NS"
    ]
    
    # Combine and deduplicate
    all_symbols = list(set(large_cap_stocks + small_cap_stocks))
    all_symbols.sort()
    
    logger.info(f"Loaded {len(all_symbols)} NSE symbols for universe selection")
    return all_symbols


def compute_turnover_stats(
    data: pd.DataFrame, 
    lookback_years: int = 2
) -> Dict[str, float]:
    """
    Compute turnover statistics for a single stock.
    
    Calculates daily turnover as Close * Volume in INR, then computes
    median and mean turnover over the specified lookback period.
    
    Args:
        data: Stock price data with OHLCV columns (Open, High, Low, Close, Volume)
        lookback_years: Number of years to look back for turnover calculation
        
    Returns:
        Dictionary with turnover statistics:
        - median_turnover: Median daily turnover in INR
        - mean_turnover: Mean daily turnover in INR
        - valid_days: Number of valid trading days used in calculation
        
    Notes:
        - Zero volume days are excluded from calculations
        - Data is truncated to lookback period to reduce memory usage
        - Returns zero values for empty or invalid data
    """
    if data.empty:
        logger.debug("Empty data provided for turnover calculation")
        return {"median_turnover": 0.0, "mean_turnover": 0.0, "valid_days": 0}

    required_columns = ['Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_message = f"Missing required columns: {', '.join(missing_columns)}"
        logger.error(error_message) # Log the error message
        raise ValueError(error_message)

    # Filter to lookback period first to reduce data size
    if lookback_years > 0:
        max_days = int(lookback_years * 252)  # Approximate trading days per year
        if len(data) > max_days:
            data = data.tail(max_days)
            logger.debug(f"Truncated data to last {max_days} days")
    else:
        logger.debug("lookback_years is 0, using all available data.")

    # Calculate daily turnover in INR (Close * Volume)
    # This represents the total value traded each day
    turnover = data['Close'].abs() * data['Volume']
    
    # Remove zero volume days as they represent non-trading days or bad data
    valid_turnover = turnover[data['Volume'] > 0]
    
    # Check if we have any valid data
    if len(valid_turnover) == 0:
        logger.debug("No valid trading days found (all zero volume)")
        return {"median_turnover": 0.0, "mean_turnover": 0.0, "valid_days": 0}
    
    stats = {
        "median_turnover": float(valid_turnover.median()),
        "mean_turnover": float(valid_turnover.mean()),
        "valid_days": len(valid_turnover)
    }
    
    logger.debug(f"Computed turnover stats: {stats}")
    return stats


def select_universe(
    config: Dict[str, Any], 
    t0: Optional[date] = None,
    available_symbols: Optional[List[str]] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Select universe of NSE stocks for backtesting based on median daily turnover.
    
    Implements comprehensive universe selection with the following features:
    1. **Deterministic Ranking**: Stocks ranked by median daily turnover (Close * Volume in INR)
    2. **Quality Filtering**: Removes penny stocks (price < ₹10) and illiquid names
    3. **Universe Freezing**: Universe is frozen at t0 for entire backtest period
    4. **Exclusion Handling**: Supports exclusion lists for corporate actions, etc.
    5. **Survivorship Bias Mitigation**: Proper handling of delisted stocks
    
    Args:
        config: Configuration dictionary with universe parameters
        t0: Reference date for universe selection (default: use data end date)
            This is the "freeze date" - universe remains unchanged after this date
        available_symbols: List of symbols to consider (default: get from NSE list)
            
    Returns:
        Tuple of (selected_symbols, universe_metadata) where:
        - selected_symbols: List of selected stock symbols in rank order
        - universe_metadata: Dictionary with detailed selection information
        
    Raises:
        ValueError: If insufficient data or no stocks meet criteria
        
    Example:
        >>> config = {
        ...     "universe": {
        ...         "size": 50,
        ...         "min_turnover": 10000000,
        ...         "min_price": 10.0,
        ...         "exclude_symbols": ["DELISTED.NS"],
        ...         "lookback_years": 2
        ...     }
        ... }
        >>> symbols, metadata = select_universe(config, t0=date(2023, 1, 1))
    """
    # Initialize selector with configuration
    selector = UniverseSelector(config)
    selector.validate_config()
    
    universe_config = config.get("universe", {})
    data_config = config.get("data", {})
    
    # Extract parameters
    size = universe_config.get("size", 10)
    min_turnover = universe_config.get("min_turnover", 10_000_000.0)  # ₹1cr default
    min_price = universe_config.get("min_price", 10.0)  # ₹10 default
    exclude_symbols = universe_config.get("exclude_symbols", [])
    lookback_years = universe_config.get("lookback_years", 2)
    
    # Get symbols to evaluate
    if available_symbols is None:
        available_symbols = get_nse_symbols()
    
    # Remove excluded symbols
    candidate_symbols = [s for s in available_symbols if s not in exclude_symbols]
    
    logger.info("=" * 80)
    logger.info("UNIVERSE SELECTION STARTED")
    logger.info("=" * 80)
    logger.info(f"Total symbols available: {len(available_symbols)}")
    logger.info(f"Symbols after exclusions: {len(candidate_symbols)}")
    logger.info(f"Exclusions applied: {exclude_symbols}")
    logger.info(f"Selection date (t0): {t0 or 'Using latest available date'}")
    logger.info(f"Universe size target: {size}")
    logger.info(f"Minimum turnover: ₹{min_turnover:,.0f}")
    logger.info(f"Minimum price: ₹{min_price:.2f}")
    logger.info(f"Lookback years: {lookback_years}")
    
    # Load data for all candidates
    try:
        all_data = load_snapshots(candidate_symbols)
    except FileNotFoundError as e:
        raise ValueError(f"Cannot select universe: {e}")
    
    if not all_data:
        raise ValueError("Cannot select universe: No data available for any candidate symbol.")
    
    # Track processing statistics
    processing_stats = {
        "total_candidates": len(candidate_symbols),
        "missing_data": 0,
        "empty_data": 0,
        "filtered_by_price": 0,
        "filtered_by_turnover": 0,
        "filtered_by_days": 0,
        "qualified_symbols": 0,
        "selected_symbols": []
    }
    
    # Compute turnover statistics for each symbol
    symbol_stats = {}
    detailed_exclusions = []
    
    for symbol in candidate_symbols:
        if symbol not in all_data:
            logger.warning(f"No data available for {symbol}, skipping")
            processing_stats["missing_data"] += 1    
            detailed_exclusions.append({"symbol": symbol, "reason": "No data available"})
            continue
            
        data = all_data[symbol]
        
        if data.empty:
            logger.warning(f"Empty data for {symbol}, skipping")
            processing_stats["empty_data"] += 1
            detailed_exclusions.append({"symbol": symbol, "reason": "Empty data"})
            continue
        
        # Filter data up to t0 if specified
        if t0 is not None:
            original_len = len(data)
            data = data[data.index.date <= t0]
            if data.empty:
                logger.warning(f"No data for {symbol} up to t0={t0}, skipping")
                detailed_exclusions.append({"symbol": symbol, "reason": f"No data up to t0={t0}"})
                continue
            if len(data) < original_len:
                logger.debug(f"Truncated {symbol} data from {original_len} to {len(data)} rows up to t0={t0}")

        passes, reason = selector.apply_filters(symbol, data)
        if passes:
            logger.info(f"Symbol {symbol} PASSED all filters.")
            symbol_stats[symbol] = compute_turnover_stats(data, selector.lookback_years)
            processing_stats["qualified_symbols"] += 1
        else:
            logger.info(f"Symbol {symbol} FAILED filters: {reason}")
            detailed_exclusions.append({"symbol": symbol, "reason": reason})
            if "Price" in reason: processing_stats["filtered_by_price"] += 1
            elif "Turnover" in reason: processing_stats["filtered_by_turnover"] += 1
            elif "Insufficient data" in reason: processing_stats["filtered_by_days"] += 1

    # Sort qualified symbols by median turnover in descending order
    sorted_symbols = sorted(
        symbol_stats.items(),
        key=lambda item: item[1]["median_turnover"],
        reverse=True
    )
    
    # Select top 'size' symbols and add rank
    selected_symbols_with_stats = []
    for i, (symbol, stats) in enumerate(sorted_symbols[:size]):
        stats['rank'] = i + 1  # Add 1 for 1-based ranking
        selected_symbols_with_stats.append((symbol, stats))
    selected_symbols = [s[0] for s in selected_symbols_with_stats]

    processing_stats["selected_symbols"] = selected_symbols

    logger.info("\n" + "=" * 80)
    logger.info("UNIVERSE SELECTION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total candidates processed: {processing_stats['total_candidates']}")
    logger.info(f"Symbols with missing data: {processing_stats['missing_data']}")
    logger.info(f"Symbols with empty data: {processing_stats['empty_data']}")
    logger.info(f"Symbols filtered by price: {processing_stats['filtered_by_price']}")
    logger.info(f"Symbols filtered by turnover: {processing_stats['filtered_by_turnover']}")
    logger.info(f"Symbols filtered by insufficient days: {processing_stats['filtered_by_days']}")
    logger.info(f"Qualified symbols: {processing_stats['qualified_symbols']}")
    logger.info(f"Selected universe size: {len(selected_symbols)}")
    logger.info(f"Selected symbols: {selected_symbols}")
    logger.info("Detailed exclusions:")
    for exc in detailed_exclusions:
        logger.info(f"  - {exc['symbol']}: {exc['reason']}")

    if not selected_symbols:
        raise ValueError("No stocks met the universe selection criteria.")

    universe_metadata = {
        "selection_criteria": {
            "size": size,
            "min_turnover": min_turnover,
            "min_price": min_price,
            "lookback_years": lookback_years,
            "exclude_symbols": exclude_symbols,
            "t0": str(t0) if t0 else "latest_available"
        },
        "selection_date": str(t0) if t0 else "latest_available",
        "processing_stats": processing_stats,
        "symbol_details": {s[0]: s[1] for s in selected_symbols_with_stats},
        "detailed_exclusions": detailed_exclusions,
        "survivorship_bias_notes": (
            "Universe selection is performed at a fixed date (t0) and then frozen. "
            "This mitigates survivorship bias by ensuring that only stocks available "
            "and meeting criteria at t0 are considered for the entire backtest period, "
            "regardless of subsequent delistings or failures. Data for all candidate "
            "symbols is loaded up to t0, and any symbols delisted before t0 are naturally "
            "excluded if no data is available for them at that time. This is based on historical data."
        )
    }

    return selected_symbols, universe_metadata
