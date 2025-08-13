"""
Stateless detector functions for identifying trading signals.
"""
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import backtest, features

__all__ = ["fit_gap_z_detector"]


def fit_gap_z_detector(
    data: pd.DataFrame,
    params_grid: Dict[str, List[Any]],
    min_hit_rate: float = 0.4,
) -> Optional[Dict[str, Any]]:
    """
    Fits the Gap-Z detector by finding the best parameters from a grid search.
    """
    best_score = -np.inf
    best_params: Optional[Dict[str, Any]] = None

    param_names = list(params_grid.keys())
    param_combinations = list(product(*params_grid.values()))

    for params_tuple in param_combinations:
        params = dict(zip(param_names, params_tuple))

        window = params["window"]
        k_low = params["k_low"]

        df_featured = features.add_gap_z_signal(data, window=window, k_low=k_low)
        signals = df_featured["signal_gapz"]

        # Construct a minimal config for the backtester
        backtest_config = {
            "portfolio": {"max_hold_days": params["max_hold"]},
            "execution": {"fees_bps": 0, "slippage_model": {}} # No costs for fitting
        }

        trade_ledger = backtest.run_backtest(df_featured, signals, backtest_config)

        if trade_ledger.empty:
            continue

        returns = trade_ledger["return_pct"]
        hit_rate = (returns > 0).mean()

        if hit_rate < min_hit_rate:
            continue

        score = returns.median()
        if score > best_score:
            best_score = score
            best_params = params

    return best_params
