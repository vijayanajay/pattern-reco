"""
Detector implementation for identifying trading signals.
This module contains the GapZDetector, which fits parameters to find anomalies.
"""
import json
from itertools import product
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.features import FeatureEngine
from src.tradesim import calculate_returns

__all__ = ["GapZDetector"]


class GapZDetector:
    """
    A detector for the Gap-Z anomaly, which identifies significant downward gaps.

    This detector finds the optimal parameters for a trading strategy based on
    the z-score of the overnight gap. It uses a grid search to find the
    combination of `window`, `k_low`, and `max_hold` that maximizes the median
    return during a fitting period.
    """

    def __init__(self, params_grid: Dict[str, List[Any]]):
        """
        Initializes the detector with a grid of parameters to search.

        Args:
            params_grid: A dictionary where keys are parameter names ('window',
                         'k_low', 'max_hold') and values are lists of values to test.
        """
        self.params_grid = params_grid
        self.best_params_: Dict[str, Any] = {}

    def fit(self, data: pd.DataFrame) -> "GapZDetector":
        """
        Fits the detector to the data by searching for the best parameters.

        Iterates through all combinations of parameters in the provided grid,
        calculates the median return for each, and selects the set of
        parameters that yields the highest median return.

        Args:
            data: In-sample OHLC data as a pandas DataFrame.

        Returns:
            The fitted detector instance.
        """
        best_score = -np.inf

        param_names = self.params_grid.keys()
        param_combinations = list(product(*self.params_grid.values()))

        for params_tuple in param_combinations:
            params = dict(zip(param_names, params_tuple))
            window, k_low, max_hold = (
                params["window"],
                params["k_low"],
                params["max_hold"],
            )

            # Generate signals using the centralized feature engine method
            df_featured = FeatureEngine.add_gap_z_signal(
                data, window=window, k_low=k_low
            )
            signals = df_featured["signal_gapz"]

            # Evaluate the signals
            returns = calculate_returns(df_featured, signals, max_hold=max_hold)

            if not returns.empty:
                score = returns.median()
                if score > best_score:
                    best_score = score
                    self.best_params_ = params
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates signals for the given data using the best-fitted parameters.

        Args:
            data: Out-of-sample OHLC data as a pandas DataFrame.

        Returns:
            A boolean Series of trading signals.
        """
        if not self.best_params_:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")

        window = self.best_params_["window"]
        k_low = self.best_params_["k_low"]

        df_featured = FeatureEngine.add_gap_z_signal(
            data, window=window, k_low=k_low
        )
        return df_featured["signal_gapz"]

    def save(self, filepath: str) -> None:
        """Saves the fitted parameters to a JSON file."""
        if not self.best_params_:
            raise RuntimeError("Nothing to save. Detector has not been fitted.")
        with open(filepath, "w") as f:
            json.dump(self.best_params_, f, indent=4)

    def load(self, filepath: str) -> "GapZDetector":
        """Loads fitted parameters from a JSON file."""
        with open(filepath, "r") as f:
            self.best_params_ = json.load(f)
        return self
