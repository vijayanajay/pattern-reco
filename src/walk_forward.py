"""
Walk-forward validation framework.
"""
from dataclasses import dataclass
from datetime import date
from typing import List

from dateutil.relativedelta import relativedelta

from src.config import WalkForwardConfig

__all__ = ["create_walk_forward_splits", "WalkForwardSplit"]


@dataclass(frozen=True)
class WalkForwardSplit:
    """
    Represents a single walk-forward split with in-sample and out-of-sample periods.
    """

    is_start_date: date
    is_end_date: date
    oos_start_date: date
    oos_end_date: date


def create_walk_forward_splits(
    start_date: date, end_date: date, wf_config: WalkForwardConfig
) -> List[WalkForwardSplit]:
    """
    Creates a list of walk-forward splits based on the provided configuration.
    """
    splits = []
    holdout_months = wf_config.holdout_years * 12
    is_months = wf_config.is_years * 12
    oos_months = wf_config.oos_years * 12

    train_end_date = end_date - relativedelta(months=holdout_months)
    current_date = start_date

    while current_date + relativedelta(months=is_months + oos_months) <= train_end_date:
        is_start = current_date
        is_end = current_date + relativedelta(months=is_months) - relativedelta(days=1)
        oos_start = current_date + relativedelta(months=is_months)
        oos_end = oos_start + relativedelta(months=oos_months) - relativedelta(days=1)

        splits.append(WalkForwardSplit(is_start, is_end, oos_start, oos_end))
        current_date += relativedelta(months=oos_months)

    return splits
