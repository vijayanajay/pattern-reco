"""
Tests for the walk-forward validation framework.
"""
from datetime import date

import pytest

from src.config import WalkForwardConfig
from src.walk_forward import WalkForwardSplit, create_walk_forward_splits


@pytest.fixture
def wf_config() -> WalkForwardConfig:
    """Provides a default WalkForwardConfig for tests."""
    return WalkForwardConfig(is_years=3, oos_years=1, holdout_years=2)


def test_create_walk_forward_splits_standard(wf_config: WalkForwardConfig):
    """
    Tests a standard case with a 10-year period, 3-year IS, 1-year OOS,
    and 2-year holdout. Expects 4 splits.
    """
    start_date = date(2010, 1, 1)
    end_date = date(2020, 1, 1)
    splits = create_walk_forward_splits(start_date, end_date, wf_config)

    assert len(splits) == 5
    assert splits[0] == WalkForwardSplit(
        is_start_date=date(2010, 1, 1),
        is_end_date=date(2012, 12, 31),
        oos_start_date=date(2013, 1, 1),
        oos_end_date=date(2013, 12, 31),
    )
    assert splits[-1] == WalkForwardSplit(
        is_start_date=date(2014, 1, 1),
        is_end_date=date(2016, 12, 31),
        oos_start_date=date(2017, 1, 1),
        oos_end_date=date(2017, 12, 31),
    )


def test_create_walk_forward_splits_no_splits_too_short(wf_config: WalkForwardConfig):
    """Tests that no splits are generated if the total period is too short."""
    start_date = date(2010, 1, 1)
    end_date = date(2015, 1, 1)  # 5 years, need 3+1+2=6 years
    splits = create_walk_forward_splits(start_date, end_date, wf_config)
    assert len(splits) == 0


def test_create_walk_forward_splits_one_split(wf_config: WalkForwardConfig):
    """Tests a period that is just long enough for one split."""
    start_date = date(2010, 1, 1)
    end_date = date(2016, 1, 1)  # 6 years, allows for one 3+1 split before 2y holdout
    splits = create_walk_forward_splits(start_date, end_date, wf_config)
    assert len(splits) == 1
    assert splits[0] == WalkForwardSplit(
        is_start_date=date(2010, 1, 1),
        is_end_date=date(2012, 12, 31),
        oos_start_date=date(2013, 1, 1),
        oos_end_date=date(2013, 12, 31),
    )


def test_create_walk_forward_splits_custom_params():
    """Tests a different set of IS/OOS/holdout parameters."""
    wf_config = WalkForwardConfig(is_years=2, oos_years=1, holdout_years=1)
    start_date = date(2015, 1, 1)
    end_date = date(2020, 1, 1)
    splits = create_walk_forward_splits(start_date, end_date, wf_config)

    assert len(splits) == 2
    assert splits[0] == WalkForwardSplit(
        is_start_date=date(2015, 1, 1),
        is_end_date=date(2016, 12, 31),
        oos_start_date=date(2017, 1, 1),
        oos_end_date=date(2017, 12, 31),
    )
    assert splits[1] == WalkForwardSplit(
        is_start_date=date(2016, 1, 1),
        is_end_date=date(2017, 12, 31),
        oos_start_date=date(2018, 1, 1),
        oos_end_date=date(2018, 12, 31),
    )
