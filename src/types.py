"""
Shared data structures for the application.
"""
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["Trade"]


class Trade(BaseModel):
    """
    Represents a single trade with entry and exit points.
    """

    symbol: str = Field(..., description="The stock symbol.")
    entry_date: date = Field(..., description="The date of trade entry.")
    exit_date: date = Field(..., description="The date of trade exit.")
    entry_price: float = Field(..., gt=0, description="The price at which the trade was entered.")
    exit_price: float = Field(..., gt=0, description="The price at which the trade was exited.")
    sample_type: Literal["IS", "OOS"] = Field(..., description="In-sample (IS) or Out-of-sample (OOS).")

    class Config:
        frozen = True  # Make trades immutable
