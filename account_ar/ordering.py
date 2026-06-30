"""Numeric ordering of detected account numbers.

Pure logic. Given 291039, 292039, 290134 → ascending → 290134, 291039, 292039,
with ranks 1, 2, 3 assigned in place.
"""
from __future__ import annotations

from typing import List

from .config import Settings
from .types import AccountNumber


def rank_accounts(accounts: List[AccountNumber], settings: Settings) -> List[AccountNumber]:
    """Assign 1-based ranks by numeric value, sharing a rank across equal values.

    Ranking by *unique value* means two papers showing the same account number get
    the same rank (and so the same overlay colour) — letting the user group files into
    stacks by colour. Returns the accounts sorted by rank. Mutates `rank` in place.
    """
    unique_values = sorted({a.value for a in accounts}, reverse=not settings.ascending)
    rank_of = {value: index for index, value in enumerate(unique_values, start=1)}
    for account in accounts:
        account.rank = rank_of[account.value]
    return sorted(accounts, key=lambda a: (a.rank, a.detection.center))
