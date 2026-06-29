"""Numeric ordering of detected account numbers.

Pure logic. Given 291039, 292039, 290134 → ascending → 290134, 291039, 292039,
with ranks 1, 2, 3 assigned in place.
"""
from __future__ import annotations

from typing import List

from .config import Settings
from .types import AccountNumber


def rank_accounts(accounts: List[AccountNumber], settings: Settings) -> List[AccountNumber]:
    """Sort by numeric value and assign 1-based ranks. Mutates and returns the list."""
    ordered = sorted(accounts, key=lambda a: a.value, reverse=not settings.ascending)
    for index, account in enumerate(ordered, start=1):
        account.rank = index
    return ordered
