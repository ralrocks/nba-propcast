"""Date-aware expanding-window cross-validation.

Key invariant: every row in a test fold has a game_date strictly later than
every row in the corresponding training fold.  This mirrors real deployment —
predictions always go forward in time, never backward.

Splitting on unique dates (not row indices) means that all players who played
on the same calendar date always land in the same fold, which prevents
information from a given game day leaking from test into train.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def expanding_date_splits(
    dates: pd.Series,
    n_splits: int = 5,
    min_train_frac: float = 0.25,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_indices, test_indices) for expanding-window time-series CV.

    Args:
        dates:           Series of game dates (any type convertible by pd.to_datetime).
        n_splits:        Number of folds.  Silently reduced if there aren't enough
                         unique dates to fill all folds.
        min_train_frac:  Fraction of the date range used for the *first* training
                         window (e.g. 0.25 → first 25 % of the season as train).

    Yields:
        (train_idx, test_idx) — integer-position arrays for .iloc[].
    """
    dates_dt = pd.to_datetime(dates)
    unique_dates = np.sort(dates_dt.unique())
    n_unique = len(unique_dates)

    min_train_n = max(1, int(n_unique * min_train_frac))
    remaining = n_unique - min_train_n
    actual_splits = max(1, min(n_splits, remaining))
    test_size = max(1, remaining // actual_splits)

    all_idx = np.arange(len(dates))
    dates_arr = dates_dt.values

    for fold in range(actual_splits):
        train_end = min_train_n + fold * test_size
        test_end = min(train_end + test_size, n_unique)

        if train_end >= n_unique:
            break

        train_cutoff = unique_dates[train_end - 1]
        test_cutoff = unique_dates[test_end - 1]

        train_mask = dates_arr <= train_cutoff
        test_mask = (dates_arr > train_cutoff) & (dates_arr <= test_cutoff)

        yield all_idx[train_mask], all_idx[test_mask]
