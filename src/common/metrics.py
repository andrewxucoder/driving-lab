"""Evaluation metrics shared across projects."""

from typing import Iterable
import numpy as np


def mean_average_precision(scores: Iterable[float]) -> float:
    """Compute a simple mean average precision placeholder."""
    scores_array = np.asarray(list(scores), dtype=float)
    if scores_array.size == 0:
        return 0.0
    return float(scores_array.mean())
