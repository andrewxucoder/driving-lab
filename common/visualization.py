"""Visualization helpers for exploratory notebooks."""

from typing import Sequence
import matplotlib.pyplot as plt


def plot_series(series: Sequence[float], title: str = "Series", xlabel: str = "Index", ylabel: str = "Value") -> None:
    """Plot a simple 1D series."""
    plt.figure(figsize=(8, 4))
    plt.plot(series)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
