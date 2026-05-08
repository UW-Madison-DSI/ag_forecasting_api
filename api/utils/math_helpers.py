"""
Mathematical helpers used by the risk models.

Kept separate from conversions so each file has a single responsibility.
"""

import numpy as np
import pandas as pd


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Compute a trailing rolling mean that requires *window* complete periods.

    Returns NaN for any position where fewer than *window* observations
    are available, which lets downstream code distinguish 'no data yet'
    from a genuine zero or near-zero value.
    """
    return series.rolling(window=window, min_periods=window).mean()


def logistic(logit: float | np.ndarray) -> float | np.ndarray:
    """Standard logistic (sigmoid) function: σ(x) = e^x / (1 + e^x)."""
    return np.exp(logit) / (1 + np.exp(logit))


def compute_logit(intercept: float, terms: list[tuple[float, float]]) -> float:
    """
    Compute a linear-predictor (logit) value.

    Args:
        intercept: The model constant / bias term.
        terms:     List of (coefficient, variable) pairs whose products
                   are *subtracted* from the intercept.

    Returns:
        intercept - Σ(coef_i × var_i)
    """
    return intercept - sum(coef * var for coef, var in terms)


def gdd_sine(
    tmin: float,
    tmax: float,
    base_temp: float = 0,
    max_temp: float = 35,
) -> float:
    """Calculate Growing Degree Days using a sine-wave approximation."""
    import math

    tmin = min(max(tmin, base_temp), max_temp)
    tmax = min(max(tmax, base_temp), max_temp)

    if tmax <= tmin:
        return 0.0

    tavg = (tmin + tmax) / 2.0
    if tavg <= base_temp:
        return 0.0

    amplitude = (tmax - tmin) / 2.0
    gdd = (tavg - base_temp) - (amplitude / math.pi) * math.cos(math.pi * (tmax - tmin) / (2.0 * amplitude))
    return max(0.0, gdd)