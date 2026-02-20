"""
Unit-conversion utilities.

All functions are pure (no side-effects) and operate on scalars or
pandas Series/arrays so they can be used in both row-wise lambdas
and vectorised column operations.
"""

import numpy as np
import pandas as pd


def fahrenheit_to_celsius(value: float | pd.Series | np.ndarray) -> float | pd.Series | np.ndarray:
    """Convert Fahrenheit to Celsius."""
    return (value - 32) * 5 / 9


def kmh_to_mps(value: float | pd.Series | np.ndarray) -> float | pd.Series | np.ndarray:
    """Convert km/h to m/s."""
    return value / 3.6