"""
Risk computation applied to DataFrames.

``compute_risks`` is intentionally a top-level function (not a method) so it
can be passed to ``ProcessPoolExecutor.submit`` without pickling issues.
"""

import logging

import pandas as pd

from api.config.constants import TEMP_INACTIVE_THRESHOLD_C
from api.models.risk_models import (
    calculate_tarspot_risk,
    calculate_gray_leaf_spot_risk,
    calculate_frogeye_leaf_spot_risk,
    calculate_whitemold_irrigated_risk,
    calculate_whitemold_non_irrigated_risk,
)

logger = logging.getLogger(__name__)

# Inactive sentinel Series for each disease, returned when data is insufficient.
_TARSPOT_INACTIVE = pd.Series({"tarspot_risk": -1, "tarspot_risk_class": "Inactive"})
_GLS_INACTIVE     = pd.Series({"gls_risk": -1, "gls_risk_class": "Inactive"})
_FE_INACTIVE      = pd.Series({"fe_risk": -1, "fe_risk_class": "Inactive"})
_WMI_INACTIVE     = pd.Series({
    "whitemold_irr_30in_risk": -1, "whitemold_irr_15in_risk": -1,
    "whitemold_irr_15in_class": "Inactive", "whitemold_irr_30in_class": "Inactive",
})
_WMN_INACTIVE = pd.Series({"whitemold_nirr_risk": -1, "whitemold_nirr_risk_class": "Inactive"})


def _is_inactive(row: pd.Series) -> bool:
    """True when the 30-day average temperature is missing or below threshold."""
    val = row.get("air_temp_avg_c_30d_ma")
    return pd.isna(val) or val < TEMP_INACTIVE_THRESHOLD_C


def compute_risks(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all disease-risk columns for a slice of the station DataFrame.

    Designed to be called in a ``ProcessPoolExecutor`` worker.  Returns a
    copy of *df_chunk* with all risk columns appended.
    """
    df = df_chunk.copy()

    # ── Tarspot ──────────────────────────────────────────────────────────────
    df[["tarspot_risk", "tarspot_risk_class"]] = df.apply(
        lambda r: _TARSPOT_INACTIVE if _is_inactive(r) else calculate_tarspot_risk(
            r["air_temp_avg_c_30d_ma"],
            r["rh_max_30d_ma"],
            r["rh_above_90_night_14d_ma"],
        ),
        axis=1,
    )

    # ── Gray Leaf Spot ────────────────────────────────────────────────────────
    df[["gls_risk", "gls_risk_class"]] = df.apply(
        lambda r: _GLS_INACTIVE if _is_inactive(r) else calculate_gray_leaf_spot_risk(
            r["air_temp_min_c_21d_ma"],
            r["dp_min_30d_c_ma"],
        ),
        axis=1,
    )

    # ── Frogeye Leaf Spot ─────────────────────────────────────────────────────
    df[["fe_risk", "fe_risk_class"]] = df.apply(
        lambda r: _FE_INACTIVE if _is_inactive(r) else calculate_frogeye_leaf_spot_risk(
            r["air_temp_max_c_30d_ma"],
            r["rh_above_80_day_30d_ma"],
        ),
        axis=1,
    )

    # ── White Mold – Irrigated ────────────────────────────────────────────────
    wmi_cols = list(_WMI_INACTIVE.index)
    df[wmi_cols] = df.apply(
        lambda r: _WMI_INACTIVE if _is_inactive(r) else calculate_whitemold_irrigated_risk(
            r["air_temp_max_c_30d_ma"],
            r["rh_max_30d_ma"],
        ),
        axis=1,
    )

    # ── White Mold – Non-Irrigated ────────────────────────────────────────────
    df[["whitemold_nirr_risk", "whitemold_nirr_risk_class"]] = df.apply(
        lambda r: _WMN_INACTIVE if _is_inactive(r) else calculate_whitemold_non_irrigated_risk(
            r["air_temp_max_c_30d_ma"],
            r["rh_max_30d_ma"],
            r["max_ws_30d_ma"],
        ),
        axis=1,
    )

    return df


def chunk_dataframe(df: pd.DataFrame, num_chunks: int) -> list[pd.DataFrame]:
    """
    Split *df* into chunks suitable for parallel processing.

    Small DataFrames (≤ 100 rows) are returned as a single chunk to avoid
    the process-pool overhead outweighing the benefit of parallelism.
    """
    n = len(df)
    if n <= 100:
        return [df]
    size = min(max(n // num_chunks, 100), 1000)
    return [df.iloc[i: i + size] for i in range(0, n, size)]