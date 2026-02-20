"""
IBM Environmental Intelligence Suite (EIS) weather-data service.

Responsibilities
----------------
- Authenticate with IBM SaaSCore to obtain a JWT.
- Fetch hourly-on-demand (HOD) weather data in time-chunked requests.
- Build hourly and daily DataFrames with the rolling averages the risk
  models need.
- Compute all disease-risk scores and return both granularities.
"""

import logging
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd
import requests

from api.config.constants import (
    IBM_CHUNK_HOURS,
    IBM_GEOSPATIAL_URL,
    IBM_ROLLING_MAP,
    IBM_SAASCORE_URL,
    TEMP_INACTIVE_THRESHOLD_C,
)
from api.models.risk_models import (
    calculate_frogeye_leaf_spot_risk,
    calculate_gray_leaf_spot_risk,
    calculate_tarspot_risk,
    calculate_whitemold_irrigated_risk,
    calculate_whitemold_non_irrigated_risk,
)
from api.utils.conversions import kmh_to_mps
from api.utils.math_helpers import rolling_mean

logger = logging.getLogger(__name__)


# ── Authentication ─────────────────────────────────────────────────────────────

def fetch_jwt(org_id: str, tenant_id: str, api_key: str) -> str | None:
    """
    Obtain a short-lived JWT from IBM SaaSCore.

    Args:
        org_id:    EIS organisation ID.
        tenant_id: EIS tenant ID.
        api_key:   Current EIS API key.

    Returns:
        JWT string, or ``None`` if the request fails.
    """
    url = f"{IBM_SAASCORE_URL}/api-key?orgId={org_id}"
    headers = {
        "x-ibm-client-Id": f"saascore-{tenant_id}",
        "x-api-key": api_key,
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        logger.error("IBM SaaSCore auth failed: %s", exc)
        return None


# ── Time chunking ──────────────────────────────────────────────────────────────

def _generate_time_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """
    Split a date range into chunks of at most ``IBM_CHUNK_HOURS`` hours.

    IBM's HOD endpoint caps the window per request, so longer date ranges
    must be split.

    Args:
        start_date: Inclusive start (YYYY-MM-DD).
        end_date:   Exclusive end (YYYY-MM-DD).

    Returns:
        List of (start, end) tuples formatted as ``"YYYY-MM-DDTHH"``.
    """
    start = datetime.combine(datetime.fromisoformat(start_date).date(), time(0, 1))
    end   = datetime.combine(datetime.fromisoformat(end_date).date(),   time(0, 1))

    chunks: list[tuple[str, str]] = []
    while start < end:
        chunk_end = min(start + timedelta(hours=IBM_CHUNK_HOURS), end)
        chunks.append((start.strftime("%Y-%m-%dT%H"), chunk_end.strftime("%Y-%m-%dT%H")))
        start = start + timedelta(hours=IBM_CHUNK_HOURS)

    return chunks


# ── Raw data fetch ─────────────────────────────────────────────────────────────

def _fetch_raw_hourly(
    lat: float,
    lng: float,
    start_date: str,
    end_date: str,
    tenant_id: str,
    jwt: str,
) -> pd.DataFrame | None:
    """
    Fetch all hourly records between *start_date* and *end_date* from IBM's
    HOD endpoint, stitching chunks together into a single DataFrame.
    """
    url = IBM_GEOSPATIAL_URL
    headers = {
        "x-ibm-client-id":  f"geospatial-{tenant_id}",
        "Authorization":    f"Bearer {jwt}",
    }

    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _generate_time_chunks(start_date, end_date):
        params = {
            "format":        "json",
            "geocode":       f"{lat},{lng}",
            "startDateTime": chunk_start,
            "endDateTime":   chunk_end,
            "units":         "m",
        }
        try:
            resp = requests.get(url, headers=headers, params=params)
            if resp.status_code == 200:
                frames.append(pd.DataFrame(resp.json()))
            else:
                logger.warning("IBM HOD returned HTTP %d for chunk %s–%s", resp.status_code, chunk_start, chunk_end)
        except requests.RequestException as exc:
            logger.error("IBM HOD request error: %s", exc)

    return pd.concat(frames, ignore_index=True) if frames else None


# ── DataFrame builders ─────────────────────────────────────────────────────────

def _build_hourly(raw: pd.DataFrame, tz: str = "US/Central") -> pd.DataFrame:
    """Add timezone-aware datetime columns to the raw hourly IBM DataFrame."""
    raw = raw.copy()
    raw["dttm_utc"] = pd.to_datetime(raw["validTimeUtc"], utc=True)
    raw["dttm"]     = raw["dttm_utc"].dt.tz_convert(tz)
    raw["date"]     = raw["dttm"].dt.date
    raw["hour"]     = raw["dttm"].dt.hour
    raw["night"]    = ~raw["hour"].between(7, 19)
    raw["date_since_night"] = (raw["dttm"] + pd.to_timedelta(4, unit="h")).dt.date
    return raw.sort_values("dttm_utc")


def _build_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly IBM data to daily stats and compute rolling averages."""
    daily = hourly.groupby("date").agg(
        temperature_min           =("temperature",          "min"),
        temperature_mean          =("temperature",          "mean"),
        temperature_max           =("temperature",          "max"),
        temperatureDewPoint_min   =("temperatureDewPoint",  "min"),
        temperatureDewPoint_mean  =("temperatureDewPoint",  "mean"),
        temperatureDewPoint_max   =("temperatureDewPoint",  "max"),
        relativeHumidity_min      =("relativeHumidity",     "min"),
        relativeHumidity_mean     =("relativeHumidity",     "mean"),
        relativeHumidity_max      =("relativeHumidity",     "max"),
        precip1Hour_sum           =("precip1Hour",          "sum"),
        windSpeed_mean            =("windSpeed",            "mean"),
        windSpeed_max             =("windSpeed",            "max"),
    ).reset_index()

    daily["windSpeed_mean"] = kmh_to_mps(daily["windSpeed_mean"])
    daily["windSpeed_max"]  = kmh_to_mps(daily["windSpeed_max"])

    night_rh = (
        hourly[hourly["night"]]
        .groupby("date_since_night")
        .agg(hours_rh90_night=("relativeHumidity", lambda x: (x >= 90).sum()))
        .reset_index()
    )
    allday_rh = (
        hourly.groupby("date")
        .agg(hours_rh80_allday=("relativeHumidity", lambda x: (x >= 80).sum()))
        .reset_index()
    )

    daily = pd.merge(daily, night_rh, left_on="date", right_on="date_since_night", how="left")
    daily = pd.merge(daily, allday_rh, on="date", how="left")

    for col, window in IBM_ROLLING_MAP.items():
        daily[f"{col}_{window}ma"] = rolling_mean(daily[col], window)

    daily["forecasting_date"] = daily["date"].apply(lambda d: d + timedelta(days=1))
    return daily


# ── Risk annotation ────────────────────────────────────────────────────────────

def _annotate_risks(daily: pd.DataFrame) -> pd.DataFrame:
    """Append disease-risk columns to the IBM daily DataFrame."""
    daily = daily.copy()

    def _inactive(row) -> bool:
        val = row.get("temperature_mean_30ma")
        return pd.isna(val) or val < TEMP_INACTIVE_THRESHOLD_C

    # Tarspot
    daily = daily.join(daily.apply(
        lambda r: pd.Series({"tarspot_risk": -1, "tarspot_risk_class": "Inactive"})
        if _inactive(r)
        else calculate_tarspot_risk(r["temperature_mean_30ma"], r["relativeHumidity_max_30ma"], r["hours_rh90_night_14ma"]),
        axis=1,
    ))

    # Gray Leaf Spot
    daily = daily.join(daily.apply(
        lambda r: calculate_gray_leaf_spot_risk(r["temperature_min_21ma"], r["temperatureDewPoint_min_30ma"]),
        axis=1,
    ))

    # Frogeye Leaf Spot
    daily = daily.join(daily.apply(
        lambda r: calculate_frogeye_leaf_spot_risk(r["temperature_max_30ma"], r["hours_rh80_allday_30ma"]),
        axis=1,
    ))

    # White Mold – Irrigated
    daily = daily.join(daily.apply(
        lambda r: calculate_whitemold_irrigated_risk(r["temperature_max_30ma"], r["relativeHumidity_max_30ma"]),
        axis=1,
    ))

    # White Mold – Non-Irrigated
    daily = daily.join(daily.apply(
        lambda r: calculate_whitemold_non_irrigated_risk(
            r["temperature_max_30ma"], r["relativeHumidity_max_30ma"], r["windSpeed_max_30ma"]
        ),
        axis=1,
    ))

    return daily


# ── Public API ─────────────────────────────────────────────────────────────────

def get_weather_with_risk(
    lat: float,
    lng: float,
    end_date: str,
    api_key: str,
    tenant_id: str,
    org_id: str,
    tz: str = "US/Central",
    lookback_days: int = 36,
) -> dict | None:
    """
    Fetch IBM weather data and return both hourly and daily DataFrames,
    with disease-risk scores appended to the daily table.

    Args:
        lat, lng:       Location coordinates.
        end_date:       Upper bound date (YYYY-MM-DD).
        api_key:        IBM EIS API key.
        tenant_id:      IBM EIS tenant ID.
        org_id:         IBM EIS organisation ID.
        tz:             Timezone for local-time conversion.
        lookback_days:  Number of days before *end_date* to fetch.

    Returns:
        ``{"hourly": pd.DataFrame, "daily": pd.DataFrame}`` on success,
        or ``{"hourly": None, "daily": None}`` on failure.
    """
    _failure = {"hourly": None, "daily": None}

    try:
        jwt = fetch_jwt(org_id, tenant_id, api_key)
        if not jwt:
            return _failure

        end_dt    = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        raw = _fetch_raw_hourly(lat, lng, start_date, end_date, tenant_id, jwt)
        if raw is None or raw.empty:
            logger.warning("No IBM data returned for (%s, %s).", lat, lng)
            return _failure

        hourly = _build_hourly(raw, tz)
        daily  = _build_daily(hourly)
        daily  = _annotate_risks(daily)

        return {"hourly": hourly, "daily": daily}

    except Exception as exc:
        logger.error("IBM risk computation failed: %s", exc)
        return _failure