"""
WiscoNet weather-station data service.

Responsibilities
----------------
- Fetch and cache the station list.
- Fetch hourly measurements for disease-risk rolling averages.
- Fetch daily measurements for biomass (cereal rye) estimation.
- Batch API calls and respect rate limits.

This module does *not* compute risk scores – that is the responsibility of
``services.risk_processor``.
"""

import asyncio
import logging
import os
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from api.config.constants import (
    BATCH_SIZE,
    BIOMASS_CACHE_DIR,
    DESIRED_MEASUREMENT_COLS,
    MEASURE_ID_AIR_TEMP_F_AVG,
    MEASURE_ID_DAILY_AIR_TEMP_F_AVG,
    MEASURE_ID_DAILY_RAIN_IN_TOT,
    MEASURE_ID_DEW_POINT_F_AVG,
    MEASURE_ID_RH_PCT_AVG,
    MEASURE_ID_WIND_SPEED_MPH_MAX,
    MEASUREMENTS_CACHE_DIR,
    MEASUREMENTS_CACHE_TTL_HOURS,
    MIN_DAYS_ACTIVE,
    ROLLING_MAP,
    STATIONS_CACHE_FILE,
    STATIONS_CACHE_TTL_DAYS,
    STATIONS_TO_EXCLUDE,
    WISCONET_BASE_URL,
)
from api.models.risk_models import calculate_cereal_rye_biomass
from api.utils.cache import ensure_cache_dirs, is_fresh_days, is_fresh_hours
from api.utils.conversions import fahrenheit_to_celsius
from api.services.http_client import get_with_retry

logger = logging.getLogger(__name__)

# Ensure cache directories exist at import time.
ensure_cache_dirs(MEASUREMENTS_CACHE_DIR, BIOMASS_CACHE_DIR)


# ── Station list ───────────────────────────────────────────────────────────────

async def load_stations(session, input_date: str) -> pd.DataFrame | None:
    """
    Return the filtered station list, preferring a local CSV cache.

    The cache is considered fresh for ``STATIONS_CACHE_TTL_DAYS`` days.
    If the API call fails, the stale cache is used as a fallback.

    Args:
        session:    Open ``aiohttp.ClientSession``.
        input_date: Reference date (YYYY-MM-DD) used to filter stations by
                    ``earliest_api_date``.

    Returns:
        DataFrame of active stations, or ``None`` if no data is available.
    """
    if is_fresh_days(STATIONS_CACHE_FILE, STATIONS_CACHE_TTL_DAYS):
        logger.info("Loading stations from cache.")
        return pd.read_csv(STATIONS_CACHE_FILE)

    logger.info("Fetching fresh station list from API.")
    url = f"{WISCONET_BASE_URL}/stations/"
    async with session.get(url) as resp:
        if resp.status != 200:
            logger.error("Station API returned HTTP %d", resp.status)
            if os.path.exists(STATIONS_CACHE_FILE):
                logger.warning("Falling back to stale station cache.")
                return pd.read_csv(STATIONS_CACHE_FILE)
            return None

        stations = pd.DataFrame(await resp.json())

    stations = stations[~stations["station_id"].isin(STATIONS_TO_EXCLUDE)]
    stations["earliest_api_date"] = pd.to_datetime(
        stations["earliest_api_date"], format="%m/%d/%Y", errors="coerce"
    )

    threshold = pd.to_datetime(input_date) - pd.Timedelta(days=MIN_DAYS_ACTIVE)
    stations = stations[stations["earliest_api_date"] <= threshold]

    stations.to_csv(STATIONS_CACHE_FILE, index=False)
    logger.info("Cached %d stations.", len(stations))
    return stations


# ── Hourly measurement helpers ─────────────────────────────────────────────────

def _extract_measure(measures: list, measure_id: int) -> float:
    """Return the value for *measure_id* from a raw measures list, or NaN."""
    return next((v for (i, v) in measures if i == measure_id), np.nan)


def _parse_hourly_measures(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unpack the nested ``measures`` column into named sensor columns and
    derive RH threshold flags.
    """
    raw_df["60min_air_temp_f_avg"]          = raw_df["measures"].apply(_extract_measure, args=(MEASURE_ID_AIR_TEMP_F_AVG,))
    raw_df["60min_dew_point_f_avg"]         = raw_df["measures"].apply(_extract_measure, args=(MEASURE_ID_DEW_POINT_F_AVG,))
    raw_df["60min_relative_humidity_pct_avg"] = raw_df["measures"].apply(_extract_measure, args=(MEASURE_ID_RH_PCT_AVG,))
    raw_df["60min_wind_speed_mph_max"]       = raw_df["measures"].apply(_extract_measure, args=(MEASURE_ID_WIND_SPEED_MPH_MAX,))

    raw_df["collection_time"] = (
        pd.to_datetime(raw_df["collection_time"], unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("US/Central")
    )
    raw_df["hour"] = raw_df["collection_time"].dt.hour
    raw_df["date"] = raw_df["collection_time"].dt.strftime("%Y-%m-%d")

    raw_df["rh_night_above_90"] = (
        (raw_df["60min_relative_humidity_pct_avg"] >= 90)
        & ((raw_df["hour"] >= 20) | (raw_df["hour"] <= 6))
    ).astype(int)
    raw_df["rh_day_above_80"] = (raw_df["60min_relative_humidity_pct_avg"] >= 80).astype(int)

    return raw_df


def _aggregate_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly records to daily stats and apply rolling averages."""
    agg = hourly_df.groupby("date").agg(
        nhours_rh_above_90    =("rh_night_above_90",              "sum"),
        hours_rh_above_80_day =("rh_day_above_80",                "sum"),
        rh_max                =("60min_relative_humidity_pct_avg", "max"),
        min_dp                =("60min_dew_point_f_avg",           "min"),
        max_ws                =("60min_wind_speed_mph_max",        "max"),
        air_temp_max_f        =("60min_air_temp_f_avg",            "max"),
        air_temp_min_f        =("60min_air_temp_f_avg",            "min"),
    ).reset_index()

    agg["min_dp_c"]       = fahrenheit_to_celsius(agg["min_dp"])
    agg["air_temp_max_c"] = fahrenheit_to_celsius(agg["air_temp_max_f"])
    agg["air_temp_min_c"] = fahrenheit_to_celsius(agg["air_temp_min_f"])
    agg["air_temp_avg_c"] = (agg["air_temp_max_c"] + agg["air_temp_min_c"]) / 2

    for new_col, (src_col, window) in ROLLING_MAP.items():
        agg[new_col] = agg[src_col].rolling(window=window, min_periods=window).mean()

    return agg


# ── Hourly measurement fetch ───────────────────────────────────────────────────

async def _fetch_station_measurements(
    session, station_id: str, end_time: str
) -> pd.DataFrame | None:
    """
    Fetch raw hourly measurements for *station_id* and return a daily aggregate.
    """
    end_dt   = datetime.strptime(end_time, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=MIN_DAYS_ACTIVE)

    params = {
        "start_time": int(start_dt.timestamp()),
        "end_time":   int(end_dt.timestamp()),
        "fields": (
            "60min_relative_humidity_pct_avg,"
            "60min_air_temp_f_avg,"
            "60min_dew_point_f_avg,"
            "60min_wind_speed_mph_max"
        ),
    }

    try:
        payload = await get_with_retry(session, f"{WISCONET_BASE_URL}/stations/{station_id}/measures", params)
        data = payload.get("data", []) if payload else []
        if not data:
            return None

        df = pd.DataFrame(data)
        if df.empty:
            return None

        df = _parse_hourly_measures(df)
        agg = _aggregate_to_daily(df)
        agg["station_id"] = station_id
        return agg

    except Exception:
        logger.error("Error fetching measurements for station %s", station_id)
        traceback.print_exc()
        return None


async def fetch_station_measurements_cached(
    session, station_id: str, end_time: str, days: int
) -> pd.DataFrame | None:
    """
    Fetch (or restore from cache) the most recent *days* rows of daily data.

    Args:
        session:    Open ``aiohttp.ClientSession``.
        station_id: WiscoNet station identifier.
        end_time:   Upper bound date (YYYY-MM-DD).
        days:       Number of recent days to return.

    Returns:
        DataFrame slice or ``None`` on failure.
    """
    cache_file = os.path.join(MEASUREMENTS_CACHE_DIR, f"{station_id}_{end_time}_{days}.pkl")

    if is_fresh_hours(cache_file, MEASUREMENTS_CACHE_TTL_HOURS):
        return pd.read_pickle(cache_file)

    try:
        df = await _fetch_station_measurements(session, station_id, end_time)
        if df is None or df.empty:
            return None

        df = df.sort_values("date", ascending=False).head(days)

        for col in DESIRED_MEASUREMENT_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[DESIRED_MEASUREMENT_COLS]

        df.to_pickle(cache_file)
        return df

    except Exception:
        traceback.print_exc()
        return None


# ── Batched measurement pipeline ───────────────────────────────────────────────

async def fetch_measurements_in_batches(
    session, station_ids: list[str], input_date: str, days: int
) -> list[pd.DataFrame] | None:
    """
    Fetch measurements for all *station_ids* in ``BATCH_SIZE`` chunks with a
    small sleep between batches to respect API rate limits.

    Returns:
        List of DataFrames (one per station that returned data), or ``None``.
    """
    results: list[pd.DataFrame] = []
    for i in range(0, len(station_ids), BATCH_SIZE):
        batch = station_ids[i: i + BATCH_SIZE]
        tasks = [
            fetch_station_measurements_cached(session, sid, input_date, days)
            for sid in batch
        ]
        batch_out = await asyncio.gather(*tasks)
        results.extend(df for df in batch_out if isinstance(df, pd.DataFrame))
        if i + BATCH_SIZE < len(station_ids):
            await asyncio.sleep(0.5)

    return results or None


# ── Biomass fetch ──────────────────────────────────────────────────────────────

async def fetch_station_biomass(
    session, station_id: str, planting_date: str, termination_date: str
) -> dict | None:
    """
    Fetch daily temperature and rainfall between *planting_date* and
    *termination_date*, compute the cereal-rye biomass model inputs, and
    return the biomass estimate alongside the raw inputs.

    Measure IDs used:
        3  → daily_air_temp_f_avg
        15 → daily_rain_in_tot
    """
    try:
        start_dt = datetime.strptime(planting_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(termination_date, "%Y-%m-%d")

        params = {
            "start_time": int(start_dt.timestamp()),
            "end_time":   int(end_dt.timestamp()),
            "fields":     "daily_air_temp_f_avg,daily_rain_in_tot",
        }

        payload = await get_with_retry(
            session,
            f"{WISCONET_BASE_URL}/stations/{station_id}/measures",
            params,
        )
        data = payload.get("data", []) if payload else []
        if not data:
            return None

        df = pd.DataFrame(data)
        df["temp_avg_f"] = df["measures"].apply(
            lambda m: _extract_measure(m, MEASURE_ID_DAILY_AIR_TEMP_F_AVG)
        )
        df["rain_in"] = df["measures"].apply(
            lambda m: _extract_measure(m, MEASURE_ID_DAILY_RAIN_IN_TOT)
        )
        df["collection_time"] = pd.to_datetime(df["collection_time"], unit="s")
        df = df.sort_values("collection_time")

        # First 60 days after planting
        df_60d      = df[df["collection_time"] <= start_dt + timedelta(days=60)]
        cgdd_60d_ap = float((df_60d["temp_avg_f"] - 32).clip(lower=0).sum())
        rain_60d_ap = float(df_60d["rain_in"].sum())

        # 30 days before termination
        df_bt       = df[df["collection_time"] >= end_dt - timedelta(days=30)]
        cgdd_60d_bt = float((df_bt["temp_avg_f"] - 32).clip(lower=0).sum())

        report = calculate_cereal_rye_biomass(cgdd_60d_ap, rain_60d_ap, cgdd_60d_bt)

        return {
            "station_id":      station_id,
            "cgdd_60d_ap":     cgdd_60d_ap,
            "rain_60d_ap":     rain_60d_ap,
            "cgdd_60d_bt":     cgdd_60d_bt,
            "biomass_lb_acre": report["biomass"],
            "biomass_color":   report["color"],
            "biomass_message": report["message"],
        }

    except Exception as exc:
        logger.error("Biomass fetch failed for station %s: %s", station_id, exc)
        return None


async def fetch_biomass_in_batches(
    session, station_ids: list[str], planting_date: str, termination_date: str
) -> pd.DataFrame | None:
    """
    Fetch biomass data for all stations in ``BATCH_SIZE`` chunks.

    Returns:
        DataFrame (one row per station) or ``None`` if no data was returned.
    """
    results: list[dict] = []
    for i in range(0, len(station_ids), BATCH_SIZE):
        batch = station_ids[i: i + BATCH_SIZE]
        tasks = [
            fetch_station_biomass(session, sid, planting_date, termination_date)
            for sid in batch
        ]
        batch_out = await asyncio.gather(*tasks)
        results.extend(r for r in batch_out if r is not None)
        if i + BATCH_SIZE < len(station_ids):
            await asyncio.sleep(0.5)

    return pd.DataFrame(results) if results else None