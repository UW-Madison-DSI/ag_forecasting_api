from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import os
import traceback
from datetime import datetime, timedelta, date
from concurrent.futures import ProcessPoolExecutor, as_completed

from ag_models_wrappers.forecasting_models import (
    calculate_tarspot_risk_function,
    calculate_gray_leaf_spot_risk_function,
    calculate_frogeye_leaf_spot_function,
    calculate_irrigated_risk,
    calculate_non_irrigated_risk,
    fahrenheit_to_celsius
)

import logging
logging.basicConfig(level=logging.INFO)

today = date.today()

# ─── Constants ────────────────────────────────────────────────────────────────
BATCH_SIZE = 20
BASE_URL = "https://wisconet.wisc.edu/api/v1"
MIN_DAYS_ACTIVE = 38

MEASUREMENTS_CACHE_DIR = "station_measurements_cache"
os.makedirs(MEASUREMENTS_CACHE_DIR, exist_ok=True)

STATIONS_CACHE_FILE = "wisconsin_stations_cache.csv"
STATIONS_TO_EXCLUDE = ['MITEST1', 'WNTEST1']

ROLLING_MAP = {
    'rh_above_90_night_14d_ma': ('nhours_rh_above_90', 14),
    'rh_above_80_day_30d_ma':   ('hours_rh_above_80_day', 30),
    'air_temp_min_c_21d_ma':    ('air_temp_min_c', 21),
    'air_temp_max_c_30d_ma':    ('air_temp_max_c', 30),
    'air_temp_avg_c_30d_ma':    ('air_temp_avg_c', 30),
    'rh_max_30d_ma':            ('rh_max', 30),
    'max_ws_30d_ma':            ('max_ws', 30),
    'dp_min_30d_c_ma':          ('min_dp_c', 30),
}

DESIRED_COLS = [
    'station_id', 'date',
    'rh_above_90_night_14d_ma', 'rh_above_80_day_30d_ma',
    'air_temp_min_c_21d_ma', 'air_temp_max_c_30d_ma', 'air_temp_avg_c_30d_ma',
    'rh_max_30d_ma', 'max_ws_30d_ma', 'dp_min_30d_c_ma'
]

FINAL_COLUMNS = [
    'station_id', 'date', 'forecasting_date', 'station_name', 'city', 'county',
    'latitude', 'longitude', 'region', 'state', 'station_timezone',
    'tarspot_risk', 'tarspot_risk_class', 'gls_risk', 'gls_risk_class',
    'fe_risk', 'fe_risk_class', 'whitemold_nirr_risk', 'whitemold_nirr_risk_class',
    'whitemold_irr_30in_risk', 'whitemold_irr_15in_risk',
    'whitemold_irr_15in_class', 'whitemold_irr_30in_class'
]

# ─── Helper Functions ─────────────────────────────────────────────────────────

async def api_call_with_retry(session, url, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 429:
                    await asyncio.sleep(2 ** attempt)
                else:
                    await asyncio.sleep(1)
        except Exception:
            await asyncio.sleep(1)
    return None


def get_async_session():
    """
    BUG FIX: Was declared as `async def`, which meant callers got a coroutine
    instead of a session object. It's a plain factory — no async needed here.
    The actual session is created and managed with `async with` inside the
    async functions that need it.
    """
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=50, ttl_dns_cache=300),
        timeout=aiohttp.ClientTimeout(total=60)
    )


async def api_call_wisconet_data_async(session, station_id, end_time):
    """
    BUG FIX: Was calling /latest_measures which doesn't support a date range.
    Switched to /measures (the correct bulk endpoint) so historical dates work.
    """
    try:
        end_dt   = datetime.strptime(end_time, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=MIN_DAYS_ACTIVE)
        params = {
            "start_time": int(start_dt.timestamp()),
            "end_time":   int(end_dt.timestamp()),
            "fields": "60min_relative_humidity_pct_avg,60min_air_temp_f_avg,"
                      "60min_dew_point_f_avg,60min_wind_speed_mph_max"
        }

        payload = await api_call_with_retry(
            session,
            f"{BASE_URL}/stations/{station_id}/measures",  # fixed endpoint
            params
        )
        data = payload.get("data", []) if payload else []
        if not data:
            return None

        df = pd.DataFrame(data)
        if df.empty:
            return None

        df['collection_time'] = (
            pd.to_datetime(df['collection_time'], unit='s')
              .dt.tz_localize('UTC')
              .dt.tz_convert('US/Central')
        )

        # Extract individual measure values from the nested measures list
        for mid, col in {
            2:  "60min_air_temp_f_avg",
            10: "60min_dew_point_f_avg",
            19: "60min_relative_humidity_pct_avg",
            57: "60min_wind_speed_mph_max"
        }.items():
            df[col] = df['measures'].apply(
                lambda m, _mid=mid: next((v for (i, v) in m if i == _mid), np.nan)
            )

        df['hour'] = df['collection_time'].dt.hour
        df['date'] = df['collection_time'].dt.strftime('%Y-%m-%d')

        df['rh_night_above_90'] = (
            ((df['60min_relative_humidity_pct_avg'] >= 90) &
             ((df['hour'] >= 20) | (df['hour'] <= 6)))
            .astype(int)
        )
        df['rh_day_above_80'] = (df['60min_relative_humidity_pct_avg'] >= 80).astype(int)

        agg = df.groupby('date').agg(
            nhours_rh_above_90    = ('rh_night_above_90', 'sum'),
            hours_rh_above_80_day = ('rh_day_above_80', 'sum'),
            rh_max                = ('60min_relative_humidity_pct_avg', 'max'),
            min_dp                = ('60min_dew_point_f_avg', 'min'),
            max_ws                = ('60min_wind_speed_mph_max', 'max'),
            air_temp_max_f        = ('60min_air_temp_f_avg', 'max'),
            air_temp_min_f        = ('60min_air_temp_f_avg', 'min'),
        ).reset_index()

        agg['min_dp_c']       = fahrenheit_to_celsius(agg['min_dp'])
        agg['air_temp_max_c'] = fahrenheit_to_celsius(agg['air_temp_max_f'])
        agg['air_temp_min_c'] = fahrenheit_to_celsius(agg['air_temp_min_f'])
        agg['air_temp_avg_c'] = (agg['air_temp_max_c'] + agg['air_temp_min_c']) / 2

        for newcol, (src, window) in ROLLING_MAP.items():
            agg[newcol] = agg[src].rolling(window=window, min_periods=window).mean()

        agg['station_id'] = station_id
        return agg

    except Exception:
        traceback.print_exc()
        return None


async def one_day_measurements_async(session, station_id, end_time, days):
    try:
        cache_file = f"{MEASUREMENTS_CACHE_DIR}/{station_id}_{end_time}_{days}.pkl"
        if os.path.exists(cache_file):
            age_hr = (
                datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            ).total_seconds() / 3600
            if age_hr < 6:
                return pd.read_pickle(cache_file)

        df = await api_call_wisconet_data_async(session, station_id, end_time)
        if df is None or df.empty:
            return None

        df = df.sort_values('date', ascending=False).head(days)
        for col in DESIRED_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[DESIRED_COLS]

        df.to_pickle(cache_file)
        return df

    except Exception:
        traceback.print_exc()
        return None


async def process_stations_in_batches(session, station_ids, input_date, days):
    results = []
    for i in range(0, len(station_ids), BATCH_SIZE):
        batch = station_ids[i:i + BATCH_SIZE]
        tasks = [
            one_day_measurements_async(session, sid, input_date, days)
            for sid in batch
        ]
        batch_out = await asyncio.gather(*tasks)
        results.extend([df for df in batch_out if isinstance(df, pd.DataFrame)])
        if i + BATCH_SIZE < len(station_ids):
            await asyncio.sleep(0.5)
    return results or None


def compute_risks(df_chunk):
    df = df_chunk.copy()

    # Tarspot
    tar = df.apply(
        lambda r: pd.Series({'tarspot_risk': -1, 'tarspot_risk_class': 'Inactive'})
        if pd.isna(r['air_temp_avg_c_30d_ma']) or r['air_temp_avg_c_30d_ma'] < 15
        else calculate_tarspot_risk_function(
            r['air_temp_avg_c_30d_ma'],
            r['rh_max_30d_ma'],
            r['rh_above_90_night_14d_ma']
        ),
        axis=1
    )
    df[['tarspot_risk', 'tarspot_risk_class']] = tar

    # Gray Leaf Spot
    gls = df.apply(
        lambda r: pd.Series({'gls_risk': -1, 'gls_risk_class': 'Inactive'})
        if pd.isna(r['air_temp_avg_c_30d_ma']) or r['air_temp_avg_c_30d_ma'] < 15
        else calculate_gray_leaf_spot_risk_function(
            r['air_temp_min_c_21d_ma'],
            r['dp_min_30d_c_ma']
        ),
        axis=1
    )
    df[['gls_risk', 'gls_risk_class']] = gls

    # Frogeye Leaf Spot
    fe = df.apply(
        lambda r: pd.Series({'fe_risk': -1, 'fe_risk_class': 'Inactive'})
        if pd.isna(r['air_temp_avg_c_30d_ma']) or r['air_temp_avg_c_30d_ma'] < 15
        else calculate_frogeye_leaf_spot_function(
            r['air_temp_max_c_30d_ma'],
            r['rh_above_80_day_30d_ma']
        ),
        axis=1
    )
    df[['fe_risk', 'fe_risk_class']] = fe

    # White Mold Irrigated
    wmi = df.apply(
        lambda r: pd.Series({
            'whitemold_irr_30in_risk': -1, 'whitemold_irr_15in_risk': -1,
            'whitemold_irr_15in_class': 'Inactive', 'whitemold_irr_30in_class': 'Inactive'
        })
        if pd.isna(r['air_temp_avg_c_30d_ma']) or r['air_temp_avg_c_30d_ma'] < 15
        else calculate_irrigated_risk(
            r['air_temp_max_c_30d_ma'],
            r['rh_max_30d_ma']
        ),
        axis=1
    )
    df[wmi.columns] = wmi

    # White Mold Non-Irrigated
    wmn = df.apply(
        lambda r: pd.Series({'whitemold_nirr_risk': -1, 'whitemold_nirr_risk_class': 'Inactive'})
        if pd.isna(r['air_temp_avg_c_30d_ma']) or r['air_temp_avg_c_30d_ma'] < 15
        else calculate_non_irrigated_risk(
            r['air_temp_max_c_30d_ma'],
            r['rh_max_30d_ma'],
            r['max_ws_30d_ma']
        ),
        axis=1
    )
    df[wmn.columns] = wmn

    return df


def chunk_dataframe(df, num_chunks):
    n = len(df)
    if n <= 100:
        return [df]
    size = min(max(n // num_chunks, 100), 1000)
    return [df.iloc[i:i + size] for i in range(0, n, size)]


async def _load_stations(session, input_date: str) -> pd.DataFrame | None:
    """
    BUG FIX: The original code used `today.day == 32` (never true!) and
    `today.day == 1` (only refreshes once a month) as conditions to fetch
    fresh station data. This meant the cache file was almost never written
    and always expected to exist already.

    New logic:
      - Always try to read from the cache file first.
      - If the cache is missing OR older than 7 days, fetch from the API
        and write a fresh cache.
    """
    cache_stale = True
    if os.path.exists(STATIONS_CACHE_FILE):
        age_days = (
            datetime.now() - datetime.fromtimestamp(os.path.getmtime(STATIONS_CACHE_FILE))
        ).total_seconds() / 86400
        if age_days < 7:
            cache_stale = False

    if not cache_stale:
        logging.info("Loading stations from cache.")
        return pd.read_csv(STATIONS_CACHE_FILE)

    logging.info("Fetching fresh station list from API.")
    url = f"{BASE_URL}/stations/"
    async with session.get(url) as resp:
        if resp.status != 200:
            logging.error(f"Station API returned {resp.status}")
            # Fall back to stale cache if it exists
            if os.path.exists(STATIONS_CACHE_FILE):
                return pd.read_csv(STATIONS_CACHE_FILE)
            return None

        stations = pd.DataFrame(await resp.json())

    stations = stations[~stations['station_id'].isin(STATIONS_TO_EXCLUDE)]
    stations['earliest_api_date'] = pd.to_datetime(
        stations['earliest_api_date'], format="%m/%d/%Y", errors='coerce'
    )

    input_dt  = pd.to_datetime(input_date)
    threshold = input_dt - pd.Timedelta(days=MIN_DAYS_ACTIVE)
    stations  = stations[stations['earliest_api_date'] <= threshold]

    stations.to_csv(STATIONS_CACHE_FILE, index=False)
    logging.info(f"Cached {len(stations)} stations.")
    return stations


async def retrieve_tarspot_all_stations_async(input_date, input_station_id=None, days=1):
    """
    BUG FIX SUMMARY vs original:
      1. get_async_session() is now a plain function (not async def), so
         `async with get_async_session() as session` works correctly.
      2. Station loading logic (`today.day == 32` / `today.day == 1`) replaced
         with time-based cache in `_load_stations()`.
      3. The massive duplicated block of code pasted inside the `if today.day == 32`
         branch (lines ~298-650 of original) has been removed entirely.
      4. `result_type='expand'` replaced with proper pd.Series returns so
         column assignment doesn't silently produce wrong shapes.
      5. NaN guard added to risk checks (`pd.isna(...)`) so rows with
         insufficient rolling-window data don't crash.
    """
    try:
        async with get_async_session() as session:

            # 1) Load station list
            stations = await _load_stations(session, input_date)
            if stations is None or stations.empty:
                logging.warning("No stations available.")
                return None

            # 2) Optionally filter to a single station
            if input_station_id:
                stations = stations[stations['station_id'] == input_station_id]
                if stations.empty:
                    logging.warning(f"Station {input_station_id} not found.")
                    return None

            station_ids = stations['station_id'].tolist()

            # 3) Fetch measurements
            dfs = await process_stations_in_batches(session, station_ids, input_date, days)
            if not dfs:
                logging.warning("No measurement data returned.")
                return None

            meas_df = pd.concat(dfs, ignore_index=True)

            # 4) Merge station metadata
            meta = stations[[
                'station_id', 'station_name', 'city', 'county',
                'latitude', 'longitude', 'region', 'state', 'station_timezone'
            ]]
            merged = pd.merge(meta, meas_df, on='station_id', how='inner')
            if merged.empty:
                logging.warning("Merge produced empty DataFrame.")
                return None

            # 5) Compute risks (parallel for large datasets)
            chunks = chunk_dataframe(merged, os.cpu_count() or 1)
            with ProcessPoolExecutor() as exe:
                futures   = [exe.submit(compute_risks, c) for c in chunks]
                processed = [f.result() for f in as_completed(futures)]
            final = pd.concat(processed, ignore_index=True)

            # 6) Finalize dates
            final['date'] = pd.to_datetime(final['date'])
            final['forecasting_date'] = (
                final['date'] + timedelta(days=1)
            ).dt.strftime('%Y-%m-%d')

            # Return only columns that actually exist (guards against partial data)
            available = [c for c in FINAL_COLUMNS if c in final.columns]
            return final[available]

    except Exception as e:
        logging.error(f"retrieve_tarspot_all_stations_async failed: {e}")
        traceback.print_exc()
        return None


def retrieve(input_date: str, input_station_id: str | None = None, days: int = 1):
    """
    BUG FIX: The original `retrieve()` called a non-existent `create_session()`
    function, so it always returned None immediately before even running the
    async logic. Fixed to call `asyncio.run()` directly on the async function.
    """
    return asyncio.run(
        retrieve_tarspot_all_stations_async(input_date, input_station_id, days)
    )