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
    fahrenheit_to_celsius,
    cereal_rye_report,          # ← needed for biomass risk classification
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

BIOMASS_CACHE_DIR = "station_biomass_cache"
os.makedirs(BIOMASS_CACHE_DIR, exist_ok=True)

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
    'whitemold_irr_15in_class', 'whitemold_irr_30in_class',
    # ── Biomass (only present when planting_date / termination_date are supplied) ──
    'cgdd_60d_ap', 'rain_60d_ap', 'cgdd_60d_bt',
    'biomass_lb_acre', 'biomass_color', 'biomass_message',
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
    Plain factory (not async) — callers use `async with get_async_session() as session`.
    """
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=50, ttl_dns_cache=300),
        timeout=aiohttp.ClientTimeout(total=60)
    )


# ─── Disease-risk measurements ────────────────────────────────────────────────

async def api_call_wisconet_data_async(session, station_id, end_time):
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
            f"{BASE_URL}/stations/{station_id}/measures",
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


# ─── Biomass measurements ─────────────────────────────────────────────────────

async def fetch_biomass_data_async(session, station_id, planting_date, termination_date):
    """
    Fetch daily temp + rainfall between planting_date and termination_date,
    compute the three inputs needed by cereal_rye_report(), then call it so
    the biomass estimate and classification travel together as one record.

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
            "fields":     "daily_air_temp_f_avg,daily_rain_in_tot"
        }

        payload = await api_call_with_retry(
            session,
            f"{BASE_URL}/stations/{station_id}/measures",
            params
        )
        data = payload.get("data", []) if payload else []
        if not data:
            return None

        df = pd.DataFrame(data)

        df['temp_avg_f'] = df['measures'].apply(
            lambda m: next((v for (i, v) in m if i == 3), np.nan)
        )
        df['rain_in'] = df['measures'].apply(
            lambda m: next((v for (i, v) in m if i == 15), np.nan)
        )

        df['collection_time'] = pd.to_datetime(df['collection_time'], unit='s')
        df = df.sort_values('collection_time')

        # 1. CGDD and rainfall for the first 60 days after planting
        day_60_mark  = start_dt + timedelta(days=60)
        df_60d       = df[df['collection_time'] <= day_60_mark]
        cgdd_60d_ap  = float((df_60d['temp_avg_f'] - 32).clip(lower=0).sum())
        rain_60d_ap  = float(df_60d['rain_in'].sum())

        # 2. CGDD for the 30-day window before termination
        bt_start    = end_dt - timedelta(days=30)
        df_bt       = df[df['collection_time'] >= bt_start]
        cgdd_60d_bt = float((df_bt['temp_avg_f'] - 32).clip(lower=0).sum())

        # 3. Run the biomass model so risk classification is co-located with the data
        report = cereal_rye_report(cgdd_60d_ap, rain_60d_ap, cgdd_60d_bt)

        return {
            "station_id":      station_id,
            "cgdd_60d_ap":     cgdd_60d_ap,
            "rain_60d_ap":     rain_60d_ap,
            "cgdd_60d_bt":     cgdd_60d_bt,
            "biomass_lb_acre": report["biomass"],
            "biomass_color":   report["color"],
            "biomass_message": report["message"],
        }

    except Exception as e:
        logging.error(f"Biomass fetch failed for {station_id}: {e}")
        return None


async def process_biomass_in_batches(session, station_ids, planting_date, termination_date):
    """
    Fetch biomass data for all stations in BATCH_SIZE chunks,
    mirroring process_stations_in_batches for consistent rate-limit behaviour.
    Returns a DataFrame or None.
    """
    results = []
    for i in range(0, len(station_ids), BATCH_SIZE):
        batch     = station_ids[i:i + BATCH_SIZE]
        tasks     = [
            fetch_biomass_data_async(session, sid, planting_date, termination_date)
            for sid in batch
        ]
        batch_out = await asyncio.gather(*tasks)
        results.extend([r for r in batch_out if r is not None])
        if i + BATCH_SIZE < len(station_ids):
            await asyncio.sleep(0.5)

    if not results:
        return None
    return pd.DataFrame(results)


# ─── Risk computation ─────────────────────────────────────────────────────────

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


# ─── Station loader ───────────────────────────────────────────────────────────

async def _load_stations(session, input_date: str) -> pd.DataFrame | None:
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


# ─── Main async pipeline ──────────────────────────────────────────────────────

async def retrieve_tarspot_all_stations_async(
    input_date,
    input_station_id=None,
    days=1,
    planting_date=None,       # e.g. "2024-04-15"  — biomass only
    termination_date=None,    # e.g. "2024-07-01"  — biomass only
):
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

            # 3) Fetch disease-risk measurements AND biomass data concurrently.
            #    Biomass is only fetched when both crop dates are provided.
            run_biomass = bool(planting_date and termination_date)

            if run_biomass:
                dfs, biomass_df = await asyncio.gather(
                    process_stations_in_batches(session, station_ids, input_date, days),
                    process_biomass_in_batches(session, station_ids, planting_date, termination_date),
                )
            else:
                dfs        = await process_stations_in_batches(session, station_ids, input_date, days)
                biomass_df = None

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

            # 5) Merge biomass data.
            #    Left join → stations with failed biomass fetches keep NaN
            #    rather than being silently dropped from the output.
            if biomass_df is not None and not biomass_df.empty:
                merged = pd.merge(merged, biomass_df, on='station_id', how='left')
                logging.info(
                    f"Biomass data merged for {biomass_df['station_id'].nunique()} station(s)."
                )
            else:
                # Guarantee the biomass columns exist so FINAL_COLUMNS filtering
                # is consistent regardless of whether biomass was requested.
                for col in ('cgdd_60d_ap', 'rain_60d_ap', 'cgdd_60d_bt',
                            'biomass_lb_acre', 'biomass_color', 'biomass_message'):
                    merged[col] = np.nan

            # 6) Compute disease risks (parallelised for large datasets)
            chunks = chunk_dataframe(merged, os.cpu_count() or 1)
            with ProcessPoolExecutor() as exe:
                futures   = [exe.submit(compute_risks, c) for c in chunks]
                processed = [f.result() for f in as_completed(futures)]
            final = pd.concat(processed, ignore_index=True)

            # 7) Finalise dates
            final['date'] = pd.to_datetime(final['date'])
            final['forecasting_date'] = (
                final['date'] + timedelta(days=1)
            ).dt.strftime('%Y-%m-%d')

            # Only return columns that actually exist (guards partial data)
            available = [c for c in FINAL_COLUMNS if c in final.columns]
            return final[available]

    except Exception as e:
        logging.error(f"retrieve_tarspot_all_stations_async failed: {e}")
        traceback.print_exc()
        return None


# ─── Public entry points ──────────────────────────────────────────────────────

def retrieve(
    input_date: str,
    input_station_id: str | None = None,
    days: int = 1,
    planting_date: str | None = None,
    termination_date: str | None = None,
):
    """
    Synchronous wrapper around the async pipeline.

    Args:
        input_date:        The reference date for disease-risk rolling averages
                           (YYYY-MM-DD).
        input_station_id:  Limit results to a single station (optional).
        days:              How many recent days of risk values to return.
        planting_date:     Cover-crop planting date for biomass estimation
                           (YYYY-MM-DD, optional).
        termination_date:  Cover-crop termination date for biomass estimation
                           (YYYY-MM-DD, optional).

    Returns:
        pd.DataFrame with disease-risk columns and, when crop dates are
        supplied, the biomass columns (cgdd_60d_ap, rain_60d_ap, cgdd_60d_bt,
        biomass_lb_acre, biomass_color, biomass_message).
    """
    return asyncio.run(
        retrieve_tarspot_all_stations_async(
            input_date, input_station_id, days, planting_date, termination_date
        )
    )