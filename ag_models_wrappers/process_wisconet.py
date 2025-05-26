from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import pytz
import os
import pickle
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from ag_models_wrappers.forecasting_models import (
    calculate_tarspot_risk_function,
    calculate_gray_leaf_spot_risk_function,
    calculate_frogeye_leaf_spot_function,
    calculate_irrigated_risk,
    calculate_non_irrigated_risk,
    fahrenheit_to_celsius
)

# ─── Constants ────────────────────────────────────────────────────────────────
BATCH_SIZE = 20
BASE_URL = "https://wisconet.wisc.edu/api/v1"
MIN_DAYS_ACTIVE = 38

MEASUREMENTS_CACHE_DIR = "station_measurements_cache"
os.makedirs(MEASUREMENTS_CACHE_DIR, exist_ok=True)

STATIONS_CACHE_FILE = "wisconsin_stations_cache.csv"
CACHE_EXPIRY_DAYS = 7
STATIONS_TO_EXCLUDE = ['MITEST1', 'WNTEST1']

ROLLING_MAP = {
    'rh_above_90_night_14d_ma': ('nhours_rh_above_90', 14),
    'rh_above_80_day_30d_ma': ('hours_rh_above_80_day', 30),
    'air_temp_min_c_21d_ma': ('air_temp_min_c', 21),
    'air_temp_max_c_30d_ma': ('air_temp_max_c', 30),
    'air_temp_avg_c_30d_ma': ('air_temp_avg_c', 30),
    'rh_max_30d_ma': ('rh_max', 30),
    'max_ws_30d_ma': ('max_ws', 30),
    'dp_min_30d_c_ma': ('min_dp_c', 30),
}

DESIRED_COLS = [
    'station_id', 'date',
    'rh_above_90_night_14d_ma', 'rh_above_80_day_30d_ma',
    'air_temp_min_c_21d_ma', 'air_temp_max_c_30d_ma', 'air_temp_avg_c_30d_ma',
    'rh_max_30d_ma', 'max_ws_30d_ma', 'dp_min_30d_c_ma'
]

# ─── Helper Functions ────────────────────────────────────────────────────────

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

async def get_async_session():
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=50, ttl_dns_cache=300),
        timeout=aiohttp.ClientTimeout(total=60)
    )

async def api_call_wisconet_data_async(session, station_id, end_time):
    try:
        end_dt  = datetime.strptime(end_time, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=MIN_DAYS_ACTIVE)
        params = {
            "start_time": int(start_dt.timestamp()),
            "end_time":   int(end_dt.timestamp()),
            "fields": "60min_relative_humidity_pct_avg,60min_air_temp_f_avg,60min_dew_point_f_avg,60min_wind_speed_mph_max"
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
                lambda m: next((v for (i, v) in m if i == mid), np.nan)
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
            age_hr = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).total_seconds() / 3600
            if age_hr < 6:
                return pd.read_pickle(cache_file)

        df = await api_call_wisconet_data_async(session, station_id, end_time)
        if df is None or df.empty:
            return None

        df = df.sort_values('date', ascending=False).head(days)
        for col in DESIRED_COLS:
            if col not in df:
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
        batch = station_ids[i:i+BATCH_SIZE]
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
        lambda r: (-1, 'Inactive') if r['air_temp_avg_c_30d_ma'] < 15 else
                  calculate_tarspot_risk_function(
                      r['air_temp_avg_c_30d_ma'],
                      r['rh_max_30d_ma'],
                      r['rh_above_90_night_14d_ma']
                  ),
        axis=1, result_type='expand'
    )
    tar.columns = ['tarspot_risk', 'tarspot_risk_class']
    df[['tarspot_risk', 'tarspot_risk_class']] = tar

    # Gray Leaf Spot
    gls = df.apply(
        lambda r: (-1, 'Inactive') if r['air_temp_avg_c_30d_ma'] < 15 else
                  calculate_gray_leaf_spot_risk_function(
                      r['air_temp_min_c_21d_ma'],
                      r['dp_min_30d_c_ma']
                  ),
        axis=1, result_type='expand'
    )
    gls.columns = ['gls_risk', 'gls_risk_class']
    df[['gls_risk', 'gls_risk_class']] = gls

    # Frogeye Leaf Spot
    fe = df.apply(
        lambda r: (-1, 'Inactive') if r['air_temp_avg_c_30d_ma'] < 15 else
                  calculate_frogeye_leaf_spot_function(
                      r['air_temp_max_c_30d_ma'],
                      r['rh_above_80_day_30d_ma']
                  ),
        axis=1, result_type='expand'
    )
    fe.columns = ['fe_risk', 'fe_risk_class']
    df[['fe_risk', 'fe_risk_class']] = fe

    # White Mold Irrigated
    wmi = df.apply(
        lambda r: (-1, -1, 'Inactive', 'Inactive') if r['air_temp_avg_c_30d_ma'] < 15 else
                  calculate_irrigated_risk(
                      r['air_temp_max_c_30d_ma'],
                      r['rh_max_30d_ma']
                  ),
        axis=1, result_type='expand'
    )
    wmi.columns = [
        'whitemold_irr_30in_risk',
        'whitemold_irr_15in_risk',
        'whitemold_irr_15in_class',
        'whitemold_irr_30in_class'
    ]
    df[wmi.columns] = wmi

    # White Mold Non-Irrigated
    wmn = df.apply(
        lambda r: (-1, 'Inactive') if r['air_temp_avg_c_30d_ma'] < 15 else
                  calculate_non_irrigated_risk(
                      r['air_temp_max_c_30d_ma'],
                      r['rh_max_30d_ma'],
                      r['max_ws_30d_ma']
                  ),
        axis=1, result_type='expand'
    )
    wmn.columns = ['whitemold_nirr_risk', 'whitemold_nirr_risk_class']
    df[wmn.columns] = wmn

    return df

def chunk_dataframe(df, num_chunks):
    n = len(df)
    if n <= 100:
        return [df]
    size = min(max(n // num_chunks, 100), 1000)
    return [df.iloc[i:i+size] for i in range(0, n, size)]

async def retrieve_tarspot_all_stations_async(input_date, input_station_id=None, days=1):
    FINAL_COLUMNS = [
        'station_id','date','forecasting_date','station_name','city','county',
        'latitude','longitude','region','state','station_timezone',
        'tarspot_risk','tarspot_risk_class','gls_risk','gls_risk_class',
        'fe_risk','fe_risk_class','whitemold_nirr_risk','whitemold_nirr_risk_class',
        'whitemold_irr_30in_risk','whitemold_irr_15in_risk',
        'whitemold_irr_15in_class','whitemold_irr_30in_class'
    ]

    async with await get_async_session() as session:
        # 1) Load or fetch station list, with caching
        stations = None
        if os.path.exists(STATIONS_CACHE_FILE):
            age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(STATIONS_CACHE_FILE))).days
            if age < CACHE_EXPIRY_DAYS:
                stations = pd.read_csv(STATIONS_CACHE_FILE)
        if stations is None:
            url = (BASE_URL.replace('/api/v1','')
                   + "/pywisconet_wrapper/wisconet/active_stations/"
                   + f"?min_days_active=15&start_date={input_date}")
            async with session.get(url) as resp:
                if resp.status == 200:
                    stations = pd.DataFrame(await resp.json())
                    stations.to_csv(STATIONS_CACHE_FILE, index=False)
                else:
                    return None

        # Exclude tests
        stations = stations[~stations['station_id'].isin(STATIONS_TO_EXCLUDE)]

        # ---- Fix for timezone comparison error ----
        if 'earliest_api_date' in stations.columns:
            # parse as UTC
            stations['earliest_api_date'] = pd.to_datetime(
                stations['earliest_api_date'], utc=True
            )
            # make cutoff tz-aware UTC
            cutoff = pd.to_datetime(input_date).tz_localize('UTC') - pd.Timedelta(days=32)
            stations = stations[stations['earliest_api_date'] < cutoff]

        # 2) Filter by input_station_id if provided
        if input_station_id:
            ids = [s.strip() for s in input_station_id.split(',')]
            stations = stations[stations['station_id'].isin(ids)]

        station_ids = stations['station_id'].tolist()
        if not station_ids:
            return None

        # 3) Fetch measurements
        dfs = await process_stations_in_batches(session, station_ids, input_date, days)
        if not dfs:
            return None
        meas_df = pd.concat(dfs, ignore_index=True)

        # 4) Merge metadata
        meta = stations[[
            'station_id','station_name','city','county',
            'latitude','longitude','region','state','station_timezone'
        ]]
        merged = pd.merge(meta, meas_df, on='station_id', how='inner')

        # 5) Compute risks in parallel
        chunks = chunk_dataframe(merged, os.cpu_count() or 1)
        with ProcessPoolExecutor() as exe:
            futures = [exe.submit(compute_risks, c) for c in chunks]
            processed = [f.result() for f in as_completed(futures)]
        final = pd.concat(processed, ignore_index=True)

        # 6) Finalize
        final['date'] = pd.to_datetime(final['date'])
        final['forecasting_date'] = (final['date'] + timedelta(days=1)).dt.strftime('%Y-%m-%d')
        return final[FINAL_COLUMNS]

def retrieve(input_date, input_station_id=None, days=1):
    return asyncio.run(retrieve_tarspot_all_stations_async(input_date, input_station_id, days))
