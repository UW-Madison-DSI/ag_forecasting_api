from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
import pytz

from ag_models_wrappers.forecasting_models import *

# Constants
BASE_URL = "https://wisconet.wisc.edu/api/v1"
STATIONS_TO_EXCLUDE = ['MITEST1', 'WNTEST1']
MIN_DAYS_ACTIVE = 38
MAX_WORKERS = 20  # Adjust based on your system capabilities

# Map measures to corresponding columns
MEASURE_MAP = {
    4: 'air_temp_max_f',
    6: 'air_temp_min_f',
    20: 'rh_max',
    12: 'min_dp',
    56: 'max_ws'  # units in mph
}

# Session setup for concurrent requests
session = requests.Session()


# Define an async session for fully asynchronous operations
async def get_async_session():
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=MAX_WORKERS))


async def api_call_wisconet_data_daily_async(session, station_id, input_date):
    """Asynchronous version of api_call_wisconet_data_daily"""
    end_date = datetime.strptime(input_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=MIN_DAYS_ACTIVE)

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    url = f"{BASE_URL}/stations/{station_id}/measures"
    params = {
        "start_time": start_timestamp,
        "end_time": end_timestamp,
        "fields": "daily_air_temp_f_avg,daily_air_temp_f_max,daily_air_temp_f_min,daily_dew_point_f_max,daily_dew_point_f_min,daily_relative_humidity_pct_max,daily_relative_humidity_pct_min,daily_wind_speed_mph_max,daily_dew_point_f_avg"
    }

    async with session.get(url, params=params) as response:
        if response.status == 200:
            data = await response.json()
            data = data.get("data", [])
            df = pd.DataFrame(data)

            # Rest of the processing remains the same as in the original function
            result_df = pd.DataFrame({
                'o_collection_time': pd.to_datetime(df['collection_time'], unit='s'),
                'collection_time': pd.to_datetime(df['collection_time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(
                    'US/Central'),
                'air_temp_max_f': np.nan,
                'air_temp_min_f': np.nan,
                'rh_max': np.nan,
                'min_dp': np.nan,
                'max_ws': np.nan
            })

            # Vectorized population of measures
            for measure_id, column_name in MEASURE_MAP.items():
                result_df[column_name] = df['measures'].apply(
                    lambda measures: next((m[1] for m in measures if m[0] == measure_id), np.nan))

            # Convert Fahrenheit to Celsius
            result_df['min_dp_c'] = fahrenheit_to_celsius(result_df['min_dp'])
            result_df['air_temp_max_c'] = fahrenheit_to_celsius(result_df['air_temp_max_f'])
            result_df['air_temp_min_c'] = fahrenheit_to_celsius(result_df['air_temp_min_f'])
            result_df['air_temp_avg_c'] = fahrenheit_to_celsius(
                result_df[['air_temp_max_f', 'air_temp_min_f']].mean(axis=1))

            # Calculate moving averages
            result_df['air_temp_min_c_21d_ma'] = result_df['air_temp_min_c'].rolling(window=21, min_periods=1).mean()
            result_df['air_temp_max_c_30d_ma'] = result_df['air_temp_max_c'].rolling(window=30, min_periods=1).mean()
            result_df['air_temp_avg_c_30d_ma'] = result_df['air_temp_avg_c'].rolling(window=30, min_periods=1).mean()
            result_df['rh_max_30d_ma'] = result_df['rh_max'].rolling(window=30, min_periods=1).mean()
            result_df['max_ws_30d_ma'] = result_df['max_ws'].rolling(window=30, min_periods=1).mean()
            result_df['dp_min_30d_c_ma'] = result_df['min_dp_c'].rolling(window=30, min_periods=1).mean()
            result_df['collection_time'] = pd.to_datetime(result_df['collection_time'])
            result_df['date'] = result_df['collection_time'].dt.strftime('%Y-%m-%d')

            return result_df
        else:
            print(f"Error fetching data for station {station_id}, status code {response.status}")
            return None


async def api_call_wisconet_data_rh_async(session, station_id, end_time):
    """Asynchronous version of api_call_wisconet_data_rh"""
    try:
        endpoint = f'/stations/{station_id}/measures'

        end_date = datetime.strptime(end_time, "%Y-%m-%d")
        start_date = end_date - timedelta(days=MIN_DAYS_ACTIVE)

        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        params = {
            "start_time": start_timestamp,
            "end_time": end_timestamp,
            "fields": "60min_relative_humidity_pct_avg"
        }

        async with session.get(f"{BASE_URL}{endpoint}", params=params) as response:
            if response.status == 200:
                data = await response.json()
                data = data.get("data", [])
                df = pd.DataFrame(data)

                # Create the result DataFrame
                result_df = pd.DataFrame({
                    'collection_time': pd.to_datetime(df['collection_time'], unit='s').dt.tz_localize(
                        'UTC').dt.tz_convert('US/Central'),
                    'rh_avg': [
                        next((m[1] for m in item.measures if m[0] == 19), np.nan)
                        for item in df.itertuples()
                    ]
                })

                # Processing steps remain the same
                result_df['hour'] = result_df['collection_time'].dt.hour

                result_df['rh_night_above_90'] = np.where(
                    (result_df['rh_avg'] >= 90) & ((result_df['hour'] >= 20) | (result_df['hour'] <= 6)),
                    1, 0
                )

                result_df['rh_day_above_80'] = np.where(
                    (result_df['rh_avg'] >= 80),
                    1, 0
                )
                result_df['collection_time'] = pd.to_datetime(result_df['collection_time'])
                result_df['date'] = result_df['collection_time'].dt.strftime('%Y-%m-%d')

                daily_rh_above_90 = result_df.groupby('date').agg(
                    nhours_rh_above_90=('rh_night_above_90', 'sum'),
                    hours_rh_above_80_day=('rh_day_above_80', 'sum')
                ).reset_index()

                daily_rh_above_90['rh_above_90_night_14d_ma'] = daily_rh_above_90['nhours_rh_above_90'].rolling(
                    window=14, min_periods=14).mean()
                daily_rh_above_90['rh_above_80_day_30d_ma'] = daily_rh_above_90['hours_rh_above_80_day'].rolling(
                    window=30, min_periods=30).mean()

                return daily_rh_above_90
            else:
                print(f"Error: {response.status}")
                return None

    except Exception as e:
        print(f"Failed to retrieve or process data: {e}")
        return None


async def one_day_measurements_async(session, station_id, end_time, days):
    """Asynchronous version of one_day_measurements"""
    try:
        if days is None or days == 0:
            days = 1
        elif days > 7:
            days = 8

        # Run both API calls concurrently using asyncio
        tasks = [
            api_call_wisconet_data_rh_async(session, station_id, end_time),
            api_call_wisconet_data_daily_async(session, station_id, end_time)
        ]

        # Wait for both tasks to complete
        results = await asyncio.gather(*tasks)
        daily_rh_above_90, result_df = results

        # Validate the data
        if daily_rh_above_90 is None or daily_rh_above_90.empty:
            raise ValueError(f"No RH data found for station {station_id}.")
        if result_df is None or result_df.empty:
            raise ValueError(f"No daily data found for station {station_id}.")

        # Process the data
        daily_rh_above_90 = daily_rh_above_90.sort_values('date', ascending=False).head(days)
        result_df = result_df.sort_values('date', ascending=False).head(days)

        # Merge on the 'date' column
        combined_df = pd.merge(
            daily_rh_above_90[['date', 'rh_above_90_night_14d_ma', 'rh_above_80_day_30d_ma']],
            result_df[['date', 'air_temp_max_c_30d_ma', 'air_temp_min_c_21d_ma',
                       'air_temp_avg_c_30d_ma', 'rh_max_30d_ma', 'max_ws_30d_ma', 'dp_min_30d_c_ma']],
            on='date', how='inner'
        )

        combined_df['station_id'] = station_id
        combined_df = combined_df.sort_values('date', ascending=False).head(days)

        return combined_df

    except Exception as e:
        print(f"Error in processing data for station {station_id}: {e}")
        return None


def compute_risks(df_chunk):
    """Compute risk metrics for a chunk of data"""
    df_chunk = df_chunk.copy()

    # Calculate risks in parallel using apply with n_jobs parameter
    # or split further into smaller chunks for parallelization

    # Apply risk calculations using vectorized operations where possible
    # For tarspot risk
    tarspot_results = df_chunk.apply(
        lambda row: pd.Series(
            calculate_tarspot_risk_function(
                row['air_temp_avg_c_30d_ma'],
                row['rh_max_30d_ma'],
                row['rh_above_90_night_14d_ma']
            )
        ),
        axis=1
    )
    df_chunk[['tarspot_risk', 'tarspot_risk_class']] = tarspot_results

    # For gray leaf spot risk
    gls_results = df_chunk.apply(
        lambda row: pd.Series(
            calculate_gray_leaf_spot_risk_function(
                row['air_temp_min_c_21d_ma'],
                row['dp_min_30d_c_ma']
            )
        ),
        axis=1
    )
    df_chunk[['gls_risk', 'gls_risk_class']] = gls_results

    # For frogeye leaf spot risk
    fe_results = df_chunk.apply(
        lambda row: pd.Series(
            calculate_frogeye_leaf_spot_function(
                row['air_temp_max_c_30d_ma'],
                row['rh_above_80_day_30d_ma']
            )
        ),
        axis=1
    )
    df_chunk[['fe_risk', 'fe_risk_class']] = fe_results

    # For white mold irrigated risk
    whitemold_irr_results = df_chunk.apply(
        lambda row: pd.Series(
            calculate_irrigated_risk(
                row['air_temp_max_c_30d_ma'],
                row['rh_max_30d_ma']
            )
        ),
        axis=1
    )
    df_chunk[['whitemold_irr_30in_risk', 'whitemold_irr_15in_risk']] = whitemold_irr_results

    # For white mold non-irrigated risk
    #whitemold_nirr_results = df_chunk.apply(
    #    lambda row: pd.Series(
    #        calculate_non_irrigated_risk(
    #            row['air_temp_max_c_30d_ma'],
    #            row['max_ws_30d_ma']
    #        )
    #    ),
    #    axis=1
    #)
    #df_chunk['whitemold_nirr_risk'] = whitemold_nirr_results

    return df_chunk


# Function to split data into roughly equal chunks
def chunk_dataframe(df, num_chunks):
    """Split DataFrame into roughly equal chunks"""
    chunk_size = max(1, len(df) // num_chunks)
    return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]


import os
from datetime import datetime, timedelta
import pandas as pd

# Add these constants
STATIONS_CACHE_FILE = "wisconsin_stations_cache.csv"
CACHE_EXPIRY_DAYS = 7  # How often to refresh the cache


async def get_stations_with_caching(session, input_date):
    """
    Retrieves station data with caching mechanism to avoid unnecessary API calls.
    Only fetches from API if cache doesn't exist, is expired, or it's the first day of the month
    (to check for new stations).
    """
    today = datetime.now()
    is_first_day = today.day == 8  # Check if it's the first day of the month

    # Check if cache exists and is not expired
    if os.path.exists(STATIONS_CACHE_FILE) and not is_first_day:
        # Check cache file modification time
        cache_mod_time = datetime.fromtimestamp(os.path.getmtime(STATIONS_CACHE_FILE))
        cache_age_days = (today - cache_mod_time).days

        if cache_age_days < CACHE_EXPIRY_DAYS:
            print(f"Using cached station data (last updated {cache_age_days} days ago)")
            return pd.read_csv(STATIONS_CACHE_FILE)

    # If we get here, we need to fetch fresh data
    print("Fetching fresh station data from API...")
    allstations_url = (
        f"https://connect.doit.wisc.edu/pywisconet_wrapper/wisconet/active_stations/"
        f"?min_days_active=15&start_date={input_date}"
    )

    async with session.get(allstations_url) as response:
        if response.status == 200:
            stations_data = await response.json()
            allstations = pd.DataFrame(stations_data)

            # Save to cache file
            allstations.to_csv(STATIONS_CACHE_FILE, index=False)
            print(f"Updated station cache with {len(allstations)} stations")

            return allstations
        else:
            # If API call fails but cache exists, use cache as fallback
            if os.path.exists(STATIONS_CACHE_FILE):
                print(f"API call failed (status {response.status}). Using cached station data as fallback.")
                return pd.read_csv(STATIONS_CACHE_FILE)
            else:
                print(f"Error fetching station data, status code {response.status}")
                return None


# Update the main retrieve function to use the cached stations
async def retrieve_tarspot_all_stations_async1(input_date, input_station_id=None, days=1):
    """
    Main asynchronous function to retrieve and process data for all stations
    """
    FINAL_COLUMNS = [
        'station_id', 'date', 'forecasting_date', 'location',
        'station_name', 'city', 'county', 'latitude',
        'longitude', 'region', 'state',
        'station_timezone',
        'tarspot_risk', 'tarspot_risk_class',
        'gls_risk', 'gls_risk_class',
        'fe_risk', 'fe_risk_class',
        'whitemold_irr_30in_risk',
        'whitemold_irr_15in_risk'
    ]

    print('input_date --->>>>>', input_date)

    async with aiohttp.ClientSession() as session:
        # Use the cached station data
        allstations = await get_stations_with_caching(session, input_date)

        if allstations is None:
            return None

        print("Stations retrieved:", len(allstations))

        if input_station_id:
            stations = allstations[
                (allstations['station_id'] == input_station_id) &
                (~allstations['station_id'].isin(STATIONS_TO_EXCLUDE))
                ]
            # Process single station
            async with await get_async_session() as data_session:
                all_results = await one_day_measurements_async(data_session, input_station_id, input_date, days)
        else:
            stations = allstations[~allstations['station_id'].isin(STATIONS_TO_EXCLUDE)]
            station_ids = stations['station_id'].values

            # Process all stations concurrently using asyncio
            async with await get_async_session() as data_session:
                tasks = [one_day_measurements_async(data_session, st, input_date, days) for st in station_ids]
                results = await asyncio.gather(*tasks)

                # Filter out None results and concat
                valid_results = [res for res in results if res is not None]
                if not valid_results:
                    return None
                all_results = pd.concat(valid_results, ignore_index=True)

        # Rest of the function remains the same...
        # Merge station info with API data
        daily_data = stations.merge(all_results, on='station_id', how='inner')

        # Compute risk metrics code...

        # Return the final DataFrame with selected columns
        return daily_data[FINAL_COLUMNS]

async def retrieve_tarspot_all_stations_async(input_date, input_station_id=None, days=1):
    """
    Main asynchronous function to retrieve and process data for all stations.
    """
    FINAL_COLUMNS = [
        'station_id', 'date', 'forecasting_date', 'location',
        'station_name', 'city', 'county', 'latitude',
        'longitude', 'region', 'state',
        'station_timezone',
        'tarspot_risk', 'tarspot_risk_class',
        'gls_risk', 'gls_risk_class',
        'fe_risk', 'fe_risk_class',
        'whitemold_irr_30in_risk',
        'whitemold_irr_15in_risk'
        # 'whitemold_nirr_risk'
    ]

    print('input_date --->>>>>', input_date)

    allstations_url = (
        f"https://connect.doit.wisc.edu/pywisconet_wrapper/wisconet/active_stations/"
        f"?min_days_active=15&start_date={input_date}"
    )

    # Retrieve all active stations data
    ct = pytz.timezone('America/Chicago')
    today_ct = datetime.now(ct)

    # If it's not day 1 at 6 am CT, load from backup and filter
    if not (today_ct.day == 1 and today_ct.hour == 6):
        df = pd.read_csv('stations_backup.csv')
        df['earliest_api_date'] = pd.to_datetime(df['earliest_api_date'], utc=True).dt.tz_localize(None)
        # Exclude stations in the exclusion list
        df = df[~df['station_id'].isin(STATIONS_TO_EXCLUDE)]
        # Filter by earliest_api_date
        input_date_transformed = pd.to_datetime(input_date)
        date_limit = input_date_transformed - pd.Timedelta(days=32)
        allstations = df[df['earliest_api_date'] < pd.to_datetime(date_limit)]
        print(allstations[['station_id', 'earliest_api_date']])
    else:
        async with session.get(allstations_url) as response:
            if response.status == 200:
                stations_data = await response.json()
                allstations = pd.DataFrame(stations_data)
                allstations = allstations[~allstations['station_id'].isin(STATIONS_TO_EXCLUDE)]
                allstations.to_csv('stations_backup.csv')
                print("All stations retrieved:", allstations)
            else:
                print(f"Error fetching station data, status code {response.status}")
                return None

    # Check if a specific station or a list of stations is provided
    if input_station_id:
        # Check if input_station_id is a comma-separated string
        if isinstance(input_station_id, str) and ',' in input_station_id:
            input_str = input_station_id
            station_list = [s.strip() for s in input_str.split(",")]
            stations = allstations[allstations['station_id'].isin(station_list)]
            async with (await get_async_session()) as data_session:
                tasks = [one_day_measurements_async(data_session, st, input_date, days) for st in station_list]
                results = await asyncio.gather(*tasks)
                valid_results = [res for res in results if res is not None]
                if not valid_results:
                    return None
                all_results = pd.concat(valid_results, ignore_index=True)
        else:
            # Process as a single station if input_station_id is not comma separated.
            stations = allstations[allstations['station_id'] == input_station_id]
            async with (await get_async_session()) as data_session:
                all_results = await one_day_measurements_async(data_session, input_station_id, input_date, days)
    else:
        stations = allstations[~allstations['station_id'].isin(STATIONS_TO_EXCLUDE)]
        station_ids = stations['station_id'].values
        async with (await get_async_session()) as data_session:
            tasks = [one_day_measurements_async(data_session, st, input_date, days) for st in station_ids]
            results = await asyncio.gather(*tasks)
        valid_results = [res for res in results if res is not None]
        if not valid_results:
            return None
        all_results = pd.concat(valid_results, ignore_index=True)

    # Merge station info with API data
    daily_data = stations.merge(all_results, on='station_id', how='inner')

    # Compute risk metrics in parallel using ProcessPoolExecutor for CPU-bound tasks
    num_workers = min(len(daily_data), 8)  # Adjust based on CPU cores
    if num_workers == 0:
        return None

    chunks = chunk_dataframe(daily_data, num_workers)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_risks, chunk) for chunk in chunks]
        processed_chunks = [future.result() for future in as_completed(futures)]

    daily_data = pd.concat(processed_chunks, ignore_index=True)

    # Post-process date columns
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    daily_data['state'] = 'WI'
    daily_data['forecasting_date'] = (daily_data['date'] + timedelta(days=1)).dt.strftime('%Y-%m-%d')

    print("computed:", daily_data[['station_id', 'forecasting_date']])
    return daily_data[FINAL_COLUMNS]


# Entry point function for running the parallelized code
def main(input_date, input_station_id=None, days=1):
    """Run the async function using asyncio event loop"""
    return asyncio.run(retrieve_tarspot_all_stations_async(input_date, input_station_id, days))
