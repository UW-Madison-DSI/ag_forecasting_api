from zoneinfo import ZoneInfo
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone

from ag_models_wrappers.forecasting_models import *

# Define your URL base and any other constants
base_url = "https://wisconet.wisc.edu/api/v1"

stations_exclude = ['MITEST1','WNTEST1']
min_days_active = 38

# Map measures to corresponding columns using vectorized operations
measure_map = {
    4: 'air_temp_max_f',
    6: 'air_temp_min_f',
    20: 'rh_max',
    12: 'min_dp',
    56: 'max_ws'  # units in mph
}

def api_call_wisconet_data_daily(station_id, input_date):
    """
    Fetches and processes daily weather data for a given station and date range.

    Args:
        station_id (str): The unique ID of the station.
        input_date (str): The end date for the query in "YYYY-MM-DD" format.

    Returns:
        pd.DataFrame: A DataFrame with the latest weather data and 30-day moving averages.
    """

    # Define start and end date
    end_date = datetime.strptime(input_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=min_days_active)

    # Convert to Unix timestamps (seconds since epoch)
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    # Construct the API URL and request parameters
    url = f"{base_url}/stations/{station_id}/measures"
    params = {
        "start_time": start_timestamp,
        "end_time": end_timestamp,
        "fields": "daily_air_temp_f_avg,daily_air_temp_f_max,daily_air_temp_f_min,daily_dew_point_f_max,daily_dew_point_f_min,daily_relative_humidity_pct_max,daily_relative_humidity_pct_min,daily_wind_speed_mph_max,daily_dew_point_f_avg"
    }

    # Send the API request
    response = requests.get(url, params=params)

    # Check if the response is successful
    if response.status_code == 200:
        data = response.json().get("data", [])
        df = pd.DataFrame(data)

        # Prepare the result DataFrame with pre-allocated columns
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
        for measure_id, column_name in measure_map.items():
            # Find the rows where the measure_id matches and assign the value
            result_df[column_name] = df['measures'].apply(
                lambda measures: next((m[1] for m in measures if m[0] == measure_id), np.nan))

        # Convert Fahrenheit to Celsius
        result_df['min_dp_c'] = fahrenheit_to_celsius(result_df['min_dp'])
        result_df['air_temp_max_c'] = fahrenheit_to_celsius(result_df['air_temp_max_f'])
        result_df['air_temp_min_c'] = fahrenheit_to_celsius(result_df['air_temp_min_f'])
        result_df['air_temp_avg_c'] = fahrenheit_to_celsius(
            result_df[['air_temp_max_f', 'air_temp_min_f']].mean(axis=1))

        # Calculate 30-day moving averages using vectorized operations
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
        print(f"Error fetching data for station {station_id}, status code {response.status_code}")
        return None


def api_call_wisconet_data_rh(station_id, end_time):
    '''

    Args:
        station_id:
        end_time:

    Returns:

    '''
    try:
        # Set base URL and endpoint
        endpoint = f'/stations/{station_id}/measures'

        # Convert end_time string to datetime
        end_date = datetime.strptime(end_time, "%Y-%m-%d")
        start_date = end_date - timedelta(days=min_days_active)

        # Convert to Unix timestamps (seconds since epoch)
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        # Set parameters for the API request
        params = {
            "start_time": start_timestamp,
            "end_time": end_timestamp,
            "fields": "60min_relative_humidity_pct_avg"
        }

        # Make the GET request
        response = requests.get(f"{base_url}{endpoint}", params=params)
        scode = response.status_code

        # If the status code is 200, process the data
        if scode == 200:
            data = response.json().get("data", [])
            df = pd.DataFrame(data)
            # Create the result DataFrame
            # Extract RH values and process times
            result_df = pd.DataFrame({
                'collection_time': pd.to_datetime(df['collection_time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(
                    'US/Central'),
                'rh_avg': [
                    next((m[1] for m in item.measures if m[0] == 19), np.nan)  # Accessing measures attribute directly
                    for item in df.itertuples()
                ]
            })

            # Add new columns for processing the night RH >= 90 counts
            result_df['hour'] = result_df['collection_time'].dt.hour

            result_df['rh_night_above_90'] = np.where(
                (result_df['rh_avg'] >= 90) & ((result_df['hour'] >= 20) | (result_df['hour'] <= 6)),
                1, 0
            )

            # Add the new variable for RH >= 80 during the day (6 AM to 8 PM)
            result_df['rh_day_above_80'] = np.where(
                (result_df['rh_avg'] >= 80),
                1, 0
            )
            result_df['collection_time'] = pd.to_datetime(result_df['collection_time'])
            result_df['date'] = result_df['collection_time'].dt.strftime('%Y-%m-%d')

            # Group by adjusted date and sum RH above 90 counts for each day
            daily_rh_above_90 = result_df.groupby('date').agg(
                nhours_rh_above_90=('rh_night_above_90', 'sum'),
                hours_rh_above_80_day=('rh_day_above_80', 'sum')
            ).reset_index()

            daily_rh_above_90['rh_above_90_night_14d_ma'] = daily_rh_above_90['nhours_rh_above_90'].rolling(window=14,
                                                                                                           min_periods=14).mean()
            daily_rh_above_90['rh_above_80_day_30d_ma'] = daily_rh_above_90['hours_rh_above_80_day'].rolling(window=30,
                                                                                                             min_periods=30).mean()
            return daily_rh_above_90
        else:
            print(f"Error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Failed to retrieve or process data: {e}")
        return None

def one_day_measurements(station_id, end_time, days):
    '''
    Fetches and processes daily relative humidity and temperature data for a given station,
    and merges the latest data based on the specified number of days.

    Args:
        station_id (str): The station ID.
        end_time (str): The end time for fetching data, in YYYY-MM-DD format.
        days (int): The number of most recent days of data to fetch.

    Returns:
        pd.DataFrame: A merged DataFrame containing the processed data.
    '''
    try:
        if (days>7):
            days = 8
        if (days==None) or days==0:
            days = 1

        # Fetch the relative humidity data
        daily_rh_above_90 = api_call_wisconet_data_rh(station_id, end_time)
        if daily_rh_above_90 is None or daily_rh_above_90.empty:
            raise ValueError(f"No RH data found for station {station_id}.")

        # Sort by collection_time and get the most recent `days` rows
        daily_rh_above_90 = daily_rh_above_90.sort_values('date', ascending=False).head(days)
        # Process the RH data: Convert collection_time to datetime and extract date
        print("RH --------------")
        print(daily_rh_above_90)

        # Fetch the daily temperature data
        result_df = api_call_wisconet_data_daily(station_id, end_time)
        result_df = result_df.sort_values('date', ascending=False).head(days)
        if result_df is None or result_df.empty:
            raise ValueError(f"No daily data found for station {station_id}.")

        # Process the daily data: Convert collection_time to datetime and extract date

        print("AT --------------")
        print(result_df)
        # Merge the two DataFrames on 'date'
        combined_df = pd.merge(
            daily_rh_above_90[['date', 'rh_above_90_night_14d_ma', 'rh_above_80_day_30d_ma']],
            result_df[['date', 'air_temp_max_c_30d_ma', 'air_temp_min_c_21d_ma',
                       'air_temp_avg_c_30d_ma', 'rh_max_30d_ma', 'max_ws_30d_ma', 'dp_min_30d_c_ma']],
            on='date', how='inner'
        )

        # Add the station_id and sort by collection_time
        combined_df['station_id'] = station_id
        combined_df = combined_df.sort_values('date', ascending=False).head(days)

        return combined_df

    except Exception as e:
        print(f"Error in processing data for station {station_id}: {e}")
        return None


# Main function to retrieve and process data for all stations
def retrieve_tarspot_all_stations(input_date, input_station_id, days):
    '''

    Args:
        input_date:
        input_station_id:

    Returns:

    '''

    allstations_url = f"https://connect.doit.wisc.edu/pywisconet_wrapper/wisconet/active_stations/?min_days_active={min_days_active}&start_date={input_date}"
    response = requests.get(allstations_url)

    if response.status_code == 200:
        allstations = pd.DataFrame(response.json())
        print("all stations -----", allstations)

        # Filter stations if input_station_id is provided
        if input_station_id:
            stations = allstations[(allstations['station_id'] == input_station_id)
                        & (~allstations['station_id'].isin(stations_exclude))]
            all_results = one_day_measurements(input_station_id, input_date, days)

        else:
            stations = allstations[~allstations['station_id'].isin(stations_exclude)]
            st_res_list = []
            for st in list(stations['station_id'].values):
                daily_data = one_day_measurements(st, input_date, days)
                st_res_list.append(daily_data)

            all_results = pd.concat(st_res_list, ignore_index=True)

        daily_data = (stations
                  .merge(all_results, on='station_id', how='inner'))

        daily_data = daily_data.join(
            daily_data.apply(lambda row: calculate_tarspot_risk_function(
                row['air_temp_avg_c_30d_ma'], row['rh_max_30d_ma'], row['rh_above_90_night_14d_ma']), axis=1)
        )

        daily_data = daily_data.join(
            daily_data.apply(lambda row: calculate_gray_leaf_spot_risk_function(
                row['air_temp_min_c_21d_ma'], row['dp_min_30d_c_ma']), axis=1)
        )

        daily_data = daily_data.join(
            daily_data.apply(lambda row: calculate_frogeye_leaf_spot_function(
                row['air_temp_max_c_30d_ma'], row['rh_above_80_day_30d_ma']), axis=1)
        )

        daily_data = daily_data.join(
            daily_data.apply(lambda row: calculate_irrigated_risk(
                row['air_temp_max_c_30d_ma'], row['rh_max_30d_ma']), axis=1)
        )

        daily_data = daily_data.join(daily_data.apply(lambda row: calculate_non_irrigated_risk(
                row['air_temp_max_c_30d_ma'], row['max_ws_30d_ma']), axis=1)
        )
        return daily_data[['station_id','date', 'location',
                            'station_name', 'city', 'county', 'earliest_api_date', 'latitude',
                            'longitude', 'region', 'state',
                            'station_timezone',
                            'tarspot_risk',#'tarspot_risk_class',
                            'gls_risk',#'gls_risk_class',
                            'fe_risk',
                            'whitemold_irr_30in_risk',
                            'whitemold_irr_15in_risk',
                            'whitemold_nirr_risk']]
    else:
        print(f"Error fetching station data, status code {response.status_code}")
        return None