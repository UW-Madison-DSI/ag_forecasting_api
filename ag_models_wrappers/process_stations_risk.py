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

def api_call_wisconet_data_daily(station_id, input_date):
    '''

    Args:
        station_id:
        input_date:

    Returns:

    '''
    # Define start and end date
    end_date = datetime.strptime(input_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=35)

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

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json().get("data", [])
        df = pd.DataFrame(data)

        # Prepare the result DataFrame
        result_df = pd.DataFrame({
            'o_collection_time': pd.to_datetime(df['collection_time'], unit='s'),# o_ because original measurement
            'collection_time': pd.to_datetime(df['collection_time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('US/Central'),
            'air_temp_max_f': np.nan,
            'air_temp_min_f': np.nan,
            'rh_max': np.nan,
            'min_dp': np.nan,
            'max_ws': np.nan
        })

        # Populate measures
        for i, row in df.iterrows():
            for measure in row['measures']:
                if measure[0] == 4:
                    result_df.at[i, 'air_temp_max_f'] = measure[1]
                elif measure[0] == 6:
                    result_df.at[i, 'air_temp_min_f'] = measure[1]
                elif measure[0] == 20:
                    result_df.at[i, 'rh_max'] = measure[1]
                elif measure[0] == 12:
                    result_df.at[i, 'min_dp'] = measure[1]
                elif measure[0] == 56:
                    result_df.at[i, 'max_ws'] = measure[1] #units mph

        # Convert Fahrenheit to Celsius
        result_df['min_dp_c'] = fahrenheit_to_celsius(result_df['min_dp'])
        result_df['air_temp_max_c'] = fahrenheit_to_celsius(result_df['air_temp_max_f'])
        result_df['air_temp_min_c'] = fahrenheit_to_celsius(result_df['air_temp_min_f'])
        result_df['air_temp_avg_c'] = fahrenheit_to_celsius(result_df[['air_temp_max_f', 'air_temp_min_f']].mean(axis=1))

        # Calculate 30-day moving averages
        result_df['air_temp_max_c_30d_ma'] = result_df['air_temp_max_c'].rolling(window=30, min_periods=1).mean()
        result_df['air_temp_min_c_21d_ma'] = result_df['air_temp_min_c'].rolling(window=21, min_periods=1).mean()
        result_df['air_temp_avg_c_30d_ma'] = result_df['air_temp_avg_c'].rolling(window=30, min_periods=1).mean()
        result_df['rh_max_30d_ma'] = result_df['rh_max'].rolling(window=30, min_periods=1).mean()
        result_df['max_ws_30d_ma'] = result_df['max_ws'].rolling(window=30, min_periods=1).mean()
        result_df['dp_min_30d_c_ma'] = result_df['min_dp_c'].rolling(window=30, min_periods=1).mean()

        # Return the closest row (latest data)
        result_df = result_df.sort_values('collection_time', ascending=False).head(1)
        result_df = result_df[['collection_time', 'air_temp_max_c_30d_ma', 'air_temp_min_c_21d_ma',
                               'air_temp_avg_c_30d_ma', 'rh_max_30d_ma', 'max_ws_30d_ma',
                               'dp_min_30d_c_ma']]
        result_df['station_id'] = station_id

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
        start_date = end_date - timedelta(days=31)

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
            result_df = pd.DataFrame({
                'o_collection_time': pd.to_datetime(df['collection_time'], unit='s'),
                'collection_time': pd.to_datetime(df['collection_time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('US/Central'),
                'rh_avg': np.nan,
            })
            # Extract RH values
            for i, item in df.iterrows():
                measures = item.get('measures', [])
                for measure in measures:
                    if measure[0] == 19:
                        result_df.at[i, 'rh_avg'] = measure[1]

            # Add new columns for processing the night RH >= 90 counts
            result_df['hour'] = result_df['collection_time'].dt.hour
            result_df['adjusted_date'] = np.where(
                (result_df['hour'] >= 0) & (result_df['hour'] <= 6),
                result_df['collection_time'] - pd.Timedelta(days=1),
                result_df['collection_time']
            )

            result_df['adjusted_date'] = result_df['adjusted_date'].dt.floor('D')

            result_df['rh_night_above_90'] = np.where(
                (result_df['rh_avg'] >= 90) & ((result_df['hour'] >= 20) | (result_df['hour'] <= 6)),
                1, 0
            )

            # Add the new variable for RH >= 80 during the day (6 AM to 8 PM)
            result_df['rh_day_above_80'] = np.where(
                (result_df['rh_avg'] >= 80),
                1, 0
            )

            # Group by adjusted date and sum RH above 90 counts for each day
            daily_rh_above_90 = result_df.groupby('adjusted_date').agg(
                nhours_rh_above_90=('rh_night_above_90', 'sum'),
                hours_rh_above_80_day=('rh_day_above_80', 'sum')
            ).reset_index()

            daily_rh_above_90['rh_above_90_night_14d_ma'] = daily_rh_above_90['nhours_rh_above_90'].rolling(window=14,
                                                                                                           min_periods=1).mean()
            daily_rh_above_90['rh_above_80_day_30d_ma'] = daily_rh_above_90['hours_rh_above_80_day'].rolling(window=30,
                                                                                                             min_periods=1).mean()

            daily_rh_above_90 = daily_rh_above_90.sort_values('adjusted_date', ascending=False).head(1)
            daily_rh_above_90['station_id'] = station_id

            return daily_rh_above_90[['adjusted_date','station_id', 'rh_above_90_night_14d_ma', 'rh_above_80_day_30d_ma']]
        else:
            print(f"Error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Failed to retrieve or process data: {e}")
        return None

# Main function to retrieve and process data for all stations
def retrieve_tarspot_all_stations(input_date, input_station_id):
    '''

    Args:
        input_date:
        input_station_id:

    Returns:

    '''
    allstations_url = f"https://connect.doit.wisc.edu/pywisconet_wrapper/all_stations/31?start_date={input_date}"
    response = requests.get(allstations_url)

    if response.status_code == 200:
        allstations = pd.DataFrame(response.json())
        allstations = allstations[~allstations['station_id'].isin(stations_exclude)]

        # Filter stations if input_station_id is provided
        if input_station_id:
            stations = allstations[allstations['station_id'] == input_station_id]
            daily_data = api_call_wisconet_data_daily(stations['station_id'].iloc[0], input_date)
            rh=api_call_wisconet_data_rh(stations['station_id'].iloc[0], input_date)
            result = pd.merge(stations, daily_data, on='station_id', how='left')
            result = pd.merge(result, rh, on='station_id', how='left')

        else:
            stations = allstations
            st_res_list = []
            st_rh_list = []
            for st in list(stations['station_id'].values):
                st_res = api_call_wisconet_data_daily(st, input_date)
                rh = api_call_wisconet_data_rh(st, input_date)
                #if st_res is not None:
                st_res['station_id'] = st  # Add station_id to each result
                rh['station_id'] = st  # Add station_id to each result
                st_res_list.append(st_res)
                st_rh_list.append(rh)

            all_results = pd.concat(st_res_list, ignore_index=True)
            rh_all_results = pd.concat(st_rh_list, ignore_index=True)
            result = pd.merge(stations, all_results, on='station_id', how='left')
            result = pd.merge(result, rh_all_results, on='station_id', how='left')

        daily_data = result
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

        daily_data = daily_data.join(result.apply(lambda row: calculate_non_irrigated_risk(
                row['air_temp_max_c_30d_ma'], row['max_ws_30d_ma']), axis=1)
        )
        return daily_data[['station_id','collection_time', 'location',
                            'station_name', 'city', 'county', 'earliest_api_date', 'latitude',
                            'longitude', 'region', 'state',
                            'station_timezone', 'tarspot_risk',
                            'tarspot_risk_class', 'gls_risk',
                            'gls_risk_class', 'fe_risk', 'fe_risk_class',
                            'whitemold_irr_30in_risk', 'whitemold_irr_15in_risk',
                            'whitemold_nirr_risk']]
    else:
        print(f"Error fetching station data, status code {response.status_code}")
        return None