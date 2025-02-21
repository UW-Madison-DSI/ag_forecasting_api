import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone

# Set up your IBM API key
import os
from ag_models_wrappers.forecasting_models import *


API_KEY = os.getenv("API_KEY")

def ibm_chunks(start_date, end_date):
    '''

    Args:
        start_date:
        end_date:

    Returns:

    '''
    chunks = []
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    while start < end:
        next_start = start + timedelta(hours=999)
        chunks.append((start.isoformat(), min(next_start, end).isoformat()))
        start = next_start
    return chunks


def get_ibm_weather(lat, lng, start_date, end_date):
    '''

    Args:
        lat:
        lng:
        start_date:
        end_date:

    Returns:

    '''
    chunks = ibm_chunks(start_date, end_date)
    all_data = []

    for start, end in chunks:
        url = "https://api.weather.com/v3/wx/hod/r1/direct"
        params = {
            "format": "json",
            "geocode": f"{lat},{lng}",
            "startDateTime": start,
            "endDateTime": end,
            "units": "m",
            "apiKey": API_KEY
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            all_data.append(pd.DataFrame(data))
        else:
            print(f"Failed to fetch data for {start} to {end}")

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def build_hourly(data, tz='US/Central'):
    '''

    Args:
        data:
        tz:

    Returns:

    '''
    # Convert validTimeUtc to datetime with UTC timezone
    data['dttm_utc'] = pd.to_datetime(data['validTimeUtc'], utc=True)

    # Convert to Central Time (America/Chicago)
    data['dttm'] = data['dttm_utc'].dt.tz_convert('US/Central')

    # Extract additional fields
    data['date'] = data['dttm'].dt.date
    data['hour'] = data['dttm'].dt.hour
    data['night'] = ~data['hour'].between(7, 19)  # Define night hours as between 8 PM and 6 AM

    data['date_since_night'] = (
                data['dttm'] + pd.to_timedelta(4, unit='h')).dt.date  # Shift 4 hours for night calculations

    return data.sort_values('dttm_utc')


def build_daily(hourly):
    '''

    Args:
        hourly:

    Returns:

    '''
    daily = hourly.groupby('date').agg({
        'temperature': ['min', 'mean', 'max'],
        'temperatureDewPoint': ['min', 'mean', 'max'],
        'relativeHumidity': ['min', 'mean', 'max'],
        'precip1Hour': 'sum',
        'windSpeed': ['mean', 'max']
    })

    daily.columns = ['_'.join(col) for col in daily.columns]
    daily.reset_index(inplace=True)

    night_rh = hourly[hourly['night']].groupby('date_since_night').agg(
        hours_rh90_night=('relativeHumidity', lambda x: sum(x >= 90))
    ).reset_index()

    allday_rh = hourly.groupby('date').agg(
        hours_rh80_allday=('relativeHumidity', lambda x: sum(x >= 80))
    ).reset_index()

    ds1 = pd.merge(daily, night_rh, left_on='date', right_on='date_since_night', how='left')
    return pd.merge(ds1, allday_rh, left_on='date', right_on='date', how='left')


def add_moving_averages(data):
    '''

    Args:
        data:

    Returns:

    '''
    for col in ['temperature_max', 'temperature_mean',
                'temperatureDewPoint_min',
                'relativeHumidity_max', 'windSpeed_max',
                'hours_rh90_night', 'hours_rh80_allday']:
        if col in ['hours_rh90_night']:
            data[f'{col}_14ma'] = rolling_mean(data[col], 14)
        else:
            data[f'{col}_30ma'] = rolling_mean(data[col], 30)

    data[f'temperature_min_21ma'] = rolling_mean(data['temperature_min'], 21)

    return data


def get_weather(lat, lng, end_date):
    '''

    Args:
        lat:
        lng:
        end_date:

    Returns:

    '''
    try:
        tz = 'US/Central'
        print(f"Fetching weather for point {lat}, {lng} ({tz})")
        # Convert the string to a datetime object
        date_obj = datetime.strptime(end_date, "%Y-%m-%d")

        # Subtract 31 days
        date_31_days_before = date_obj - timedelta(days=36)

        # Convert back to string if needed
        start_date = date_31_days_before.strftime("%Y-%m-%d")
        print("End date ", end_date, " start date ", start_date)
        hourly_data = get_ibm_weather(lat, lng, str(start_date), str(date_obj))
        if hourly_data.empty:
            print("No data returned from API.")
            return None

        hourly = build_hourly(hourly_data, "US/Central")
        daily = build_daily(hourly)
        daily_data = add_moving_averages(daily)
        daily_data['forecasting_date'] = daily_data['date'].apply(lambda x: x + timedelta(days=1))

        daily_data = daily_data.join(
            daily_data.apply(lambda row: calculate_tarspot_risk_function(
                row['temperature_mean_30ma'], row['relativeHumidity_max_30ma'], row['hours_rh90_night_14ma']), axis=1)
        )

        daily_data = daily_data.join(
            daily_data.apply(lambda row: calculate_gray_leaf_spot_risk_function(
                row['temperature_min_21ma'], row['temperatureDewPoint_min_30ma']), axis=1)
        )

        daily_data = daily_data.join(
            daily_data.apply(lambda row: calculate_frogeye_leaf_spot_function(
                row['temperature_max_30ma'], row['hours_rh80_allday_30ma']), axis=1)
        )

        daily_data = daily_data.join(
            daily_data.apply(lambda row: calculate_irrigated_risk(
                row['temperature_max_30ma'], row['relativeHumidity_max_30ma']), axis=1)
        )

        daily_data = daily_data.join(
            daily_data.apply(lambda row: calculate_non_irrigated_risk(
                row['temperature_max_30ma'], row['windSpeed_max_30ma']), axis=1)
        )

        return {"hourly": hourly, "daily": daily_data}
    except Exception as e:
        print("Error --------", e)
        print("The input was ", lat, lng, end_date)
        return {"hourly": None, "daily": None}
