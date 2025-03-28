import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone
import os

from datetime import datetime, timedelta, time

from ag_models_wrappers.forecasting_models import (
    rolling_mean,
    calculate_tarspot_risk_function,
    calculate_gray_leaf_spot_risk_function,
    calculate_frogeye_leaf_spot_function,
    calculate_irrigated_risk,
    calculate_non_irrigated_risk
)


URL_saascore = os.getenv("URL_saascore")

def kmh_to_mps(speed_kmh):
    """Convert speed from km/h to m/s."""
    return speed_kmh / 3.6

def generate_chunks(start_date, end_date):
    chunks = []

    # Parse the input dates using fromisoformat (works for date-only strings)
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    # Ensure the time is set to midnight (00:00)
    start = datetime.combine(start.date(), time(0, 1))
    end = datetime.combine(end.date(), time(0, 1))

    while start < end:
        next_start = start + timedelta(hours=999)
        chunk_end = min(next_start, end)
        # Format dates with only the hour using strftime
        chunks.append((start.strftime("%Y-%m-%dT%H"), chunk_end.strftime("%Y-%m-%dT%H")))
        start = next_start

    return chunks

def retrieve_saascore_api_key(org_id, tenant_id, current_api_key):
  """
  Makes a GET request to the IBM SaaSCore authentication retrieve API key endpoint.

  Args:
    org_id: Your EIS organization ID.
    tenant_id: Your EIS tenant ID.
    current_api_key: Your existing EIS API key.

  Returns:
    The JSON response from the API if the request is successful (status code 200),
    otherwise None. You should check the response for the retrieved API key.
    Prints an error message if the request fails.
  """
  url = f'{URL_saascore}/api-key?orgId={org_id}'
  headers = {
      'x-ibm-client-Id': f'saascore-{tenant_id}',
      'x-api-key': current_api_key
  }

  try:
      response = requests.get(url, headers=headers)
      response.raise_for_status()  # Raise an exception for bad status codes
      return response.text
  except requests.exceptions.RequestException as e:
      print(f"Error during API call: {e}")
      return None

def get_ibm_weather(lat, lng, start_date, end_date,
                    ORG_ID, TENANT_ID, API_KEY):
    '''

    Args:
        lat:
        lng:
        start_date:
        end_date:

    Returns:

    '''
    #org_id, tenant_id, current_api_key
    jwt = retrieve_saascore_api_key(ORG_ID, TENANT_ID, API_KEY)

    url = 'https://api.ibm.com/geospatial/run/v3/wx/hod/r1/direct'
    headers = {
        'x-ibm-client-id': f'geospatial-{TENANT_ID}',
        'Authorization': f'Bearer {jwt}'
    }
    try:
        chunks = generate_chunks(start_date, end_date)
        all_data = []

        for start, end in chunks:
            params = {
                "format": "json",
                "geocode": f"{lat},{lng}",
                "startDateTime": start,
                "endDateTime": end,
                "units": "m"
            }
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                all_data.append(pd.DataFrame(data))
            else:
                print(f"---- Failed to fetch data for {start} to {end}")

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    except Exception as e:
        print('-------- Exception ',e)
        return None

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
    daily['windSpeed_mean']=daily['windSpeed_mean'].apply(lambda x: kmh_to_mps(x))
    daily['windSpeed_max'] = daily['windSpeed_max'].apply(lambda x: kmh_to_mps(x))
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


def get_weather(lat, lng, end_date, API_KEY, TENANT_ID, ORG_ID):
    '''

    Args:
        lat:
        lng:
        end_date:
        API_KEY:
        TENANT_ID:
        ORG_ID:

    Returns:

    '''
    try:
        tz = 'US/Central'
        # Convert the string to a datetime object
        date_obj = datetime.strptime(end_date, "%Y-%m-%d")

        # Subtract 31 days
        date_31_days_before = date_obj - timedelta(days=36)

        # Convert back to string if needed
        start_date = date_31_days_before.strftime("%Y-%m-%d")
        hourly_data = get_ibm_weather(lat, lng, str(start_date), str(date_obj),
                                      API_KEY, TENANT_ID, ORG_ID)
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
                row['temperature_max_30ma'], row['relativeHumidity_max_30ma'], row['windSpeed_max_30ma']), axis=1)
        )
        return {"hourly": hourly, "daily": daily_data}
    except Exception as e:
        return {"hourly": None, "daily": None}
