import sys
import os

# Add the app directory to the Python path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'pywisconet')))

from utils import set_page_title, local_css, remote_css
from app.stations import stationslist
from app.functions_map import *

from datetime import timedelta
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium
from pywisconet.variables import CollectionFrequency, MeasureType, Units
from pywisconet.data import (
    bulk_measures,
    station_fields,
)
from pywisconet.process import filter_fields, bulk_measures_to_df
import requests
from geopy.distance import geodesic

import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import zoneinfo
import json


set_page_title('Forecasting Tool')
local_css("app/frontend/style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')


# Define cache file path and expiry duration
CACHE_FILE = 'cache.json'
CACHE_EXPIRY = timedelta(days=1)  # Cache duration


def retrieve_frompywisconet(station_code: str, this_station: str) -> bool:
    """Fetches and processes bulk measure data for the specified station with caching.

    Args:
        station_code (str): ID of the weather station.

    Returns:
        pd.DataFrame: DataFrame containing the processed weather data.
    """
    def fetch_data_from_api():
        """Fetches data from the API and returns it as a DataFrame."""
        # Get current UTC time
        today_utc = datetime.now(tz=zoneinfo.ZoneInfo("UTC"))
        today = today_utc.astimezone(zoneinfo.ZoneInfo("UTC"))

        # Calculate 30 days ago
        thirty_days_ago = today - timedelta(days=60)

        # Get station fields
        this_station_fields = station_fields(station_code)

        # Format the dates as strings
        start_date_str = thirty_days_ago.strftime("%Y-%m-%d")
        end_date_str = today.strftime("%Y-%m-%d")

        # Split the date strings into components and convert to datetime objects
        start_year, start_month, start_day = map(int, start_date_str.split('-'))
        end_year, end_month, end_day = map(int, end_date_str.split('-'))

        start_date = datetime(start_year, start_month, start_day, tzinfo=zoneinfo.ZoneInfo("UTC"))
        end_date = datetime(end_year, end_month, end_day, tzinfo=zoneinfo.ZoneInfo("UTC"))

        # Filter field standard names
        filtered_field_standard_names = filter_fields(
            this_station_fields,
            criteria=[
                MeasureType.AIRTEMP,
                MeasureType.DEW_POINT,
                CollectionFrequency.MIN60,
                Units.FAHRENHEIT
            ]
        )

        # Fetch bulk measure data
        bulk_measure_response = bulk_measures(
            station_id=station_code,
            start_time=start_date,
            end_time=end_date,
            fields=filtered_field_standard_names
        )

        # Convert response to DataFrame
        df = bulk_measures_to_df(bulk_measure_response)
        return df, filtered_field_standard_names

    def get_cached_data():
        """Fetches cached data if available and valid."""
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as file:
                cache = json.load(file)
                timestamp = datetime.fromisoformat(cache['timestamp'])
                if datetime.now() - timestamp < CACHE_EXPIRY:
                    return pd.read_json(cache['data'])

        # If no valid cache, fetch new data
        return None

    def cache_data(df):
        """Caches the DataFrame."""
        cache = {
            'timestamp': datetime.now().isoformat(),
            'data': df.to_json()
        }
        with open(CACHE_FILE, 'w') as file:
            json.dump(cache, file)

    # Try to get cached data
    df = get_cached_data()
    if df is None:
        df, filtered_field_standard_names = fetch_data_from_api()

        cache_data(df)

    df['datetime'] = pd.to_datetime(df['collection_time'])

    sns.set_theme(style="darkgrid")
    st.write(f"### Station Weather Trend {this_station}")

    # Plotting
    sns.lineplot(x="datetime", y="value", hue="measure_type", data=df)
    plt.xticks(rotation=15)

    # Display the plot in Streamlit
    st.pyplot(plt)

    df['parameter'] = df['measure_type']
    df['timestamp'] = df['collection_time']
    summary_df = summarize_data(df)
    st.dataframe(summary_df)

def get_key_by_name(stationslist, name):
    """Finds the key of a dictionary entry given the name.

    Args:
        stationslist (dict): The dictionary containing station data.
        name (str): The name of the station.

    Returns:
        str: The key of the dictionary entry matching the name, or None if not found.
    """
    for key, value in stationslist.items():
        if value['name'] == name:
            return key
    return None


def summarize_data(df):
    """

    :param df:
    :return:
    """
    #st.write(df)
    summary = []
    for param in df['parameter'].unique():
        param_data = df[df['parameter'] == param]
        units = param_data['final_units'].values[0]
        min_value = param_data['value'].min()
        max_value = param_data['value'].max()
        min_date = param_data[param_data['value'] == min_value]['timestamp'].dt.strftime('%b %d').values[0]
        max_date = param_data[param_data['value'] == max_value]['timestamp'].dt.strftime('%b %d').values[0]
        summary.append({
            'Parameter': param,
            'Units': units,
            'Observations': len(param_data),
            'Min': f"{min_value:.1f}",
            'Max': f"{max_value:.1f}",
            'Date of Min': min_date,
            'Date of Max': max_date
        })
    return pd.DataFrame(summary)


def main():
    """

    :return:
    """
    # stations
    data = pd.DataFrame.from_dict(stationslist, orient='index')
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'station_id'}, inplace=True)

    # Initialize session state variables
    if 'stationcode_nearest' not in st.session_state:
        st.session_state.stationcode_nearest = None
        st.session_state.nearest_station_name = None

    # Button for finding the nearest weather station
    nearest_station=None
    nearest_activation = st.sidebar.button("My nearest Wisconet Weather Station", key="nearest_button")
    if nearest_activation:
        # Get user location
        user_location = get_ip_location()

        # Find the nearest station
        nearest_station, stationcode_nearest = find_nearest_station(user_location)
        st.session_state.stationcode_nearest = stationcode_nearest
        st.session_state.nearest_station_name = nearest_station['name']

        if nearest_station is not None:
            pass
        else:
            st.write("No stations found.")

    station_name = st.session_state.get('nearest_station_name', 'No station selected')
    #st.write(f"Current Station: {station_name}")
    if station_name == 'No station selected' or not station_name:
        stations = list(stationslist.keys())
        stationslist2 = [stationslist[k]['name'] for k in stations]
        stationslist2.sort()

        station_name = st.sidebar.selectbox("Select Station", stationslist2)

    station_id = get_key_by_name(stationslist, station_name)

    map_creation1(stationslist, station_id)
    if st.session_state.nearest_station_name is not None:
        st.write(
            f"Nearest station: **{st.session_state.nearest_station_name}**")

    if st.session_state.stationcode_nearest is None:
        retrieve_frompywisconet(station_id, station_name)
    else:
        retrieve_frompywisconet(st.session_state.stationcode_nearest, station_name)

main()