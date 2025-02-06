import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import requests
from typing import List, Dict, Optional
import concurrent.futures

from ag_models_wrappers.forecasting_models import *

base_url = "https://wisconet.wisc.edu/api/v1"

stations_to_exclude = ['MITEST1','WNTEST1']
min_days_active = 38


class WeatherDataFetcher:
    def __init__(self, base_url: str, min_days_active: int, stations_to_exclude: List[str]):
        self.base_url = base_url
        self.min_days_active = min_days_active
        self.stations_to_exclude = stations_to_exclude
        self.session = None
        self.measure_map = {
            4: 'air_temp_max_f',
            6: 'air_temp_min_f',
            20: 'rh_max',
            12: 'min_dp',
            56: 'max_ws'
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_data(self, url: str, params: Dict) -> Optional[Dict]:
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            return None

    async def fetch_daily_data(self, station_id: str, input_date: str) -> Optional[pd.DataFrame]:
        """
        Fetches and processes daily weather data for a given station and date range.
        """
        # Define start and end date
        end_date = datetime.strptime(input_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=self.min_days_active)

        # Convert to Unix timestamps
        params = {
            "start_time": int(start_date.timestamp()),
            "end_time": int(end_date.timestamp()),
            "fields": "daily_air_temp_f_avg,daily_air_temp_f_max,daily_air_temp_f_min,daily_dew_point_f_max,daily_dew_point_f_min,daily_relative_humidity_pct_max,daily_relative_humidity_pct_min,daily_wind_speed_mph_max,daily_dew_point_f_avg"
        }

        url = f"{self.base_url}/stations/{station_id}/measures"
        response_data = await self.fetch_data(url, params)

        if not response_data:
            return None

        df = pd.DataFrame(response_data.get("data", []))
        if df.empty:
            return None

        # Prepare result DataFrame
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
        for measure_id, column_name in self.measure_map.items():
            result_df[column_name] = df['measures'].apply(
                lambda measures: next((m[1] for m in measures if m[0] == measure_id), np.nan)
            )

        # Temperature conversions and calculations
        result_df['min_dp_c'] = self.fahrenheit_to_celsius(result_df['min_dp'])
        result_df['air_temp_max_c'] = self.fahrenheit_to_celsius(result_df['air_temp_max_f'])
        result_df['air_temp_min_c'] = self.fahrenheit_to_celsius(result_df['air_temp_min_f'])
        result_df['air_temp_avg_c'] = self.fahrenheit_to_celsius(
            result_df[['air_temp_max_f', 'air_temp_min_f']].mean(axis=1)
        )

        # Calculate moving averages
        windows = {
            'air_temp_min_c_21d_ma': ('air_temp_min_c', 21),
            'air_temp_max_c_30d_ma': ('air_temp_max_c', 30),
            'air_temp_avg_c_30d_ma': ('air_temp_avg_c', 30),
            'rh_max_30d_ma': ('rh_max', 30),
            'max_ws_30d_ma': ('max_ws', 30),
            'dp_min_30d_c_ma': ('min_dp_c', 30)
        }

        for col_name, (source_col, window) in windows.items():
            result_df[col_name] = result_df[source_col].rolling(
                window=window, min_periods=1
            ).mean()

        result_df['date'] = result_df['collection_time']#.dt.strftime('%Y-%m-%d')

        return result_df

    @staticmethod
    def fahrenheit_to_celsius(f):
        """Convert Fahrenheit to Celsius"""
        return (f - 32) * 5 / 9

    async def get_station_data(self, station_id: str, input_date: str, days: int) -> Optional[pd.DataFrame]:
        try:
            # Fetch RH and daily data concurrently
            rh_task = self.fetch_rh_data(station_id, input_date)
            daily_task = self.fetch_daily_data(station_id, input_date)

            daily_rh_above_90, result_df = await asyncio.gather(rh_task, daily_task)

            if daily_rh_above_90 is None or result_df is None:
                return None

            # Process data (existing logic, but vectorized)
            combined_df = self.process_station_data(daily_rh_above_90, result_df, station_id, days)
            return combined_df

        except Exception as e:
            print(f"Error processing station {station_id}: {e}")
            return None

    def process_station_data(self, daily_rh_above_90: pd.DataFrame, result_df: pd.DataFrame,
                             station_id: str, days: int) -> pd.DataFrame:
        """Process and combine RH and daily data for a station"""
        if days > 7:
            days = 8
        if days is None or days == 0:
            days = 1

        # Sort and get most recent days
        daily_rh_above_90 = daily_rh_above_90.sort_values('collection_time', ascending=False).head(days)
        daily_rh_above_90['date'] = daily_rh_above_90['collection_time']#.dt.strftime('%Y-%m-%d')
        print('---------------->>>',daily_rh_above_90)
        result_df = result_df.sort_values('date', ascending=False).head(days)
        result_df['date'] = result_df['collection_time']#.dt.strftime('%Y-%m-%d')

        # Merge the dataframes
        # Convert collection_time to datetime in both dataframes
        daily_rh_above_90['date'] = pd.to_datetime(daily_rh_above_90['collection_time'])
        result_df['date'] = pd.to_datetime(result_df['collection_time'])

        # Now perform the merge
        combined_df = pd.merge(
            daily_rh_above_90[['date','rh_above_90_night_14d_ma', 'rh_above_80_day_30d_ma']],
            result_df[['date','air_temp_max_c_30d_ma', 'air_temp_min_c_21d_ma',
                       'air_temp_avg_c_30d_ma', 'rh_max_30d_ma', 'max_ws_30d_ma', 'dp_min_30d_c_ma']],
            on='date', how='inner'
        )

        combined_df['station_id'] = station_id
        return combined_df.sort_values('date', ascending=False).head(days)

    async def fetch_rh_data(self, station_id: str, end_time: str) -> Optional[pd.DataFrame]:
        end_date = datetime.strptime(end_time, "%Y-%m-%d")
        start_date = end_date - timedelta(days=self.min_days_active)

        params = {
            "start_time": int(start_date.timestamp()),
            "end_time": int(end_date.timestamp()),
            "fields": "60min_relative_humidity_pct_avg"
        }

        url = f"{self.base_url}/stations/{station_id}/measures"
        data = await self.fetch_data(url, params)

        if not data:
            return None

        return self.process_rh_data(pd.DataFrame(data.get("data", [])))

    @staticmethod
    def process_rh_data(df: pd.DataFrame) -> pd.DataFrame:
        # Vectorized RH data processing
        df['collection_time'] = pd.to_datetime(df['collection_time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(
            'US/Central')
        df['rh_avg'] = df['measures'].apply(lambda x: next((m[1] for m in x if m[0] == 19), np.nan))
        df['hour'] = df['collection_time'].dt.hour

        # Vectorized conditions
        night_mask = (df['hour'] >= 20) | (df['hour'] <= 6)
        df['rh_night_above_90'] = ((df['rh_avg'] >= 90) & night_mask).astype(int)
        df['rh_day_above_80'] = (df['rh_avg'] >= 80).astype(int)

        # Group and calculate moving averages
        daily_data = df.groupby(df['collection_time'].dt.strftime('%Y-%m-%d')).agg({
            'rh_night_above_90': 'sum',
            'rh_day_above_80': 'sum'
        }).reset_index()

        daily_data['rh_above_90_night_14d_ma'] = daily_data['rh_night_above_90'].rolling(14, min_periods=14).mean()
        daily_data['rh_above_80_day_30d_ma'] = daily_data['rh_day_above_80'].rolling(30, min_periods=30).mean()

        return daily_data

    async def process_all_stations(self, input_date: str, input_station_id: Optional[str], days: int) -> pd.DataFrame:
        allstations_url = f"https://connect.doit.wisc.edu/pywisconet_wrapper/wisconet/active_stations/?min_days_active={self.min_days_active}&start_date={input_date}"

        async with self.session.get(allstations_url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch stations: {response.status}")

            allstations = pd.DataFrame(await response.json())

        if input_station_id:
            stations = allstations[
                (allstations['station_id'] == input_station_id) &
                (~allstations['station_id'].isin(self.stations_to_exclude))
                ]
            tasks = [self.get_station_data(input_station_id, input_date, days)]
        else:
            stations = allstations[~allstations['station_id'].isin(self.stations_to_exclude)]
            tasks = [self.get_station_data(st, input_date, days) for st in stations['station_id']]

        results = await asyncio.gather(*tasks)
        all_results = pd.concat([df for df in results if df is not None], ignore_index=True)

        return self.process_final_results(all_results, stations)

    @staticmethod
    def process_final_results(all_results: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
        # Merge and calculate risk factors using vectorized operations
        daily_data = stations.merge(all_results, on='station_id', how='inner')

        # Vectorized risk calculations
        daily_data = daily_data.assign(
            tarspot_risk=daily_data.apply(lambda row: calculate_tarspot_risk_function(
                row['air_temp_avg_c_30d_ma'], row['rh_max_30d_ma'], row['rh_above_90_night_14d_ma']), axis=1),
            gls_risk=daily_data.apply(lambda row: calculate_gray_leaf_spot_risk_function(
                row['air_temp_min_c_21d_ma'], row['dp_min_30d_c_ma']), axis=1),
            fe_risk=daily_data.apply(lambda row: calculate_frogeye_leaf_spot_function(
                row['air_temp_max_c_30d_ma'], row['rh_above_80_day_30d_ma']), axis=1),
            whitemold_irr_risk=daily_data.apply(lambda row: calculate_irrigated_risk(
                row['air_temp_max_c_30d_ma'], row['rh_max_30d_ma']), axis=1),
            whitemold_nirr_risk=daily_data.apply(lambda row: calculate_non_irrigated_risk(
                row['air_temp_max_c_30d_ma'], row['max_ws_30d_ma']), axis=1)
        )
        print(daily_data)
        #daily_data['date'] = pd.to_datetime(daily_data['date'])
        #daily_data['forecasting_date'] = (daily_data['date'] + timedelta(days=1)).dt.strftime('%Y-%m-%d')

        return daily_data[['station_id', #'date',
                           'forecasting_date', 'location',
                           'station_name', 'city', 'county', 'earliest_api_date', 'latitude',
                           'longitude', 'region', 'state', 'station_timezone',
                           'tarspot_risk', 'tarspot_risk_class',
                           'gls_risk', 'gls_risk_class',
                           'fe_risk', 'fe_risk_class',
                           'whitemold_irr_30in_risk',
                           'whitemold_irr_15in_risk',
                           'whitemold_nirr_risk']]


async def main(input_date: str, input_station_id: Optional[str] = None, days: int = 28):
    async with WeatherDataFetcher(base_url, min_days_active, stations_to_exclude) as fetcher:
        return await fetcher.process_all_stations(input_date, input_station_id, days)
