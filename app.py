from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from starlette.middleware.wsgi import WSGIMiddleware
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from pytz import timezone
from typing import List, Dict, Any
from pydantic import BaseModel
import numpy as np
import asyncio
import time
import os

from pywisconet.data import *
from pywisconet.process import *

from ag_models_wrappers.process_ibm_risk_v2 import *
from ag_models_wrappers.process_wisconet import *

app = FastAPI()


# ---------------------------------------------------------------------------
# Static field registry
# ---------------------------------------------------------------------------
# The legacy /fields/?legacy_only=True endpoint is no longer available.
# Fields are now fetched from /fields/{station_id}/available_fields per station,
# or can be seeded from the known global field definitions below.
# This list mirrors the full fields catalogue returned by the new endpoint.

_STATIC_FIELDS_JSON = [
    {"id": 1,  "standard_name": "5min_air_temp_f_avg",              "use_for": "",                        "measure_type": "Air Temp",          "qualifier": "avg",       "sensor": "", "source_field": "airtemp_c_avg@Table5",           "data_type": "float",   "source_units": "celsius",      "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "5min"},
    {"id": 2,  "standard_name": "60min_air_temp_f_avg",             "use_for": "",                        "measure_type": "Air Temp",          "qualifier": "avg",       "sensor": "", "source_field": "airtemp_c_avg@Table60",          "data_type": "float",   "source_units": "celsius",      "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "60min"},
    {"id": 3,  "standard_name": "daily_air_temp_f_avg",             "use_for": "",                        "measure_type": "Air Temp",          "qualifier": "avg",       "sensor": "", "source_field": "WN_calc",                        "data_type": "float",   "source_units": "celsius",      "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "daily"},
    {"id": 4,  "standard_name": "daily_air_temp_f_max",             "use_for": "",                        "measure_type": "Air Temp",          "qualifier": "max",       "sensor": "", "source_field": "airtemp_c_max@Table24",          "data_type": "float",   "source_units": "celsius",      "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "daily"},
    {"id": 6,  "standard_name": "daily_air_temp_f_min",             "use_for": "",                        "measure_type": "Air Temp",          "qualifier": "min",       "sensor": "", "source_field": "airtemp_c_min@Table24",          "data_type": "float",   "source_units": "celsius",      "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "daily"},
    {"id": 7,  "standard_name": "60min_battery_v_avg",              "use_for": "",                        "measure_type": "Battery",           "qualifier": "avg",       "sensor": "", "source_field": "battery_v_avg@Table60",          "data_type": "float",   "source_units": "volts",        "final_units": "volts",      "units_abbrev": "v",   "conversion_type": "",        "collection_frequency": "60min"},
    {"id": 9,  "standard_name": "5min_dew_point_f_avg",             "use_for": "",                        "measure_type": "Dew Point",         "qualifier": "avg",       "sensor": "", "source_field": "dewpointtemp_c_avg@Table5",      "data_type": "float",   "source_units": "celsius",      "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "5min"},
    {"id": 10, "standard_name": "60min_dew_point_f_avg",            "use_for": "",                        "measure_type": "Dew Point",         "qualifier": "avg",       "sensor": "", "source_field": "dewpointtemp_c_avg@Table60",     "data_type": "float",   "source_units": "celsius",      "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "60min"},
    {"id": 11, "standard_name": "daily_dew_point_f_max",            "use_for": "",                        "measure_type": "Dew Point",         "qualifier": "max",       "sensor": "", "source_field": "dewpointtemp_c_max@Table24",     "data_type": "float",   "source_units": "celsius",      "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "daily"},
    {"id": 12, "standard_name": "daily_dew_point_f_min",            "use_for": "",                        "measure_type": "Dew Point",         "qualifier": "min",       "sensor": "", "source_field": "dewpointtemp_c_min@Table24",     "data_type": "float",   "source_units": "celsius",      "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "daily"},
    {"id": 15, "standard_name": "daily_rain_in_tot",                "use_for": "",                        "measure_type": "Rain",              "qualifier": "total",     "sensor": "", "source_field": "rain_mm_tot@Table24",            "data_type": "float",   "source_units": "millimeters",  "final_units": "inches",     "units_abbrev": "in",  "conversion_type": "mm2in",   "collection_frequency": "daily"},
    {"id": 16, "standard_name": "5min_rain_in_tot",                 "use_for": "",                        "measure_type": "Rain",              "qualifier": "total",     "sensor": "", "source_field": "rain_mm_tot@Table5",             "data_type": "float",   "source_units": "millimeters",  "final_units": "inches",     "units_abbrev": "in",  "conversion_type": "mm2in",   "collection_frequency": "5min"},
    {"id": 17, "standard_name": "60min_rain_in_tot",                "use_for": "",                        "measure_type": "Rain",              "qualifier": "total",     "sensor": "", "source_field": "rain_mm_tot@Table60",            "data_type": "float",   "source_units": "millimeters",  "final_units": "inches",     "units_abbrev": "in",  "conversion_type": "mm2in",   "collection_frequency": "60min"},
    {"id": 18, "standard_name": "5min_relative_humidity_pct_avg",   "use_for": "",                        "measure_type": "Relative Humidity", "qualifier": "avg",       "sensor": "", "source_field": "relhum_avg@Table5",              "data_type": "float",   "source_units": "pct",          "final_units": "pct",        "units_abbrev": "pct", "conversion_type": "",        "collection_frequency": "5min"},
    {"id": 19, "standard_name": "60min_relative_humidity_pct_avg",  "use_for": "",                        "measure_type": "Relative Humidity", "qualifier": "avg",       "sensor": "", "source_field": "relhum_avg@Table60",             "data_type": "float",   "source_units": "pct",          "final_units": "pct",        "units_abbrev": "pct", "conversion_type": "",        "collection_frequency": "60min"},
    {"id": 20, "standard_name": "daily_relative_humidity_pct_max",  "use_for": "",                        "measure_type": "Relative Humidity", "qualifier": "max",       "sensor": "", "source_field": "relhum_max@Table24",             "data_type": "float",   "source_units": "pct",          "final_units": "pct",        "units_abbrev": "pct", "conversion_type": "",        "collection_frequency": "daily"},
    {"id": 21, "standard_name": "daily_relative_humidity_pct_min",  "use_for": "",                        "measure_type": "Relative Humidity", "qualifier": "min",       "sensor": "", "source_field": "relhum_min@Table24",             "data_type": "float",   "source_units": "pct",          "final_units": "pct",        "units_abbrev": "pct", "conversion_type": "",        "collection_frequency": "daily"},
    {"id": 55, "standard_name": "5min_wind_speed_mph_avg",          "use_for": "",                        "measure_type": "Wind Speed",        "qualifier": "avg",       "sensor": "", "source_field": "windspd_ms_3m_avg@Table5",       "data_type": "float",   "source_units": "meters/sec",   "final_units": "mph",        "units_abbrev": "mph", "conversion_type": "ms2mph",  "collection_frequency": "5min"},
    {"id": 56, "standard_name": "daily_wind_speed_mph_max",         "use_for": "",                        "measure_type": "Wind Speed",        "qualifier": "max",       "sensor": "", "source_field": "windspd_ms_3m_max@Table24",      "data_type": "float",   "source_units": "meters/sec",   "final_units": "mph",        "units_abbrev": "mph", "conversion_type": "ms2mph",  "collection_frequency": "daily"},
    {"id": 57, "standard_name": "60min_wind_speed_mph_max",         "use_for": "",                        "measure_type": "Wind Speed",        "qualifier": "max",       "sensor": "", "source_field": "windspd_ms_3m_max@Table60",      "data_type": "float",   "source_units": "meters/sec",   "final_units": "mph",        "units_abbrev": "mph", "conversion_type": "ms2mph",  "collection_frequency": "60min"},
    {"id": 60, "standard_name": "daily_air_temp_f_max_time",        "use_for": "",                        "measure_type": "Air Temp",          "qualifier": "max_time",  "sensor": "", "source_field": "airtemp_c_tmx@Table24",          "data_type": "integer", "source_units": "seconds",      "final_units": "seconds",    "units_abbrev": "f",   "conversion_type": "date2ts", "collection_frequency": "daily"},
    {"id": 61, "standard_name": "daily_air_temp_f_min_time",        "use_for": "",                        "measure_type": "Air Temp",          "qualifier": "min_time",  "sensor": "", "source_field": "airtemp_c_tmn@Table24",          "data_type": "integer", "source_units": "seconds",      "final_units": "seconds",    "units_abbrev": "f",   "conversion_type": "date2ts", "collection_frequency": "daily"},
    {"id": 62, "standard_name": "daily_dew_point_f_max_time",       "use_for": "",                        "measure_type": "Dew Point",         "qualifier": "max_time",  "sensor": "", "source_field": "dewpointtemp_c_tmx@Table24",     "data_type": "integer", "source_units": "seconds",      "final_units": "seconds",    "units_abbrev": "f",   "conversion_type": "date2ts", "collection_frequency": "daily"},
    {"id": 63, "standard_name": "daily_relative_humidity_min_time", "use_for": "",                        "measure_type": "Relative Humidity", "qualifier": "min_time",  "sensor": "", "source_field": "relhum_tmn@Table24",             "data_type": "integer", "source_units": "seconds",      "final_units": "seconds",    "units_abbrev": "",    "conversion_type": "date2ts", "collection_frequency": "daily"},
    {"id": 64, "standard_name": "daily_relative_humidity_max_time", "use_for": "",                        "measure_type": "Relative Humidity", "qualifier": "max_time",  "sensor": "", "source_field": "relhum_tmx@Table24",             "data_type": "integer", "source_units": "seconds",      "final_units": "seconds",    "units_abbrev": "",    "conversion_type": "date2ts", "collection_frequency": "daily"},
    {"id": 65, "standard_name": "60min_canopy_wetness_pct",         "use_for": "",                        "measure_type": "Canopy Wetness",    "qualifier": "pct",       "sensor": "", "source_field": "lw_canopy_mv_hst@Table60",       "data_type": "float",   "source_units": "pct",          "final_units": "pct",        "units_abbrev": "pct", "conversion_type": "lw2status","collection_frequency": "60min"},
    {"id": 87, "standard_name": "60min_wind_speed_mph_avg",         "use_for": "",                        "measure_type": "Wind Speed",        "qualifier": "avg",       "sensor": "", "source_field": "WN_calc",                        "data_type": "float",   "source_units": "meters/sec",   "final_units": "mph",        "units_abbrev": "mph", "conversion_type": "ms2mph",  "collection_frequency": "60min"},
]

# Build a reusable list of Field objects from the static registry.
# This replaces the old call to the deprecated /fields/?legacy_only=True endpoint.
ALL_FIELDS: list[Field] = [Field(**f) for f in _STATIC_FIELDS_JSON]


def get_station_fields(station_id: str) -> list[Field]:
    """
    Return available Field objects for a station.

    Strategy:
      1. Try the per-station endpoint /fields/{station_id}/available_fields
         (this is the current live endpoint).
      2. Fall back to the static ALL_FIELDS registry if the request fails
         (e.g. during tests or if the station is not yet indexed).
    """
    try:
        return station_fields(station_id)   # from pywisconet.data
    except Exception:
        return ALL_FIELDS


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get('/bulk_measures/{station_id}')
def bulk_measures_query(
    station_id: str,
    start_date: str = Query(..., description="Start date in format YYYY-MM-DD (e.g., 2024-07-01) assumed CT"),
    end_date: str = Query(..., description="End date in format YYYY-MM-DD (e.g., 2024-07-02) assumed CT"),
    measurements: str = Query(..., description="Measurements (e.g., AIRTEMP, DEW_POINT, WIND_SPEED, RELATIVE_HUMIDITY, ALL)"),
    frequency: str = Query(..., description="Frequency of measurements (e.g., MIN60, MIN5, DAILY)")
):
    """
    Query bulk measurements for a given station, date range, and measurement type.
    """
    cols = ['collection_time', 'collection_time_ct', 'hour_ct',
            'value', 'id', 'collection_frequency',
            'final_units', 'measure_type', 'qualifier', 'source_field',
            'standard_name', 'units_abbrev']

    if measurements is not None:
        # Fetch real Field objects — either from the live per-station endpoint
        # or from the static fallback registry.
        this_station_fields = get_station_fields(station_id)

        if measurements == 'ALL':
            filtered_field_standard_names = filter_fields(
                this_station_fields,
                criteria=[
                    MeasureType.RELATIVE_HUMIDITY,
                    MeasureType.AIRTEMP,
                    MeasureType.DEW_POINT,
                    MeasureType.WIND_SPEED,
                    CollectionFrequency[frequency]
                ]
            )
        elif measurements == 'RELATIVE_HUMIDITY':
            filtered_field_standard_names = filter_fields(
                this_station_fields,
                criteria=[MeasureType.RELATIVE_HUMIDITY, CollectionFrequency[frequency], Units.PCT]
            )
        elif measurements == 'AIRTEMP':
            filtered_field_standard_names = filter_fields(
                this_station_fields,
                criteria=[MeasureType.AIRTEMP, CollectionFrequency[frequency], Units.FAHRENHEIT]
            )
        elif measurements == 'DEW_POINT':
            filtered_field_standard_names = filter_fields(
                this_station_fields,
                criteria=[MeasureType.DEW_POINT, CollectionFrequency[frequency], Units.FAHRENHEIT]
            )
        elif measurements == 'WIND_SPEED':
            filtered_field_standard_names = filter_fields(
                this_station_fields,
                criteria=[MeasureType.WIND_SPEED, CollectionFrequency[frequency], Units.METERSPERSECOND]
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown measurement type: {measurements}")

        bulk_measure_response = bulk_measures(
            station_id,
            start_date,
            end_date,
            filtered_field_standard_names
        )
        df = bulk_measures_to_df(bulk_measure_response)
        df['collection_time_utc'] = pd.to_datetime(df['collection_time']).dt.tz_localize('UTC')
        df['collection_time_ct'] = df['collection_time_utc'].dt.tz_convert('US/Central')
        df['hour_ct'] = df['collection_time_ct'].dt.hour
        return df[cols].to_dict(orient="records")


@app.get("/wisconet/active_stations/")
def stations_query(
        min_days_active: int,
        start_date: str = Query(..., description="Start date in format YYYY-MM-DD (e.g., 2024-07-01)")
):
    try:
        start_date = datetime.strptime(start_date.strip(), "%Y-%m-%d").replace(tzinfo=ZoneInfo("UTC"))
        result = all_stations(min_days_active, start_date)
        if result is None:
            raise HTTPException(status_code=404, detail="Stations not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ag_models_wrappers/ibm")
def all_data_from_ibm_query(
        forecasting_date: str,
        latitude: float = Query(..., description="Latitude of the location"),
        longitude: float = Query(..., description="Longitude of the location"),
        API_KEY: str = Query(..., description="api key"),
        TENANT_ID: str = Query(..., description="Tenant id"),
        ORG_ID: str = Query(..., description="organization id")
):
    try:
        weather_data = get_weather(latitude, longitude, forecasting_date,
                                   ORG_ID, TENANT_ID, API_KEY)
        df = weather_data['daily']
        df_cleaned = df.replace([np.inf, -np.inf, np.nan], None).where(pd.notnull(df), None)
        return df_cleaned.to_dict(orient="records")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/ag_models_wrappers/wisconet")
def all_data_from_wisconet_query(
    forecasting_date: str,
    risk_days: int = 1,
    station_id: str = None
):
    try:
        df = retrieve(input_date=forecasting_date, input_station_id=station_id, days=risk_days)
        if df is None or len(df) == 0:
            return {}
        df_cleaned = df.replace([np.inf, -np.inf, np.nan], None).where(pd.notnull(df), None)
        return df_cleaned.to_dict(orient="records")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input in all_data_from_wisconet_query: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/ag_models_wrappers/wisconet_g")
def wisconet_geojson_grouped(
    forecasting_date: str,
    risk_days: int = 1,
    station_id: str = None
):
    try:
        df = retrieve(
            input_date=forecasting_date,
            input_station_id=station_id,
            days=risk_days
        )

        if df is None or len(df) == 0:
            return {"type": "FeatureCollection", "features": []}

        df = df.replace([np.inf, -np.inf, np.nan], None)

        features = []

        for station, group in df.groupby("station_id"):

            first_row = group.iloc[0]

            lat = first_row["latitude"]
            lon = first_row["longitude"]

            # Build time series list
            time_series = group.drop(
                columns=["latitude", "longitude"]
            ).to_dict(orient="records")

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "station_id": station,
                    "station_name": first_row["station_name"],
                    "city": first_row["city"],
                    "county": first_row["county"],
                    "region": first_row["region"],
                    "state": first_row["state"],
                    "time_series": time_series
                }
            }

            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Welcome to the Wisconsin Weather API"}

# Remove the WSGI code - it's not needed for FastAPI


# Create a WSGI application
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import ASGIApp

def create_wsgi_app():
    """
    Create a WSGI app to handle HTTP requests for the FastAPI application.
    """
    async def app(scope, receive, send):
        if scope["type"] == "http":
            await app(scope, receive, send)
        else:
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"text/plain")]
            })
            await send({
                "type": "http.response.body",
                "body": b"Not Found"
            })

    return app

wsgi_app = WSGIMiddleware(create_wsgi_app())
