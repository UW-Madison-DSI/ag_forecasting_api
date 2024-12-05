from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from pywisconet.data import *
from pywisconet.process import *
from ibm_wrapper.process import *
from starlette.middleware.wsgi import WSGIMiddleware
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from pytz import timezone
from typing import Dict, Any  # Import Any here

app = FastAPI()

# Testing station_fields
@app.get("/station_fields/{station_id}")
def station_fields_query(station_id: str):
    try:
        result = station_fields(station_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Station {station_id} not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/all_stations/{min_days_active}")
def stations_query(
        min_days_active: int,
        start_date: str = Query(..., description="Start date in format YYYY-MM-DD (e.g., 2024-07-01)")
):
    try:
        start_date = datetime.strptime(start_date.strip(), "%Y-%m-%d").replace(tzinfo=ZoneInfo("UTC"))
        result = all_stations(min_days_active, start_date)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Stations not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/bulk_measures/{station_id}')
def bulk_measures_query(
    station_id: str,
    start_date: str = Query(..., description="Start date in format YYYY-MM-DD (e.g., 2024-07-01) assumed CT"),
    end_date: str = Query(..., description="End date in format YYYY-MM-DD (e.g., 2024-07-02) assumed CT"),
    measurements: str = Query(..., description="Measurements (e.g., AIRTEMP, DEW_POINT, WIND_SPEED, RELATIVE_HUMIDITY, ALL the units are F, M/S and % respectively, all means the last 4)"),
    #measurements_list = None, #str = Query(..., description="List of measurements eg '60min_air_temp_f_avg', '60min_dew_point_f_avg', '60min_relative_humidity_pct_avg', '60min_wind_speed_mph_max', '60min_wind_speed_mph_avg', '60min_wind_speed_max_time'"),
    #units: str = Query(..., description="Units for measurements (e.g., FAHRENHEIT, PCT, METERSPERSECOND, MPH)"),
    frequency: str = Query(..., description="Frequency of measurements (e.g., MIN60, MIN5, DAILY)")
):

    cols = ['collection_time', 'collection_time_ct', 'hour_ct',
            'value', 'id', 'collection_frequency',
            'final_units', 'measure_type','qualifier', #'source_field',
            'standard_name', 'units_abbrev']

    print("Dates:", start_date, end_date)
    if measurements is not None:
        # Retrieve fields for the station
        this_station_fields = station_fields(station_id)
        if measurements=='ALL':
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
        if measurements =='RELATIVE_HUMIDITY':
            filtered_field_standard_names = filter_fields(
                this_station_fields,
                criteria=[
                    MeasureType.RELATIVE_HUMIDITY,
                    CollectionFrequency[frequency],
                    Units.PCT
                ]
            )
        elif measurements =='AIRTEMP':
            print('this_station_fields ', this_station_fields)
            filtered_field_standard_names = filter_fields(
                this_station_fields,
                criteria=[
                    MeasureType.AIRTEMP,
                    CollectionFrequency[frequency],
                    Units.FAHRENHEIT
                ]
            )
        elif measurements == 'DEW_POINT':
            filtered_field_standard_names = filter_fields(
                this_station_fields,
                criteria=[
                    MeasureType.DEW_POINT,
                    CollectionFrequency[frequency],
                    Units.FAHRENHEIT
                ]
            )
        elif measurements== 'WIND_SPEED':
            filtered_field_standard_names = filter_fields(
                this_station_fields,
                criteria=[
                    MeasureType.WIND_SPEED,
                    CollectionFrequency[frequency],
                    Units.METERSPERSECOND
                ]
            )

        # Fetch data for the date range
        bulk_measure_response = bulk_measures(
            station_id,
            start_date,
            end_date,
            filtered_field_standard_names
        )

        df = bulk_measures_to_df(bulk_measure_response)
        df['collection_time_utc'] = pd.to_datetime(df['collection_time']).dt.tz_localize('UTC')
        ## Fix CT because all the Wisconet Stations are in CT
        df['collection_time_ct']=df['collection_time_utc'].dt.tz_convert('US/Central')
        df['hour_ct'] = df['collection_time_ct'].dt.hour

        print(df[cols])
        # Return data as a dictionary with local times
        return df[cols].to_dict(orient="records")

@app.get('/moving_averages/{station_id}')
def moving_averages_query(
    station_id: str,
    end_date: str = Query(..., description="End date in format YYYY-MM-DD (e.g., 2024-07-02) assumed CT"),
    disease_name: str = Query(..., description="Disease name (e.g., tarspot, gls, frogeye_ls, sporecaster")
):
    cols = ['collection_time', 'collection_time_ct', 'hour_ct',
            'value', 'id', 'collection_frequency',
            'final_units', 'measure_type','qualifier', #'source_field',
            'standard_name', 'units_abbrev']
    start_date = end_date - days(31)

    print("Dates:", end_date, " start_date ", start_date)

    this_station_fields = station_fields(station_id)

    if disease_name =='tarspot':
        # Retrieve fields for the station
        filtered_field_standard_names = filter_fields(
            this_station_fields,
            criteria=[
                MeasureType.RELATIVE_HUMIDITY,
                MeasureType.AIRTEMP,
                CollectionFrequency[frequency]
                ]
            )
    elif disease_name == 'gls':
        filtered_field_standard_names = filter_fields(
            this_station_fields,
            criteria=[
                MeasureType.RELATIVE_HUMIDITY,
                MeasureType.AIRTEMP,
                CollectionFrequency[frequency]
            ]
        )

    elif disease_name == 'frogeye_ls':
        filtered_field_standard_names = filter_fields(
            this_station_fields,
            criteria=[
                MeasureType.RELATIVE_HUMIDITY,
                MeasureType.AIRTEMP,
                CollectionFrequency[frequency]
            ]
        )

    elif disease_name == 'sporecaster':
        filtered_field_standard_names = filter_fields(
            this_station_fields,
            criteria=[
                MeasureType.RELATIVE_HUMIDITY,
                MeasureType.AIRTEMP,
                CollectionFrequency[frequency]
            ]
        )

    # Fetch data for the date range
    bulk_measure_response = bulk_measures(
            station_id,
            start_date,
            end_date,
            filtered_field_standard_names
        )

    df = bulk_measures_to_df(bulk_measure_response)
    df['collection_time_utc'] = pd.to_datetime(df['collection_time']).dt.tz_localize('UTC')
    ## Fix CT because all the Wisconet Stations are in CT
    df['collection_time_ct']=df['collection_time_utc'].dt.tz_convert('US/Central')
    df['hour_ct'] = df['collection_time_ct'].dt.hour

    print(df[cols])
    # Return data as a dictionary with local times
    return df[cols].to_dict(orient="records")

@app.get("/ibm_wrapper/{forecasting_date}")
def all_data_from_ibm_query(
    forecasting_date: str,  # Passed as part of the URL path
    latitude: float = Query(..., description="Latitude of the location"),
    longitude: float = Query(..., description="Longitude of the location"),
):
    """
    Query weather data using the IBM Weather API.

    Args:
        forecasting_date (str): Start date for the forecast (YYYY-MM-DD).
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.

    Returns:
        dict: Weather data including hourly and daily summaries.
    """
    try:
        # Fetch weather data
        weather_data = get_weather(latitude, longitude, forecasting_date)
        df=weather_data['daily']
        df_cleaned = df.replace([np.inf, -np.inf, np.nan], None).where(pd.notnull(df), None)

        df_dict = df_cleaned.to_dict(orient="records")

        return df_dict

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

#Check this one
#@app.get("/all_data_for_station/{station_id}")
#def all_data_for_station_query(
#    station_id: str
#):
#    try:
#        stations = all_stations(n_days_active=1)
#        this_station = [i for i in stations if i['station_id']==station_id][0]
#        print("-----------------------------, This station ", this_station)
#        this_station_fields = station_fields(station_id)
#        filtered_field_standard_names = filter_fields(
#            this_station_fields,
#            criteria=[
#                MeasureType.AIRTEMP,
#                MeasureType.DEW_POINT,
#                CollectionFrequency.MIN60,
#                Units.FAHRENHEIT
#            ]
#        )
        #print("filtered station fields ", filtered_field_standard_names)
#        all_station_field_data_bms = all_data_for_station(this_station,
#                                        filtered_field_standard_names)
#        print("all_station_field_data_bms ",
#              all_station_field_data_bms)
#        all_station_field_data_df = bulk_measures_to_df(all_station_field_data_bms)

#        if all_station_field_data_df is None:
#            raise HTTPException(status_code=404, detail=f"Station {this_station.station_id} not found")
#        return all_station_field_data_df
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Wisconsin Weather API"}

# Create a WSGI application
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import ASGIApp

def create_wsgi_app():
    async def app(scope, receive, send):
        if scope["type"] == "http":
            # Delegate to FastAPI for HTTP requests
            await app(scope, receive, send)
        else:
            # Handle other types of requests
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