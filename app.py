from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from pywisconet.data import *
from pywisconet.process import *
from starlette.middleware.wsgi import WSGIMiddleware
from datetime import datetime
from zoneinfo import ZoneInfo

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
        min_days_active: int
):
    try:
        result = all_stations(min_days_active)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Stations not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/bulk_measures/{station_id}')
def bulk_measures_query(
        station_id: str,
        start_date: str = Query(..., description="Start date in format YYYY-MM-DD (e.g., 2024-07-01)"),
        end_date: str = Query(..., description="End date in format YYYY-MM-DD (e.g., 2024-07-02)")
):
    try:
        start_date = datetime.strptime(start_date.strip(), "%Y-%m-%d").replace(tzinfo=ZoneInfo("UTC"))
        end_date = datetime.strptime(end_date.strip(), "%Y-%m-%d").replace(tzinfo=ZoneInfo("UTC"))
    except ValueError as e:
        return {"error": f"Invalid date format. Use YYYY-MM-DD. {e}"}
    print("Dates ", start_date, end_date)

    this_station_fields = station_fields(station_id)
    filtered_field_standard_names = filter_fields(
        this_station_fields,
        criteria=[
            MeasureType.AIRTEMP,
            MeasureType.DEW_POINT,
            CollectionFrequency.MIN60,
            Units.FAHRENHEIT
        ]
    )
    print("Filtered stations ", filtered_field_standard_names)

    bulk_measure_response = bulk_measures(
        station_id,
        start_date,
        end_date,
        filtered_field_standard_names
    )
    df = bulk_measures_to_df(bulk_measure_response)
    return df.to_dict(orient='records')

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