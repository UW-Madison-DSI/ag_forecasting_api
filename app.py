from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from pywisconet.data import *
from pywisconet.process import *
from starlette.middleware.wsgi import WSGIMiddleware

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

@app.get("/all_stations")
def stations_query(
        n_days_active: int
):
    try:
        result = all_stations(n_days_active)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Stations not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

'''Check this one
@app.get("/all_data_for_station/{station_id}")
def all_data_for_station_query(
    station_id: str,
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to include in the response")
):
    try:
        field_list = fields.split(",") if fields else None
        # Call the function to retrieve data for the station
        result = all_data_for_station(station_id, field_list)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Station {station_id} not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''

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