from fastapi import FastAPI, HTTPException
from pywisconet.data import *
from pywisconet.process import *
from starlette.middleware.wsgi import WSGIMiddleware

# Initialize FastAPI app
app = FastAPI()

# Testing station_fields
@app.get("/station_fields/{station_id}")
def read_weather(station_id: str):
    try:
        result = station_fields(station_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Station {station_id} not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

# This is the key part for RStudio Connect
wsgi_app = WSGIMiddleware(create_wsgi_app())