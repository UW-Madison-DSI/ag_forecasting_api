from fastapi import FastAPI
from pywisconet.data import *
from pywisconet.process import *

# Initialize FastAPI app
app = FastAPI()

# Example endpoint: station_fields
@app.get("/station_fields/{station_id}")
def read_weather(station_id: str):
    return station_fields(station_id)

