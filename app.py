from fastapi import FastAPI, HTTPException
from pywisconet.data import *
from pywisconet.process import *

# Initialize FastAPI app
app = FastAPI()

# Example endpoint: station_fields
@app.get("/station_fields/{station_id}")
def read_weather(station_id: str):
    try:
        result = station_fields(station_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Station {station_id} not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a root route to resolve the 404 on root
@app.get("/")
def read_root():
    return {"message": "Welcome to the Wisconsin Weather API"}