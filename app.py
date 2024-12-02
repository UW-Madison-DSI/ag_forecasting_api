from fastapi import FastAPI, HTTPException
from pywisconet.data import *
from pywisconet.process import *
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Your existing routes
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

# Add this for local testing and potential server compatibility
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)