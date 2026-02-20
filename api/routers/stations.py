from fastapi import APIRouter, HTTPException
from datetime import datetime
from zoneinfo import ZoneInfo
from pywisconet.data import all_stations

router = APIRouter(prefix="/stations", tags=["Stations"])


@router.get("/active")
def active_stations(min_days_active: int, start_date: str):
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(
            tzinfo=ZoneInfo("UTC")
        )

        result = all_stations(min_days_active, start_date)

        if result is None:
            raise HTTPException(404, "Stations not found")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))