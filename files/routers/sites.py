"""
/api/v1/sites

GET /sites                  → all active stations (GeoJSON FeatureCollection)
GET /sites/{station_id}     → single station detail (GeoJSON Feature)
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query

from api import retrieve
from app_v1.schemas.stations import StationFeature, StationFeatureCollection, StationProperties

router = APIRouter(prefix="/sites", tags=["Sites"])


def _build_feature(row: dict) -> StationFeature:
    sid = row["station_id"]
    return StationFeature(
        id       = sid,
        geometry = {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]},
        properties = StationProperties(
            station_id = sid,
            name       = row.get("station_name", ""),
            city       = row.get("city", ""),
            county     = row.get("county", ""),
            region     = row.get("region", ""),
            state      = row.get("state", ""),
            timezone   = row.get("station_timezone", ""),
        ),
    )


@router.get(
    "",
    response_model=StationFeatureCollection,
    summary="List all active weather stations",
)
def list_sites(
    date: Optional[str] = Query(
        None,
        description="Filter to stations active as of this date (YYYY-MM-DD). Defaults to today.",
    ),
):
    input_date = date or str(datetime.utcnow().date())
    df = retrieve(input_date=input_date, days=1)
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Station data currently unavailable.")

    seen: set[str] = set()
    features = []
    for row in df.to_dict(orient="records"):
        sid = row["station_id"]
        if sid not in seen:
            seen.add(sid)
            features.append(_build_feature(row))

    return StationFeatureCollection(features=features)


@router.get(
    "/{station_id}",
    response_model=StationFeature,
    summary="Get a single station's details",
)
def get_site(
    station_id: str = Path(..., description="WiscoNet station identifier, e.g. 'ALTN'"),
):
    df = retrieve(
        input_date       = str(datetime.utcnow().date()),
        input_station_id = station_id,
        days             = 1,
    )
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found.")
    return _build_feature(df.iloc[0].to_dict())
