"""
/api/v1/predictions

All stations
  GET /predictions?date=&model=&format=&include=
  GET /predictions?start=&end=&model=&format=&include=

Single station (nested under /sites)
  GET /sites/{station_id}/predictions?date=&model=&format=&include=
  GET /sites/{station_id}/predictions?start=&end=&model=&format=&include=

IBM coordinate-based (legacy passthrough)
  GET /predictions/ibm?forecasting_date=&latitude=&longitude=&API_KEY=&TENANT_ID=&ORG_ID=
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query

from api import retrieve
from api.services.ibm_service import get_weather_with_risk
from app_v1.dependencies import DateRangeDep, FormatDep, IncludeMetaDep, ModelsDep
from app_v1.formatters import to_feature_collection, to_legacy_json
from app_v1.schemas.predictions import PredictionFeatureCollection

router = APIRouter(tags=["Predictions"])


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _fetch_and_clean(end_date: str, days: int, station_id: str | None) -> pd.DataFrame | None:
    """Run the WiscoNet pipeline and sanitise the result for JSON serialisation."""
    df = retrieve(input_date=end_date, input_station_id=station_id, days=days)
    if df is None or df.empty:
        return None
    return df.replace([np.inf, -np.inf, np.nan], None)


def _format_response(
    df: pd.DataFrame | None,
    models: list[str],
    fmt: str,
    include_meta: bool,
    station_id: str | None = None,
):
    """
    Choose formatter and handle the empty-data case.
    Raises 404 for single-station requests with no data; returns an empty
    FeatureCollection for all-stations requests.
    """
    if df is None or df.empty:
        if station_id:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for station '{station_id}' in the requested range.",
            )
        return PredictionFeatureCollection(
            metadata=to_feature_collection(pd.DataFrame(), models, False).metadata,
            features=[],
        )

    return to_legacy_json(df, models) if fmt == "json" else to_feature_collection(df, models, include_meta)


# ── All-stations predictions ───────────────────────────────────────────────────

@router.get(
    "/predictions",
    response_model=PredictionFeatureCollection,
    summary="Disease-risk predictions across all stations",
)
def get_predictions(
    date_range:   DateRangeDep,
    models:       ModelsDep,
    fmt:          FormatDep,
    include_meta: IncludeMetaDep,
):
    df = _fetch_and_clean(date_range.end, date_range.days, station_id=None)
    return _format_response(df, models, fmt, include_meta)


# ── Single-station predictions ─────────────────────────────────────────────────

@router.get(
    "/sites/{station_id}/predictions",
    response_model=PredictionFeatureCollection,
    summary="Disease-risk predictions for a single station",
)
def get_station_predictions(
    station_id:   str = Path(..., description="WiscoNet station identifier, e.g. 'ALTN'"),
    date_range:   DateRangeDep   = None,
    models:       ModelsDep      = None,
    fmt:          FormatDep      = None,
    include_meta: IncludeMetaDep = None,
):
    df = _fetch_and_clean(date_range.end, date_range.days, station_id=station_id)
    # Single-station responses always embed station metadata (no opt-in required)
    return _format_response(df, models, fmt, include_meta=True, station_id=station_id)


# ── IBM coordinate-based (legacy passthrough) ──────────────────────────────────

@router.get(
    "/predictions/ibm",
    summary="IBM EIS weather-driven risk predictions for a lat/lng coordinate",
)
def get_ibm_predictions(
    forecasting_date: str   = Query(..., description="Reference date (YYYY-MM-DD)."),
    latitude:         float = Query(...),
    longitude:        float = Query(...),
    API_KEY:          str   = Query(..., description="IBM EIS API key."),
    TENANT_ID:        str   = Query(..., description="IBM EIS tenant ID."),
    ORG_ID:           str   = Query(..., description="IBM EIS organisation ID."),
):
    result = get_weather_with_risk(
        lat=latitude, lng=longitude, end_date=forecasting_date,
        api_key=API_KEY, tenant_id=TENANT_ID, org_id=ORG_ID,
    )
    df = result.get("daily") if result else None
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="IBM weather data currently unavailable.")

    return df.replace([np.inf, -np.inf, np.nan], None).to_dict(orient="records")
