from fastapi import APIRouter, HTTPException
from api.services.model_registry import registry
from api.services.ibm_service import run_ibm_forecast
from api.services.wisconet_service import run_wisconet_forecast
from api.services.geojson_service import dataframe_to_geojson
from ag_models_wrappers.process_wisconet import retrieve

router = APIRouter(prefix="/models", tags=["Models"])

# Register models
registry.register("ibm", run_ibm_forecast)
registry.register("wisconet", run_wisconet_forecast)


@router.get("/")
def list_models():
    return {"available_models": registry.list_models()}


@router.get("/{model_name}/forecast")
def run_model(model_name: str, **kwargs):
    model = registry.get(model_name)

    if not model:
        raise HTTPException(404, "Model not found")

    return model(**kwargs)


@router.get("/{model_name}/geojson")
def run_model_geojson(model_name: str, **kwargs):
    if model_name != "wisconet":
        raise HTTPException(400, "GeoJSON only supported for wisconet")

    df = retrieve(
        input_date=kwargs.get("forecasting_date"),
        input_station_id=kwargs.get("station_id"),
        days=kwargs.get("risk_days", 1),
    )

    if df is None or len(df) == 0:
        return {"type": "FeatureCollection", "features": []}

    return dataframe_to_geojson(df)