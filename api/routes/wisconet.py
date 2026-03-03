from fastapi import APIRouter
from api.pipeline import retrieve
from api.adapters.geojson_adapter import dataframe_to_featurecollection
from api.schemas.geojson_schema import FeatureCollection

router = APIRouter()


@router.get(
    "/wisconet_g",
    response_model=FeatureCollection
)
def wisconet_geojson_grouped(
    forecasting_date: str,
    risk_days: int = 1,
    station_id: str | None = None,
    planting_date: str | None = None,
    termination_date: str | None = None
):
    df = retrieve(
        input_date=forecasting_date,
        input_station_id=station_id,
        days=risk_days,
        planting_date=planting_date,
        termination_date=termination_date
    )

    return dataframe_to_featurecollection(df)