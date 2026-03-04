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
    #planting_date: str | None = None,
    #termination_date: str | None = None,
    disease: str | None = None
):
    df = retrieve(
        input_date=forecasting_date,
        input_station_id=station_id,
        days=risk_days,
        planting_date=None,
        termination_date=None
    )
    if disease:
        disease_cols = [
            c for c in df.columns
            if c.startswith(disease)
        ]

        keep_cols = [
                        "station_id",
                        "station_name",
                        "city",
                        "county",
                        "latitude",
                        "longitude",
                        "region",
                        "state",
                        "station_timezone",
                        "date"
                    ] + disease_cols

        df = df[keep_cols]

    return dataframe_to_featurecollection(df)