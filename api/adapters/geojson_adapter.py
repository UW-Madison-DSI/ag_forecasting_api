import pandas as pd
from api.schemas.geojson_schema import (
    FeatureCollection,
    Feature,
    Station,
    Coordinates,
    Observation,
    FieldValue,
    StandardMeasure
)

META_COLUMNS = [
    "station_id",
    "station_name",
    "city",
    "county",
    "latitude",
    "longitude",
    "region",
    "state",
    "station_timezone"
]


import numpy as np
import pandas as pd
import math

def clean_for_json(df: pd.DataFrame) -> pd.DataFrame:
    # Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Convert NaN to None (JSON null)
    df = df.astype(object).where(pd.notnull(df), None)

    return df

def dataframe_to_featurecollection(df: pd.DataFrame):
    # Replace invalid JSON floats
    df = clean_for_json(df)
    # Convert all numpy types to native Python
    df = df.applymap(
        lambda x: x.item() if hasattr(x, "item") else x
    )
    if df is None or df.empty:
        return FeatureCollection(fields=[], features=[])

    #data_columns = [c for c in df.columns if c not in META_COLUMNS]
    EXCLUDED_COLUMNS = META_COLUMNS + ["date"]

    data_columns = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    # -----------------------------
    # Build field metadata
    # -----------------------------
    fields = []
    for col in data_columns:
        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            data_type = "FLOAT"
        else:
            data_type = "STR"

        fields.append(
            StandardMeasure(
                fieldname=col,
                measure=col,
                frequency="daily",
                units=None,
                aggregation=None,
                data_type=data_type
            )
        )

    # -----------------------------
    # Build features
    # -----------------------------
    features = []

    for station_id, group in df.groupby("station_id"):
        first = group.iloc[0]

        station = Station(
            station_name=first["station_name"],
            station_id=station_id,
            coordinates=Coordinates(
                latitude=first["latitude"],
                longitude=first["longitude"]
            ),
            city=first.get("city"),
            county=first.get("county"),
            region=first.get("region"),
            state=first.get("state"),
            timezone=first.get("station_timezone")
        )

        timeseries = []

        for _, row in group.iterrows():

            values = [
                FieldValue(fieldname=col, value=row[col])
                for col in data_columns
            ]

            timeseries.append(
                Observation(
                    date=row["date"],
                    data=values
                )
            )

        features.append(
            Feature(
                station=station,
                timeseries=timeseries
            )
        )

    return FeatureCollection(
        fields=fields,
        features=features
    )