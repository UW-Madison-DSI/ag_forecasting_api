import numpy as np
from fastapi import HTTPException
from ag_models_wrappers.process_ibm_risk_v2 import get_weather


def run_ibm_forecast(
    forecasting_date: str,
    latitude: float,
    longitude: float,
    API_KEY: str,
    TENANT_ID: str,
    ORG_ID: str,
):
    try:
        weather_data = get_weather(
            latitude,
            longitude,
            forecasting_date,
            ORG_ID,
            TENANT_ID,
            API_KEY,
        )

        df = weather_data["daily"]
        df = df.replace([np.inf, -np.inf, np.nan], None)

        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))