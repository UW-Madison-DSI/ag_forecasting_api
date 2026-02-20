import numpy as np
from fastapi import HTTPException
from ag_models_wrappers.process_wisconet import retrieve


def run_wisconet_forecast(
    forecasting_date: str,
    risk_days: int = 1,
    station_id: str = None,
):
    try:
        df = retrieve(
            input_date=forecasting_date,
            input_station_id=station_id,
            days=risk_days,
        )

        if df is None or len(df) == 0:
            return []

        df = df.replace([np.inf, -np.inf, np.nan], None)
        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))