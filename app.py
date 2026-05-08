from fastapi import FastAPI, HTTPException, Query
import numpy as np
import pandas as pd
from api.routes import wisconet, models
from app_v1 import app_v1
from api.services.ibm_service import get_weather_with_risk

app = FastAPI(
    title="Ag Forecasting API",
    version="2.0.0"
)

# Mount legacy API
app.mount("/v1", app_v1)

# Version 2 routers
app.include_router(
    wisconet.router,
    prefix="/v2/ag_models_wrappers",
    tags=["Wisconet"]
)

app.include_router(
    models.router,
    prefix="/v2/ag_models_wrappers",
    tags=["Disease Models"]
)


@app.get("/v2/ag_models_wrappers/ibm")
def ibm_weather_query(
    forecasting_date: str,
    latitude: float = Query(..., description="Latitude of the location"),
    longitude: float = Query(..., description="Longitude of the location"),
    API_KEY: str = Query(..., description="IBM API key"),
    TENANT_ID: str = Query(..., description="IBM tenant ID"),
    ORG_ID: str = Query(..., description="IBM organization ID"),
    planting_date: str | None = Query(None, description="Cover crop planting date for rye biomass, YYYY-MM-DD"),
):
    try:
        weather_data = get_weather_with_risk(
            latitude,
            longitude,
            forecasting_date,
            API_KEY,
            TENANT_ID,
            ORG_ID,
            planting_date=planting_date,
        )
        df = weather_data["daily"]
        df_cleaned = df.replace([np.inf, -np.inf, np.nan], None).where(pd.notnull(df), None)
        return df_cleaned.to_dict(orient="records")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/", tags=["Meta"])
def root():
    return {
        "api_name": "Ag Forecasting API",
        "available_versions": {
            "v1": "/v1/docs",
            "v2": "/docs"
        },
        "current_default": "v2"
    }

@app.get("/v2/test/rye_biomass", tags=["Testing"])
def test_rye_biomass():
    """Test endpoint to verify rye biomass calculation with sample data."""
    from api.models.risk_models import calculate_winter_rye_biomass
    
    # Test case: mid-September planting (doy ~260), moderate precip (100mm), spring growth (1500 GDD)
    result = calculate_winter_rye_biomass(
        plant_doy=260,      # ~Sep 16
        precip_fall=100.0,  # mm
        gdd_total=1500.0    # °C base 0
    )
    
    return {
        "test_input": {
            "plant_doy": 260,
            "precip_fall_mm": 100.0,
            "gdd_total_c": 1500.0,
            "interpretation": "Mid-September planting, 100mm fall precip, 1500 GDD by forecast date"
        },
        "biomass_prediction": result
    }