from fastapi import FastAPI
from api.routes import wisconet, models
from app_v1 import app_v1

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

@app.get("/")
def root():
    return {"message": "Ag Forecasting API — v1 and v2 available"}