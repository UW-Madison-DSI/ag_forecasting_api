from fastapi import FastAPI
from api.routes import wisconet

app = FastAPI(
    title="Ag Forecasting API",
    version="1.0.0"
)

app.include_router(
    wisconet.router,
    prefix="/ag_models_wrappers",
    tags=["Wisconet"]
)

from api.routes import models

app.include_router(
    models.router,
    prefix="/ag_models_wrappers",
    tags=["Disease Models"]
)

@app.get("/")
def root():
    return {"message": "Welcome to the Ag Forecasting API"}