from fastapi import FastAPI
from api.routers import health, models, stations

app = FastAPI(
    title="AG Forecasting API",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(stations.router)


@app.get("/")
def root():
    return {"message": "AG Forecasting API"}