"""
Crop Pest Prediction API  –  entry point.

This file only wires things together. All business logic lives in:
  api/   → data pipeline, risk models, weather services
  app/   → HTTP layer (schemas, routers, formatters, dependencies)

Run with:
    uvicorn app:app --reload

Docs (auto-generated):
    http://localhost:8000/api/v1/docs
    http://localhost:8000/api/v1/redoc
"""

import sys
import os

# Ensure the project root is on sys.path so `api.*` and `app.*` resolve.
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from app_v1.routers import (
    measurements_router,
    models_router,
    predictions_router,
    sites_router,
)

# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Crop Pest Prediction API",
    version     = "1.0.0",
    description = (
        "Disease-risk forecasts for Wisconsin agricultural stations.\n\n"
        "**Base URL:** `/api/v1`\n\n"
        "Default response is GeoJSON (`format=geojson`). "
        "Pass `format=json` for the legacy flat-array response."
    ),
    docs_url      = "/api/v1/docs",
    redoc_url     = "/api/v1/redoc",
    openapi_url   = "/api/v1/openapi.json",
)

# ── Versioned router ───────────────────────────────────────────────────────────

v1 = APIRouter(prefix="/api/v1")
v1.include_router(sites_router)
v1.include_router(models_router)
v1.include_router(predictions_router)
v1.include_router(measurements_router)

app.include_router(v1)

# ── Health & root ──────────────────────────────────────────────────────────────

@app.get("/api/v1/health", tags=["Meta"])
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({
        "message": "Crop Pest Prediction API",
        "docs":    "/api/v1/docs",
        "version": "1.0.0",
    })
