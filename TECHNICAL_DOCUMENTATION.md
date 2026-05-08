# Agricultural Forecasting API - Technical Documentation

A FastAPI-based service for crop disease forecasting and biomass prediction using weather data integration from multiple sources. Developed by the University of Wisconsin-Madison Data Science Institute.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [API Endpoints](#api-endpoints)
5. [Core Features](#core-features)
6. [Data Flow](#data-flow)
7. [Disease Models](#disease-models)
8. [Biomass Models](#biomass-models)
9. [Weather Data Sources](#weather-data-sources)
10. [Response Format](#response-format)
11. [Development Setup](#development-setup)
12. [Deployment](#deployment)

---

## Overview

The Agricultural Forecasting API provides real-time crop disease risk predictions and winter rye biomass estimates for Wisconsin agricultural locations. The system integrates historical weather data with scientifically-validated disease forecasting models to help farmers and agricultural professionals make informed management decisions.

**Key Capabilities:**
- Real-time disease risk predictions for corn and soybean
- Winter rye biomass accumulation forecasts
- Multi-source weather data integration (IBM EIS, WiscoNet, NOAA)
- Geospatial queries by coordinates or weather station ID
- Batch processing for multi-station analysis
- GeoJSON-formatted responses for mapping integration

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │   /v2/ibm       │  │  /v2/wisconet_g  │  │  /v2/models │ │
│  │  (Coordinates)  │  │  (Station-based) │  │  (Metadata) │ │
│  └────────┬────────┘  └────────┬─────────┘  └─────────────┘ │
│           │                    │                             │
├───────────┼────────────────────┼─────────────────────────────┤
│           │                    │                             │
│  ┌────────▼────────────┐  ┌───▼──────────────────┐           │
│  │  IBM Service        │  │  Pipeline / WiscoNet │           │
│  │  (ibm_service.py)   │  │  (pipeline.py)       │           │
│  │                     │  │                      │           │
│  │ • Authenticate      │  │ • Load stations      │           │
│  │ • Fetch raw data    │  │ • Fetch measurements │           │
│  │ • Build hourly      │  │ • Compute biomass    │           │
│  │ • Build daily       │  │ • Annotate risks     │           │
│  │ • Compute rolling   │  │                      │           │
│  │   averages          │  │                      │           │
│  └────────┬────────────┘  └───┬──────────────────┘           │
│           │                   │                              │
├───────────┼───────────────────┼──────────────────────────────┤
│           │                   │                              │
│  ┌────────▼─────────────────────┴──────┐                     │
│  │     Risk Models & Biomass Calc       │                    │
│  │     (api/models/risk_models.py)      │                    │
│  │                                      │                    │
│  │ • Tarspot risk (corn)                │                    │
│  │ • Gray Leaf Spot (corn)              │                    │
│  │ • Frogeye Leaf Spot (soybean)        │                    │
│  │ • White Mold (soybean)               │                    │
│  │ • Winter Rye Biomass (logistic)      │                    │
│  └────────┬─────────────────────────────┘                    │
│           │                                                  │
├───────────┼──────────────────────────────────────────────────┤
│           │                                                  │
│  ┌────────▼────────────────────────────┐                     │
│  │  Output Formatting & Response        │                    │
│  │  (adapters/geojson_adapter.py)       │                    │
│  │                                      │                    │
│  │ • GeoJSON Feature Collections        │                    │
│  │ • Time-series grouping               │                    │
│  │ • Station metadata embedding         │                    │
│  └──────────────────────────────────────┘                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
ag_forecasting_api/
├── app.py                          # Main FastAPI entry point (v2)
├── app_v1.py                       # Legacy API (deprecated, v1)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image definition
├── docker-compose.yml              # Multi-container orchestration
│
├── api/                            # Core business logic
│   ├── __init__.py
│   ├── config/
│   │   └── constants.py            # Configuration, URLs, thresholds
│   │
│   ├── models/
│   │   ├── risk_models.py          # Disease & biomass calculations
│   │   └── disease_metadata.py     # Model metadata registry
│   │
│   ├── services/
│   │   ├── ibm_service.py          # IBM EIS weather integration
│   │   ├── wisconet_service.py     # WiscoNet station pipeline
│   │   └── risk_processor.py       # Risk annotation orchestration
│   │
│   ├── routes/
│   │   ├── wisconet.py             # /v2/ag_models_wrappers/wisconet_g
│   │   └── models.py               # /v2/ag_models_wrappers/models
│   │
│   ├── adapters/
│   │   └── geojson_adapter.py      # DataFrame → GeoJSON converter
│   │
│   ├── schemas/
│   │   ├── geojson_schema.py       # Response schema definitions
│   │   └── model_schema.py         # Model metadata schemas
│   │
│   ├── utils/
│   │   ├── math_helpers.py         # GDD, logistic, rolling averages
│   │   ├── conversions.py          # Unit conversions (F↔C, mph↔m/s)
│   │   └── caching.py              # Station & measurement caching
│   │
│   └── pipeline.py                 # Orchestrates WiscoNet workflow
│
├── api_cache/                      # Cached API responses (JSON)
│   ├── metadata_stations_*.json
│   └── station_data_*.json
│
└── materials/
    └── example_callapi.ipynb       # Usage examples
```

### Module Responsibilities

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `ibm_service.py` | IBM weather API integration | `get_weather_with_risk()`, `_fetch_raw_hourly()`, `_build_daily()`, `_annotate_winter_rye_biomass()` |
| `wisconet_service.py` | WiscoNet station data pipeline | `_compute_winter_rye_biomass()`, station loading & filtering |
| `risk_models.py` | Disease & biomass calculations | `calculate_tarspot_risk()`, `calculate_winter_rye_biomass()`, etc. |
| `pipeline.py` | Async WiscoNet orchestration | `retrieve()`, multi-station batch processing |
| `math_helpers.py` | Shared mathematical utilities | `gdd_sine()`, `logistic()`, `rolling_mean()` |
| `geojson_adapter.py` | Response formatting | `dataframe_to_featurecollection()` |

---

## Technology Stack

### Core Dependencies

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Web Framework | FastAPI | ^0.100 | REST API framework |
| ASGI Server | Uvicorn | ^0.24 | ASGI server |
| Data Processing | Pandas | ^2.0 | Tabular data manipulation |
| Numeric Computing | NumPy | ^1.24 | Numerical calculations |
| Data Validation | Pydantic | ^2.0 | Request/response schema validation |
| HTTP Client | requests | ^2.31 | Synchronous HTTP calls |
| Async HTTP | aiohttp | ^3.9 | Async HTTP requests |
| Logging | Python logging | stdlib | Structured logging |

### External Services

- **IBM Environmental Intelligence Suite (EIS)**: Premium weather API with historical-on-demand (HOD) data
- **WiscoNet**: Public Wisconsin mesonet weather station network
- **NOAA**: National weather service data for historical context

### Deployment

- **Container**: Docker with Alpine Linux base
- **Orchestration**: Docker Compose for multi-service deployments
- **Reverse Proxy**: Traefik for URL routing and TLS termination
- **Health Monitoring**: Periodic curl-based health checks

---

## API Endpoints

### V2 Endpoints (Current)

#### Disease Risk & Biomass by Coordinates (IBM)

```http
GET /v2/ag_models_wrappers/ibm
```

Query parameters:
- `forecasting_date` (required): Reference date (YYYY-MM-DD)
- `latitude` (required): Decimal degrees
- `longitude` (required): Decimal degrees
- `planting_date` (optional): For biomass calculation (YYYY-MM-DD)
- `termination_date` (optional): For WiscoNet biomass integration (YYYY-MM-DD)
- `API_KEY` (required): IBM EIS credential
- `TENANT_ID` (required): IBM EIS credential
- `ORG_ID` (required): IBM EIS credential

Example:
```bash
curl "http://localhost:8000/v2/ag_models_wrappers/ibm?forecasting_date=2025-05-15&latitude=43.0&longitude=-89.0&planting_date=2024-09-15&API_KEY=xxx&TENANT_ID=yyy&ORG_ID=zzz"
```

Response: GeoJSON FeatureCollection with hourly and daily weather data plus risk scores.

---

#### Disease Risk & Biomass by WiscoNet Stations

```http
GET /v2/ag_models_wrappers/wisconet_g
```

Query parameters:
- `forecasting_date` (required): Reference date (YYYY-MM-DD)
- `risk_days` (optional): Days to compute rolling risk (default: 1)
- `station_id` (optional): Filter to single station (e.g., "ALTN")
- `planting_date` (optional): For winter rye biomass (YYYY-MM-DD)
- `termination_date` (optional): For crop termination biomass (YYYY-MM-DD)
- `disease` (optional): Filter to specific disease model

Example:
```bash
curl "http://localhost:8000/v2/ag_models_wrappers/wisconet_g?forecasting_date=2025-05-15&station_id=ALTN&planting_date=2024-09-15"
```

Response: GeoJSON FeatureCollection grouped by station with time-series risk data.

---

#### Model Metadata

```http
GET /v2/ag_models_wrappers/models
```

Returns list of all available disease models with descriptions, crops, risk categories, and input requirements.

---

### Legacy V1 Endpoints

The API maintains backward compatibility with v1 endpoints (mounted at `/v1`), including:
- `/v1/sites` - Station listing
- `/v1/predictions` - Disease predictions
- `/v1/models` - Model metadata

---

## Core Features

### 1. Multi-Source Weather Integration

The API combines weather data from multiple sources with intelligent fallback:

```
┌──────────────────────────────────────────────────┐
│ Weather Data Request (lat/lng, date range)       │
└────────────────┬─────────────────────────────────┘
                 │
     ┌───────────┴──────────────┐
     │                          │
     ▼                          ▼
┌─────────────────┐      ┌──────────────────┐
│ IBM EIS API     │      │ WiscoNet Network │
│ (Premium,       │      │ (Public,         │
│ high res)       │      │ station-based)   │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         └────────────┬───────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │ Data Harmonization   │
            │ • Timezone aware     │
            │ • Unit conversion    │
            │ • Interpolation      │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Growing Degree Days  │
            │ • Sine method        │
            │ • Base: 0°C, Cap: 35°C│
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Rolling Averages     │
            │ • 7, 14, 21, 30-day  │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │ Risk Model Input     │
            │ (Ready for scoring)  │
            └──────────────────────┘
```

### 2. Efficient Data Caching

Reduces redundant API calls through tiered caching:

```python
# Cache layers (in api/utils/caching.py)
- Station metadata: 7-day TTL
- Measurements: 6-hour TTL
- IBM responses: Variable TTL based on forecast date
- File system: JSON cache in api_cache/ directory
```

### 3. Async Pipeline Processing

Multi-station workflows use async processing for performance:

```python
# pipeline.py
async def _run_pipeline(...):
    # 1. Load all active stations (filtered)
    stations = load_stations_sync(...)
    
    # 2. Fetch all measurements concurrently
    measurements = await fetch_all_measurements_async(stations)
    
    # 3. Merge metadata (station location, timezone, etc.)
    merged = merge_station_metadata(measurements, stations)
    
    # 4. Compute winter rye biomass (if planting_date provided)
    if planting_date:
        biomass = _compute_winter_rye_biomass(merged, planting_date)
        merged = merge_biomass(merged, biomass)
    
    # 5. Optionally fetch WiscoNet-specific biomass data
    if both planting_date and termination_date:
        wisconet_biomass = await fetch_wisconet_biomass(...)
        merged = merge_wisconet_biomass(merged, wisconet_biomass)
    
    # 6. Compute disease risks in parallel
    risk_df = compute_risks_in_parallel(
        merged, num_workers=4
    )
    
    return risk_df
```

### 4. Intelligent Date & Timezone Handling

- Automatic timezone conversion to station local time
- Date arithmetic aware of DST transitions
- Support for both absolute dates and relative forecasts

---

## Disease Models

### Tarspot Risk (Corn)

**Crop:** Corn  
**Disease:** *Phyllachora maydis* (Tar Spot fungus)

**Model Type:** Ensemble logistic regression (two sub-models averaged)

**Inputs:**
- 30-day mean air temperature (°C)
- 30-day max relative humidity (%)
- 14-day mean nighttime hours with RH ≥ 90%

**Output:** Risk probability (0–1), classified as:
- **Inactive** (mean temp < 10°C)
- **1.Low** (prob < 0.20)
- **2.Moderate** (prob 0.20–0.35)
- **3.High** (prob > 0.35)

**Reference:** Damon et al.

---

### Gray Leaf Spot Risk (Corn)

**Crop:** Corn  
**Disease:** *Cercospora zeae-maydis* (Gray Leaf Spot)

**Model Type:** Single logistic regression

**Inputs:**
- 21-day min air temperature (°C)
- 30-day min dew point (°C)

**Assumptions:** Growth stage V10–R3, no irrigation

**Output:** Risk probability (0–1), same classification as Tarspot

---

### Frogeye Leaf Spot Risk (Soybean)

**Crop:** Soybean  
**Disease:** *Cercospora sojina* (Frogeye Leaf Spot)

**Model Type:** Logistic regression

**Inputs:**
- Degree-days above 14°C (cumulative from emergence)
- Rainfall in previous 14 days (mm)
- Hours with RH ≥ 85% (14-day rolling)

**Output:** Risk probability (0–1)

---

### White Mold Risk (Soybean)

**Crop:** Soybean  
**Disease:** *Sclerotinia sclerotiorum* (White Mold / Sclerotinia)

**Model Type:** Dual variant (irrigated & non-irrigated)

**Inputs:**
- Days since planting
- Cumulative precipitation (mm)
- Soil moisture availability

**Variants:**
- **Dry:** Non-irrigated fields
- **Irrigated 15":** 15-inch row spacing
- **Irrigated 30":** 30-inch row spacing

**Output:** Apothecial presence probability (0–1)

---

### Winter Rye Biomass (NEW)

**Crop:** Winter Rye  
**Metric:** Aboveground dry biomass (lb/acre)

**Model Type:** Logistic growth model

**Formula:**

$$\text{logit} = b_0 + b_{pd} \cdot \text{plant\_doy} + b_{pf} \cdot \text{precip\_fall}$$

$$\text{pred} = \frac{\text{logit}}{1 + e^{-k(\text{gdd\_total}-x_0)}}$$

$$\text{biomass} = \max(0, \text{pred}^2)$$

**Coefficients:**
- $b_0 = 423.1$ (intercept)
- $b_{pd} = -1.031$ (day-of-year effect)
- $b_{pf} = -0.2878$ (fall precipitation effect)
- $k = 0.003663$ (logistic rate)
- $x_0 = 1049.0$ (inflection point GDD)

**Inputs:**
- Planting day-of-year (1–366)
- Precipitation during fall establishment (mm)
- Cumulative GDD from planting to current date (0°C base)

**Output:**
- **Biomass** (lb/acre): Numeric prediction
- **Biomass Color:**
  - **Gray**: 0–2000 lb/acre (low coverage)
  - **Yellow**: 2000–4500 lb/acre (moderate)
  - **Green**: >4500 lb/acre (high/dense)
- **Biomass Message**: Human-readable interpretation

**Use Cases:**
- Cover crop management planning
- Soil health assessment
- Erosion mitigation evaluation
- Residue management decisions

---

## Biomass Models

### Winter Rye Biomass Calculation Pipeline

```python
def _annotate_winter_rye_biomass(
    rye_daily: pd.DataFrame,
    planting_date: str
) -> pd.DataFrame:
    """
    Compute winter rye biomass from daily weather data.
    
    Steps:
    1. Parse planting_date → day-of-year
    2. Filter to dates >= planting_date
    3. Compute daily GDD (sine method, 0°C base, 35°C cap)
    4. Accumulate GDD and precipitation
    5. Call calculate_winter_rye_biomass() for each row
    6. Merge results back into DataFrame
    """
    
    # Step 1: Parse planting date
    plant_date_obj = datetime.fromisoformat(planting_date).date()
    plant_doy = plant_date_obj.timetuple().tm_yday
    
    # Step 2: Filter to >= planting date
    rye_daily = rye_daily[rye_daily['date'] >= planting_date].copy()
    
    # Step 3-4: GDD accumulation
    rye_daily['plant_doy'] = plant_doy
    rye_daily['precip_daily'] = rye_daily['precip1Hour_sum'].fillna(0)
    rye_daily['gdd_0c'] = rye_daily.apply(
        lambda row: gdd_sine(row['temperature_min'], row['temperature_max']),
        axis=1
    )
    rye_daily['gdd_total'] = rye_daily['gdd_0c'].cumsum()
    
    # Compute fall precipitation (Sep 1 to planting)
    fall_start = plant_date_obj.replace(month=9, day=1)
    fall_precip = rye_daily[
        (rye_daily['date'] >= str(fall_start)) & 
        (rye_daily['date'] < planting_date)
    ]['precip_daily'].sum()
    
    # Step 5: Model calculation
    biomass_results = rye_daily.apply(
        lambda row: calculate_winter_rye_biomass(
            plant_doy=row['plant_doy'],
            precip_fall=fall_precip,
            gdd_total=row['gdd_total']
        ),
        axis=1,
        result_type='expand'
    )
    
    # Step 6: Merge results
    for col in ['biomass_lb_acre', 'biomass_color', 'biomass_message']:
        rye_daily[col] = biomass_results[col]
    
    return rye_daily
```

---

## Weather Data Sources

### IBM Environmental Intelligence Suite (EIS)

**Characteristics:**
- Premium, high-resolution service
- Hourly data with 999-hour request window per call
- Requires JWT authentication
- Covers entire globe
- 15-minute resolution available

**Data Acquisition:**
1. Obtain JWT token via SaaSCore authentication
2. Request hourly-on-demand (HOD) data in time chunks
3. Convert to timezone-aware UTC timestamps
4. Interpolate/aggregate to daily summaries

**Cached Columns:**
- `temperature_min`, `temperature_max`, `temperature_mean`
- `temperatureDewPoint_min`, `temperatureDewPoint_max`
- `relativeHumidity_min`, `relativeHumidity_max`
- `windSpeed_max`, `windSpeed_mean`
- `precip1Hour_sum` (hourly accumulated)
- Rolling averages: 7, 14, 21, 30-day

---

### WiscoNet (Public Mesonet)

**Characteristics:**
- Wisconsin statewide mesonet network
- ~100 active weather stations
- Public API, no authentication required
- Daily data update cycle
- Station metadata includes: ID, name, lat/lng, county, region

**Measurement Types:**
- Air temperature (min, avg, max)
- Dew point (min, avg, max)
- Relative humidity (%)
- Wind speed (max, direction)
- Precipitation (daily total)
- Soil temperature (select stations)
- Leaf wetness (select stations)

**Data Retrieval:**
- Station list via `/wisconet_active_stations`
- Daily measurements via bulk query endpoint
- Cached in `station_measurements_cache/` (6-hour TTL)

---

## Response Format

### GeoJSON Feature Collection

All responses follow the GeoJSON standard with extended properties:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-89.4, 43.1]
      },
      "properties": {
        "date": "2025-05-15",
        "forecasting_date": "2025-05-15",
        "station_id": "ALTN",
        "station_name": "Alten",
        "city": "Alten",
        "county": "Columbia",
        "region": "South Central",
        "state": "WI",
        "latitude": 43.1234,
        "longitude": -89.4567,
        "station_timezone": "America/Chicago",
        
        "temperature_min_c": 8.3,
        "temperature_max_c": 21.5,
        "temperature_mean_c": 15.2,
        "relativeHumidity_max": 92,
        "precip1Hour_sum": 2.5,
        
        "tarspot_risk": 0.45,
        "tarspot_risk_class": "2.Moderate",
        "gls_risk": 0.28,
        "gls_risk_class": "1.Low",
        "frogeye_risk": 0.12,
        "frogeye_risk_class": "1.Low",
        "whitemold_dry_risk": 0.35,
        "whitemold_dry_risk_class": "2.Moderate",
        
        "biomass_lb_acre": 3250,
        "biomass_color": "Yellow",
        "biomass_message": "Moderate coverage"
      }
    }
  ]
}
```

### Time-Series Grouping (WiscoNet)

For multi-day forecasts, results are grouped by station with embedded time-series:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Point", "coordinates": [-89.4, 43.1] },
      "properties": {
        "station_id": "ALTN",
        "station_name": "Alten",
        "city": "Alten",
        "county": "Columbia",
        "region": "South Central",
        "state": "WI",
        "time_series": [
          {
            "date": "2025-05-15",
            "temperature_min_c": 8.3,
            "tarspot_risk": 0.45,
            ...
          },
          {
            "date": "2025-05-16",
            "temperature_min_c": 7.8,
            "tarspot_risk": 0.38,
            ...
          }
        ]
      }
    }
  ]
}
```

---

## Data Flow

### Complete Request → Response Journey

```
User Request
    │
    ▼
┌──────────────────────────────┐
│ Route Handler (app.py)       │
│ Validate query parameters    │
│ Extract coordinates/dates    │
└────────────────┬─────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
    ▼                         ▼
┌─────────────────┐    ┌───────────────┐
│ IBM Service     │    │ WiscoNet      │
│                 │    │ Pipeline      │
│ 1. Authenticate │    │               │
│    with IBM     │    │ 1. Load       │
│ 2. Chunk time   │    │    stations   │
│    range        │    │ 2. Fetch      │
│ 3. Fetch hourly │    │    measurements
│    data         │    │ 3. Merge      │
│ 4. Build daily  │    │    metadata   │
│    summaries    │    │ 4. Cache      │
│                 │    │    results    │
└────────┬────────┘    └────────┬──────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Math Helpers         │
         │ • Convert units      │
         │ • GDD calculation    │
         │ • Rolling averages   │
         │ • Biomass prep       │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Risk Model Scoring   │
         │ Apply all diseases   │
         │ Compute probabilities│
         │ Classify risk class  │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Biomass Calculation  │
         │ (if planting_date)   │
         │ Logistic model       │
         │ Color classification │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ GeoJSON Adapter      │
         │ • Format geometry    │
         │ • Embed properties   │
         │ • Group by station   │
         └──────────┬───────────┘
                    │
                    ▼
        Response (GeoJSON)
```

---

## Development Setup

### Prerequisites

- Python 3.10+
- pip or conda
- Git
- Docker (optional, for containerized development)

### Local Installation

```bash
# Clone repository
git clone https://github.com/UW-Madison-DSI/ag_forecasting_api.git
cd ag_forecasting_api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export IBM_API_KEY="your_key"
export TENANT_ID="your_tenant"
export ORG_ID="your_org"

# Run development server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, view interactive documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

### Testing

```bash
# Syntax validation
python -m py_compile api/models/risk_models.py api/services/*.py

# Run linting (if installed)
pylint api/

# Test endpoint with curl
curl "http://localhost:8000/v2/test/rye_biomass"
```

---

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose (Production)

```yaml
version: "3.9"
services:
  ag_forecasting_api:
    build: .
    expose:
      - 8000
    environment:
      - PYTHONPATH=/app
      - ENVIRONMENT=production
      - IBM_API_KEY=${IBM_API_KEY}
      - TENANT_ID=${TENANT_ID}
      - ORG_ID=${ORG_ID}
    networks:
      - traefik
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.ag_api.rule=Host(`your-domain.com`)"
      - "traefik.http.services.ag_api.loadbalancer.server.port=8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Environment Variables

Required for production:
- `IBM_API_KEY`: IBM EIS authentication credential
- `TENANT_ID`: IBM EIS tenant identifier
- `ORG_ID`: IBM EIS organization identifier
- `ENVIRONMENT`: Set to "production" for optimized logging

Optional:
- `PYTHONPATH`: Should include `/app` for imports
- `LOG_LEVEL`: Logging verbosity (default: INFO)

---

## Performance Considerations

### Caching Strategy

- **Station Metadata**: 7-day TTL (rarely changes)
- **Measurements**: 6-hour TTL (updated daily by WiscoNet)
- **IBM Responses**: Variable (tied to forecast date relevance)
- **File System**: JSON cache in `api_cache/` for persistence

### Concurrency

- **Async Pipeline**: Multi-station queries use async/await for I/O efficiency
- **Parallel Risk Scoring**: Disease calculations use multiprocessing (4 workers default)
- **Connection Pooling**: HTTP clients maintain connection pools (max 50 concurrent)

### Optimization Tips

1. **Batch Requests**: Query multiple stations in one call rather than sequential individual queries
2. **Use WiscoNet**: Public API has lower latency than premium IBM service
3. **Limit Date Range**: Smaller date ranges reduce computation time
4. **Filter Diseases**: Specify disease models needed to reduce calculation overhead
5. **Cache Warming**: Pre-fetch station lists during off-peak hours

---

## Additional Resources

- **GitHub Repository**: https://github.com/UW-Madison-DSI/ag_forecasting_api
- **Interactive Dashboard**: https://connect.doit.wisc.edu/ag_forecasting/
- **Example Notebook**: `/materials/example_callapi.ipynb`
- **University Contact**: UW-Madison Data Science Institute

---

## License

See LICENSE file for details.

---

## Changelog

### Version 2.0 (Current)

- ✅ Winter rye biomass model integration
- ✅ Unified API versioning (v1 legacy, v2 current)
- ✅ Enhanced GeoJSON responses
- ✅ Improved async pipeline performance
- ✅ Extended documentation

### Version 1.0 (Legacy)

- Disease risk models for corn and soybean
- WiscoNet and IBM EIS integration
- Basic GeoJSON support

---

**Last Updated:** May 2025  
**Maintained By:** UW-Madison Data Science Institute
