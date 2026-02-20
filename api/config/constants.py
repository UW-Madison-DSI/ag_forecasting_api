"""
Central configuration and constants.

All magic numbers, URLs, column names, and environment-driven values
live here so every other module imports from a single source of truth.
"""

import os

# ── Network ───────────────────────────────────────────────────────────────────
WISCONET_BASE_URL: str = "https://wisconet.wisc.edu/api/v1"
IBM_SAASCORE_URL: str = "https://api.ibm.com/saascore/run/authentication-retrieve"
IBM_GEOSPATIAL_URL: str = "https://api.ibm.com/geospatial/run/v3/wx/hod/r1/direct"

# ── Pagination / batching ─────────────────────────────────────────────────────
BATCH_SIZE: int = 20
IBM_CHUNK_HOURS: int = 999          # max hours per IBM API request

# ── Station filtering ─────────────────────────────────────────────────────────
MIN_DAYS_ACTIVE: int = 38
STATIONS_TO_EXCLUDE: list[str] = ["MITEST1", "WNTEST1"]

# ── Cache ─────────────────────────────────────────────────────────────────────
MEASUREMENTS_CACHE_DIR: str = "station_measurements_cache"
BIOMASS_CACHE_DIR: str = "station_biomass_cache"
STATIONS_CACHE_FILE: str = "wisconsin_stations_cache.csv"
MEASUREMENTS_CACHE_TTL_HOURS: int = 6
STATIONS_CACHE_TTL_DAYS: int = 7

# ── Async HTTP session ────────────────────────────────────────────────────────
HTTP_CONN_LIMIT: int = 50
HTTP_DNS_TTL: int = 300
HTTP_TIMEOUT_SECONDS: int = 60
HTTP_MAX_RETRIES: int = 3

# ── Measure IDs (WiscoNet API) ────────────────────────────────────────────────
MEASURE_ID_AIR_TEMP_F_AVG: int = 2
MEASURE_ID_DEW_POINT_F_AVG: int = 10
MEASURE_ID_RH_PCT_AVG: int = 19
MEASURE_ID_WIND_SPEED_MPH_MAX: int = 57
MEASURE_ID_DAILY_AIR_TEMP_F_AVG: int = 3
MEASURE_ID_DAILY_RAIN_IN_TOT: int = 15

# ── Risk model thresholds ─────────────────────────────────────────────────────
TEMP_INACTIVE_THRESHOLD_C: float = 15.0  # below this → Inactive for most models

# ── Rolling-average windows ───────────────────────────────────────────────────
ROLLING_MAP: dict[str, tuple[str, int]] = {
    "rh_above_90_night_14d_ma": ("nhours_rh_above_90", 14),
    "rh_above_80_day_30d_ma":   ("hours_rh_above_80_day", 30),
    "air_temp_min_c_21d_ma":    ("air_temp_min_c", 21),
    "air_temp_max_c_30d_ma":    ("air_temp_max_c", 30),
    "air_temp_avg_c_30d_ma":    ("air_temp_avg_c", 30),
    "rh_max_30d_ma":            ("rh_max", 30),
    "max_ws_30d_ma":            ("max_ws", 30),
    "dp_min_30d_c_ma":          ("min_dp_c", 30),
}

IBM_ROLLING_MAP: dict[str, int] = {
    "temperature_max":       30,
    "temperature_mean":      30,
    "temperatureDewPoint_min": 30,
    "relativeHumidity_max":  30,
    "windSpeed_max":         30,
    "hours_rh90_night":      14,
    "hours_rh80_allday":     30,
    "temperature_min":       21,
}

# ── Column schemas ────────────────────────────────────────────────────────────
DESIRED_MEASUREMENT_COLS: list[str] = [
    "station_id", "date",
    "rh_above_90_night_14d_ma", "rh_above_80_day_30d_ma",
    "air_temp_min_c_21d_ma", "air_temp_max_c_30d_ma", "air_temp_avg_c_30d_ma",
    "rh_max_30d_ma", "max_ws_30d_ma", "dp_min_30d_c_ma",
]

FINAL_COLUMNS: list[str] = [
    "station_id", "date", "forecasting_date", "station_name", "city", "county",
    "latitude", "longitude", "region", "state", "station_timezone",
    "tarspot_risk", "tarspot_risk_class",
    "gls_risk", "gls_risk_class",
    "fe_risk", "fe_risk_class",
    "whitemold_nirr_risk", "whitemold_nirr_risk_class",
    "whitemold_irr_30in_risk", "whitemold_irr_15in_risk",
    "whitemold_irr_15in_class", "whitemold_irr_30in_class",
    # biomass (present only when planting_date / termination_date are supplied)
    "cgdd_60d_ap", "rain_60d_ap", "cgdd_60d_bt",
    "biomass_lb_acre", "biomass_color", "biomass_message",
]

STATION_META_COLS: list[str] = [
    "station_id", "station_name", "city", "county",
    "latitude", "longitude", "region", "state", "station_timezone",
]

BIOMASS_COLS: list[str] = [
    "cgdd_60d_ap", "rain_60d_ap", "cgdd_60d_bt",
    "biomass_lb_acre", "biomass_color", "biomass_message",
]