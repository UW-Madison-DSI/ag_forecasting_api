"""
/api/v1/sites/{station_id}/measurements

GET /sites/{station_id}/measurements?start_date=&end_date=&sensor=&frequency=

Thin passthrough over pywisconet bulk_measures.
The static field registry lives here, co-located with the only endpoint that uses it.
"""

from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query

from pywisconet.data import (
    CollectionFrequency,
    Field,
    MeasureType,
    Units,
    bulk_measures,
    bulk_measures_to_df,
    filter_fields,
    station_fields,
)

router = APIRouter(tags=["Measurements"])

# ── Static field registry (fallback when per-station endpoint is unavailable) ──
_STATIC_FIELDS: list[Field] = [Field(**f) for f in [
    {"id": 1,  "standard_name": "5min_air_temp_f_avg",             "use_for": "", "measure_type": "Air Temp",          "qualifier": "avg",   "sensor": "", "source_field": "airtemp_c_avg@Table5",        "data_type": "float",   "source_units": "celsius",     "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "5min"},
    {"id": 2,  "standard_name": "60min_air_temp_f_avg",            "use_for": "", "measure_type": "Air Temp",          "qualifier": "avg",   "sensor": "", "source_field": "airtemp_c_avg@Table60",       "data_type": "float",   "source_units": "celsius",     "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "60min"},
    {"id": 3,  "standard_name": "daily_air_temp_f_avg",            "use_for": "", "measure_type": "Air Temp",          "qualifier": "avg",   "sensor": "", "source_field": "WN_calc",                     "data_type": "float",   "source_units": "celsius",     "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "daily"},
    {"id": 4,  "standard_name": "daily_air_temp_f_max",            "use_for": "", "measure_type": "Air Temp",          "qualifier": "max",   "sensor": "", "source_field": "airtemp_c_max@Table24",       "data_type": "float",   "source_units": "celsius",     "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "daily"},
    {"id": 6,  "standard_name": "daily_air_temp_f_min",            "use_for": "", "measure_type": "Air Temp",          "qualifier": "min",   "sensor": "", "source_field": "airtemp_c_min@Table24",       "data_type": "float",   "source_units": "celsius",     "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "daily"},
    {"id": 9,  "standard_name": "5min_dew_point_f_avg",            "use_for": "", "measure_type": "Dew Point",         "qualifier": "avg",   "sensor": "", "source_field": "dewpointtemp_c_avg@Table5",   "data_type": "float",   "source_units": "celsius",     "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "5min"},
    {"id": 10, "standard_name": "60min_dew_point_f_avg",           "use_for": "", "measure_type": "Dew Point",         "qualifier": "avg",   "sensor": "", "source_field": "dewpointtemp_c_avg@Table60",  "data_type": "float",   "source_units": "celsius",     "final_units": "fahrenheit", "units_abbrev": "f",   "conversion_type": "c2f",     "collection_frequency": "60min"},
    {"id": 15, "standard_name": "daily_rain_in_tot",               "use_for": "", "measure_type": "Rain",              "qualifier": "total", "sensor": "", "source_field": "rain_mm_tot@Table24",         "data_type": "float",   "source_units": "millimeters", "final_units": "inches",     "units_abbrev": "in",  "conversion_type": "mm2in",   "collection_frequency": "daily"},
    {"id": 17, "standard_name": "60min_rain_in_tot",               "use_for": "", "measure_type": "Rain",              "qualifier": "total", "sensor": "", "source_field": "rain_mm_tot@Table60",         "data_type": "float",   "source_units": "millimeters", "final_units": "inches",     "units_abbrev": "in",  "conversion_type": "mm2in",   "collection_frequency": "60min"},
    {"id": 19, "standard_name": "60min_relative_humidity_pct_avg", "use_for": "", "measure_type": "Relative Humidity", "qualifier": "avg",   "sensor": "", "source_field": "relhum_avg@Table60",          "data_type": "float",   "source_units": "pct",         "final_units": "pct",        "units_abbrev": "pct", "conversion_type": "",        "collection_frequency": "60min"},
    {"id": 20, "standard_name": "daily_relative_humidity_pct_max", "use_for": "", "measure_type": "Relative Humidity", "qualifier": "max",   "sensor": "", "source_field": "relhum_max@Table24",          "data_type": "float",   "source_units": "pct",         "final_units": "pct",        "units_abbrev": "pct", "conversion_type": "",        "collection_frequency": "daily"},
    {"id": 57, "standard_name": "60min_wind_speed_mph_max",        "use_for": "", "measure_type": "Wind Speed",        "qualifier": "max",   "sensor": "", "source_field": "windspd_ms_3m_max@Table60",   "data_type": "float",   "source_units": "meters/sec",  "final_units": "mph",        "units_abbrev": "mph", "conversion_type": "ms2mph",  "collection_frequency": "60min"},
    {"id": 87, "standard_name": "60min_wind_speed_mph_avg",        "use_for": "", "measure_type": "Wind Speed",        "qualifier": "avg",   "sensor": "", "source_field": "WN_calc",                     "data_type": "float",   "source_units": "meters/sec",  "final_units": "mph",        "units_abbrev": "mph", "conversion_type": "ms2mph",  "collection_frequency": "60min"},
]]

_SENSOR_CRITERIA: dict[str, list] = {
    "ALL":               [MeasureType.RELATIVE_HUMIDITY, MeasureType.AIRTEMP, MeasureType.DEW_POINT, MeasureType.WIND_SPEED],
    "RELATIVE_HUMIDITY": [MeasureType.RELATIVE_HUMIDITY, Units.PCT],
    "AIRTEMP":           [MeasureType.AIRTEMP,           Units.FAHRENHEIT],
    "DEW_POINT":         [MeasureType.DEW_POINT,         Units.FAHRENHEIT],
    "WIND_SPEED":        [MeasureType.WIND_SPEED,        Units.METERSPERSECOND],
}

_RESPONSE_COLS = [
    "collection_time", "collection_time_ct", "hour_ct",
    "value", "id", "collection_frequency",
    "final_units", "measure_type", "qualifier",
    "source_field", "standard_name", "units_abbrev",
]


def _get_fields(station_id: str) -> list[Field]:
    try:
        return station_fields(station_id)
    except Exception:
        return _STATIC_FIELDS


@router.get(
    "/sites/{station_id}/measurements",
    summary="Raw sensor measurements for a single station",
)
def get_measurements(
    station_id: str = Path(..., description="WiscoNet station identifier"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD), assumed US/Central"),
    end_date:   str = Query(..., description="End date (YYYY-MM-DD), assumed US/Central"),
    sensor: Literal["ALL", "AIRTEMP", "DEW_POINT", "WIND_SPEED", "RELATIVE_HUMIDITY"] = Query(
        ..., description="Sensor type to return.",
    ),
    frequency: Literal["MIN5", "MIN60", "DAILY"] = Query(
        ..., description="Collection frequency.",
    ),
):
    criteria    = _SENSOR_CRITERIA[sensor] + [CollectionFrequency[frequency]]
    field_names = filter_fields(_get_fields(station_id), criteria=criteria)

    try:
        response = bulk_measures(station_id, start_date, end_date, field_names)
        df       = bulk_measures_to_df(response)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream WiscoNet error: {exc}")

    df["collection_time_utc"] = pd.to_datetime(df["collection_time"]).dt.tz_localize("UTC")
    df["collection_time_ct"]  = df["collection_time_utc"].dt.tz_convert("US/Central")
    df["hour_ct"]             = df["collection_time_ct"].dt.hour

    available = [c for c in _RESPONSE_COLS if c in df.columns]
    return df[available].to_dict(orient="records")
