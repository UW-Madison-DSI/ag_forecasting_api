from typing import Any
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

"""
Schema from
https://wisconet.wisc.edu/docs
"""


BASE_URL = "https://wisconet.wisc.edu/api/v1"


class Field(BaseModel):
    id: int
    collection_frequency: str | None
    conversion_type: str | None
    data_type: str | None
    final_units: str | None
    measure_type: str | None
    qualifier: str | None
    sensor: str | None
    source_field: str | None
    source_units: str | None
    standard_name: str | None
    units_abbrev: str | None
    use_for: str | None


class Station(BaseModel):
    id: int
    elevation: float | None
    latitude: float | None
    longitude: float | None
    city: str | None
    county: str | None
    location: str | None
    region: str | None
    state: str | None
    station_id: str | None
    station_name: str | None
    station_slug: str | None
    station_timezone: str | None
    earliest_api_date: datetime | None
    days_active: float | None
    campbell_cloud_id: str | None
    legacy_id: str | None


class ShortMeasure(BaseModel):
    station_id: str
    standard_name: str
    suffix: int
    value: Any
    collection_time: int
    preceding_value: Any
    preceding_time: int


class ShortSummary(BaseModel):
    station: Station
    latest_collection: int
    daily: ShortMeasure
    current: ShortMeasure
    hourly: ShortMeasure


class StationStatus(BaseModel):
    message: str
    station: Station
    field_counts: Any | None
    latest_date: str
    hours_since_last_collection: int
    status: str
    latest_collection_time: int


class AnnotatedMeasure(BaseModel):
    standard_name: str
    value: Any
    preceding_time: int
    suffix: str
    field: Field
    station_id: str
    preceding_value: Any
    collection_time: int


class DataByTime(BaseModel):
    collection_time: int
    measures: list[list[str | int | float]]


class BulkMeasures(BaseModel):
    fieldlist: list[Field]
    data: list[DataByTime]


class SimpleValue(BaseModel):
    field: str
    units: Any


class CollectionTimeByField(BaseModel):
    field: Field
    earliest_collection_time: int
    latest_collection_time: int


class CollectionTimes(BaseModel):
     byField: CollectionTimeByField
     earliest_collection_time: int
     latest_collection_time: int
