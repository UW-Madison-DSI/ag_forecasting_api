from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime


class StandardMeasure(BaseModel):
    fieldname: str
    #disease: str
    measure: str
    frequency: Optional[str] = None
    units: Optional[str] = None
    aggregation: Optional[str] = None
    data_type: str  # STR | INT | FLOAT
    possible_values: Optional[List[str]] = None


class Coordinates(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    elevation: Optional[float] = None


class Station(BaseModel):
    station_name: str
    station_id: str
    coordinates: Coordinates
    city: Optional[str] = None
    county: Optional[str] = None
    region: Optional[str] = None
    state: Optional[str] = None
    timezone: Optional[str] = None


class FieldValue(BaseModel):
    fieldname: str
    value: Union[float, int, str, None]


class Observation(BaseModel):
    date: datetime
    data: List[FieldValue]


class Feature(BaseModel):
    station: Station
    timeseries: List[Observation]


class FeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    fields: List[StandardMeasure]
    features: List[Feature]