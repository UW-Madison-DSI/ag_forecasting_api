"""Pydantic response schemas for the /sites resource."""

from typing import Any, Literal, Optional
from pydantic import BaseModel


class StationProperties(BaseModel):
    station_id: str
    name:       str
    city:       str
    county:     str
    region:     str
    state:      str
    timezone:   str


class StationFeature(BaseModel):
    type:       Literal["Feature"] = "Feature"
    id:         str
    geometry:   dict[str, Any]
    properties: StationProperties


class StationFeatureCollection(BaseModel):
    type:     Literal["FeatureCollection"] = "FeatureCollection"
    features: list[StationFeature]
