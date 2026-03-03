"""
Response formatters.

Converts flat pipeline DataFrames into the Pydantic response schemas the
routers return. Keeping this logic here means:
  - Routers stay thin (validate → pipeline → format → return).
  - Formatters are pure functions, trivially unit-testable without HTTP.
  - Adding a new output format (CSV, msgpack…) only touches this file.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from app_v1.registry import MODELS
from app_v1.schemas.predictions import (
    PredictionFeature,
    PredictionFeatureCollection,
    PredictionProperties,
    ResponseMetadata,
    RiskValues,
    Timeslice,
    WhitemoldValues,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

_CLASS_LEVEL: dict[str, int] = {
    "inactive": 0,
    "1.low": 1,    "low": 1,
    "2.moderate": 2, "moderate": 2,
    "3.high": 3,   "high": 3,
}


def _normalise_class(raw: str | None) -> tuple[str | None, int | None]:
    """
    Map internal '1.Low' / '3.High' strings → ('low', 1) / ('high', 3).
    Returns (None, None) for unrecognised / missing input.
    """
    if not raw:
        return None, None
    key   = raw.lower()
    level = _CLASS_LEVEL.get(key)
    if level == 0:
        return "inactive", 0
    clean = key.split(".")[-1] if "." in key else key
    return clean, level


def _make_metadata(models: list[str]) -> ResponseMetadata:
    return ResponseMetadata(
        models     = models,
        updated_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        units      = {m: MODELS[m]["units"] for m in models},
    )


def _row_to_timeslice(row: dict, model: str) -> Timeslice:
    """Build one Timeslice from a flat DataFrame row dict for *model*."""
    date_str     = str(row.get("date", ""))[:10]
    forecast_str = str(row.get("forecasting_date", ""))[:10]

    if model == "whitemold":
        nirr_class,  nirr_level  = _normalise_class(row.get("whitemold_nirr_risk_class"))
        irr15_class, irr15_level = _normalise_class(row.get("whitemold_irr_15in_class"))
        irr30_class, irr30_level = _normalise_class(row.get("whitemold_irr_30in_class"))
        values = WhitemoldValues(
            non_irrigated  = RiskValues(risk=row.get("whitemold_nirr_risk"),     risk_class=nirr_class,  risk_level=nirr_level),
            irrigated_15in = RiskValues(risk=row.get("whitemold_irr_15in_risk"), risk_class=irr15_class, risk_level=irr15_level),
            irrigated_30in = RiskValues(risk=row.get("whitemold_irr_30in_risk"), risk_class=irr30_class, risk_level=irr30_level),
        )
    elif model == "tarspot":
        rc, rl = _normalise_class(row.get("tarspot_risk_class"))
        values = RiskValues(risk=row.get("tarspot_risk"), risk_class=rc, risk_level=rl)
    elif model == "gls":
        rc, rl = _normalise_class(row.get("gls_risk_class"))
        values = RiskValues(risk=row.get("gls_risk"), risk_class=rc, risk_level=rl)
    elif model == "frogeye":
        rc, rl = _normalise_class(row.get("fe_risk_class"))
        values = RiskValues(risk=row.get("fe_risk"), risk_class=rc, risk_level=rl)
    else:
        values = RiskValues()

    return Timeslice(date=date_str, forecast_date=forecast_str, values=values)


# ── Public formatters ──────────────────────────────────────────────────────────

def to_feature_collection(
    df: pd.DataFrame,
    models: list[str],
    include_meta: bool,
) -> PredictionFeatureCollection:
    """
    Convert a pipeline DataFrame to a GeoJSON FeatureCollection.

    One Feature is emitted per (station, model) pair so clients can filter
    by model without unpacking nested objects.
    """
    features: list[PredictionFeature] = []

    for station_id, group in df.groupby("station_id"):
        first = group.iloc[0]
        rows  = group.to_dict(orient="records")

        for model in models:
            timeseries = [_row_to_timeslice(r, model) for r in rows]

            meta_kwargs = (
                dict(
                    name     = first.get("station_name"),
                    city     = first.get("city"),
                    county   = first.get("county"),
                    region   = first.get("region"),
                    state    = first.get("state"),
                    timezone = first.get("station_timezone"),
                )
                if include_meta else {}
            )

            features.append(PredictionFeature(
                id         = f"{station_id}__{model}",
                geometry   = {"type": "Point", "coordinates": [first.get("longitude"), first.get("latitude")]},
                properties = PredictionProperties(
                    station_id = station_id,
                    model      = model,
                    timeseries = timeseries,
                    **meta_kwargs,
                ),
            ))

    return PredictionFeatureCollection(metadata=_make_metadata(models), features=features)


def to_legacy_json(df: pd.DataFrame, models: list[str]) -> dict[str, Any]:
    """
    Flat array response matching the legacy format=json schema from the gist.
    Preserves backward-compatibility for callers of the old endpoints.
    """
    predictions: list[dict] = []

    for row in df.to_dict(orient="records"):
        base = {
            "station_id":    row.get("station_id"),
            "name":          row.get("station_name"),
            "city":          row.get("city"),
            "county":        row.get("county"),
            "region":        row.get("region"),
            "state":         row.get("state"),
            "timezone":      row.get("station_timezone"),
            "latitude":      row.get("latitude"),
            "longitude":     row.get("longitude"),
            "date":          str(row.get("date", ""))[:10],
            "forecast_date": str(row.get("forecasting_date", ""))[:10],
        }
        for model in models:
            if model == "tarspot":
                rc, rl = _normalise_class(row.get("tarspot_risk_class"))
                predictions.append({**base, "model": "tarspot",  "risk": row.get("tarspot_risk"), "risk_class": rc, "risk_level": rl})
            elif model == "gls":
                rc, rl = _normalise_class(row.get("gls_risk_class"))
                predictions.append({**base, "model": "gls",      "risk": row.get("gls_risk"),     "risk_class": rc, "risk_level": rl})
            elif model == "frogeye":
                rc, rl = _normalise_class(row.get("fe_risk_class"))
                predictions.append({**base, "model": "frogeye",  "risk": row.get("fe_risk"),      "risk_class": rc, "risk_level": rl})
            elif model == "whitemold":
                predictions.append({
                    **base, "model": "whitemold",
                    "non_irrigated_risk":  row.get("whitemold_nirr_risk"),
                    "irrigated_15in_risk": row.get("whitemold_irr_15in_risk"),
                    "irrigated_30in_risk": row.get("whitemold_irr_30in_risk"),
                })

    return {"metadata": _make_metadata(models).model_dump(), "predictions": predictions}
