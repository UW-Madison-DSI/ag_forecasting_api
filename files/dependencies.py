"""
Shared FastAPI dependencies.

Cross-cutting concerns — date validation, model filtering, response
format selection — are expressed as ``Depends()`` callables so:
  - Each router endpoint is ~5 lines of business logic.
  - Validation is tested once, in isolation, not duplicated per route.
  - Changing the max date range or adding a new format is a one-line edit here.
"""

from datetime import date
from typing import Annotated, Literal, Optional

from fastapi import Depends, HTTPException, Query

from app_v1.registry import ALL_MODEL_IDS, validate_models

MAX_DATE_RANGE_DAYS = 90


# ── Resolved date range ────────────────────────────────────────────────────────

class DateRange:
    """Validated date range injected into endpoint handlers."""
    def __init__(self, start: str, end: str):
        self.start = start   # YYYY-MM-DD
        self.end   = end     # YYYY-MM-DD
        self.days  = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1


def _parse(value: str, param: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid date for '{param}': '{value}'. Use YYYY-MM-DD.",
        )


def resolve_date_range(
    date:  Optional[str] = Query(None, description="Single date (YYYY-MM-DD)."),
    start: Optional[str] = Query(None, description="Range start (YYYY-MM-DD)."),
    end:   Optional[str] = Query(None, description="Range end (YYYY-MM-DD)."),
) -> DateRange:
    """
    Resolve ``date`` (single day) or ``start``/``end`` (range) into a
    ``DateRange``, enforcing the 90-day maximum.
    """
    if date:
        d = _parse(date, "date")
        return DateRange(str(d), str(d))

    if not start or not end:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'date' or both 'start' and 'end'.",
        )

    s = _parse(start, "start")
    e = _parse(end,   "end")

    if e < s:
        raise HTTPException(status_code=422, detail="'end' must be >= 'start'.")
    if (e - s).days > MAX_DATE_RANGE_DAYS:
        raise HTTPException(
            status_code=422,
            detail=f"Date range may not exceed {MAX_DATE_RANGE_DAYS} days.",
        )
    return DateRange(str(s), str(e))


def resolve_models(
    model: Optional[str] = Query(
        None,
        description="Model ID or comma-separated list (e.g. 'tarspot,gls'). Omit for all models.",
    ),
) -> list[str]:
    """Validate and return the requested model IDs (defaults to all)."""
    if not model:
        return list(ALL_MODEL_IDS)
    requested = [m.strip() for m in model.split(",") if m.strip()]
    try:
        return validate_models(requested)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def resolve_format(
    format: Literal["geojson", "json"] = Query(
        "geojson",
        description="Response format. 'geojson' (default) or 'json' (legacy flat array).",
    ),
) -> str:
    return format


def resolve_include_meta(
    include: Optional[str] = Query(
        None,
        description="Pass 'station_meta' to embed full station fields in each GeoJSON Feature.",
    ),
) -> bool:
    return include == "station_meta"


# ── Type aliases ───────────────────────────────────────────────────────────────

DateRangeDep   = Annotated[DateRange,  Depends(resolve_date_range)]
ModelsDep      = Annotated[list[str],  Depends(resolve_models)]
FormatDep      = Annotated[str,        Depends(resolve_format)]
IncludeMetaDep = Annotated[bool,       Depends(resolve_include_meta)]
