"""
/api/v1/models

GET /models              → list all available pest models
GET /models/{model_id}   → detailed metadata for one model
"""

from fastapi import APIRouter, HTTPException, Path

from app_v1.registry import ALL_MODEL_IDS, MODELS, get_model
from app_v1.schemas.predictions import ModelDetail, ModelSummary

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    response_model=list[ModelSummary],
    summary="List all available pest models",
)
def list_models():
    return [
        ModelSummary(id=m["id"], name=m["name"], description=m["description"], crop=m["crop"])
        for m in MODELS.values()
    ]


@router.get(
    "/{model_id}",
    response_model=ModelDetail,
    summary="Get metadata for a single pest model",
)
def get_model_detail(
    model_id: str = Path(
        ...,
        description=f"Model identifier. One of: {', '.join(ALL_MODEL_IDS)}.",
    ),
):
    m = get_model(model_id)
    if not m:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Valid options: {', '.join(ALL_MODEL_IDS)}.",
        )
    return ModelDetail(**m)
