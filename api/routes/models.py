from fastapi import APIRouter, HTTPException
from api.models.disease_metadata import DISEASE_MODELS
from api.schemas.model_schema import DiseaseModelInfo

router = APIRouter()

@router.get("/models")
def list_models():
    return {"available_models": list(DISEASE_MODELS.keys())}

@router.get("/models/{model_name}", response_model=DiseaseModelInfo)
def get_model_info(model_name: str):
    model = DISEASE_MODELS.get(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model