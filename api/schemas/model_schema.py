from pydantic import BaseModel
from typing import List

class DiseaseModelInfo(BaseModel):
    name: str
    description: str
    variables: List[str]
    model_type: str
    risk_output: str | None = None
    inactive_rule: str | None = None
    version: str