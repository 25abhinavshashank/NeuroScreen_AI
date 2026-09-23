from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend.app.core.config import REGISTRY_PATH
from backend.app.schemas.monitoring import ModelRegistryInfo

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get("", response_model=ModelRegistryInfo)
def get_active_model() -> ModelRegistryInfo:
    if not REGISTRY_PATH.exists():
        raise HTTPException(status_code=503, detail="Model registry not found.")

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    active_models = registry.get("active_models")
    if active_models is None and registry.get("active_model"):
        # Backward compatibility with the older single-model registry.
        active_models = {"child": registry["active_model"]}

    if not active_models:
        raise HTTPException(status_code=503, detail="No active models in registry.")

    return ModelRegistryInfo(active_models=active_models, updated_at=registry.get("updated_at"))
