from __future__ import annotations

import json

from fastapi import APIRouter

from backend.app.core.config import RAG_INDEX_PATH, REGISTRY_PATH, get_settings
from backend.app.schemas.monitoring import ActiveModelSummary, HealthStatus

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    settings = get_settings()

    models: dict[str, ActiveModelSummary] = {}

    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)

        active_models = registry.get("active_models")
        if active_models is None and registry.get("active_model"):
            active_models = {"child": registry["active_model"]}

        for band, entry in (active_models or {}).items():
            models[band] = ActiveModelSummary(
                name=entry.get("name", "unknown"),
                version=entry.get("version", "unknown"),
                status=entry.get("status", "unavailable"),
                age_range_years=entry.get("age_range_years"),
            )

    return HealthStatus(
        status="healthy",
        environment=settings.environment,
        models=models,
        rag_index_available=RAG_INDEX_PATH.exists(),
    )
