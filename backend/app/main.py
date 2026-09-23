from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes.assessment import router as assessment_router
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.history import router as history_router
from backend.app.api.routes.models import router as models_router
from backend.app.api.routes.monitoring import router as monitoring_router
from backend.app.api.routes.predict import router as predict_router
from backend.app.api.routes.rag import router as rag_router
from backend.app.api.routes.video import router as video_router
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging, get_logger
from backend.app.core.security import SecurityHeadersMiddleware

configure_logging()
logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(
    title="NEUROSCREEN AI",
    version="2.0.0",
    description="Research/educational AI-assisted screening prototype; not a diagnostic system.",
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(assessment_router)
app.include_router(history_router)
app.include_router(models_router)
app.include_router(monitoring_router)
app.include_router(predict_router)
app.include_router(rag_router)
app.include_router(video_router)


@app.get("/")
def root():
    return {"name": "NEUROSCREEN AI", "version": "2.0.0", "status": "running"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
            "status_code": 422,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_wrapper(request: Request, exc: HTTPException):
    logger.warning("HTTP %s on %s: %s", exc.status_code, request.url.path, exc.detail)
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "An unexpected error occurred.",
            "status_code": 500,
        },
    )
