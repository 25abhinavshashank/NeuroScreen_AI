from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class RouteType(str, Enum):
    STANDARD_REVIEW = "STANDARD_REVIEW"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    OOD_REVIEW = "OOD_REVIEW"
    ERROR = "ERROR"


class SafetyStatus(str, Enum):
    SCREENING_ONLY = "SCREENING_ONLY"
    BLOCKED = "BLOCKED"


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int
