from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds baseline security response headers.

    This is a research/local-network prototype, not a hardened public
    service — these headers cover the cheap, uncontroversial defaults
    (MIME sniffing, clickjacking, referrer leakage) without imposing a
    CSP or HSTS policy that would need to be tuned per deployment.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response
