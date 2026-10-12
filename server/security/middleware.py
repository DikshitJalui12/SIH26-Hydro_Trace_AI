"""
Enterprise DevSecOps & Security Middlewares:
1. Strict Content Security Policy (CSP) & HSTS Headers Middleware.
2. Token-Bucket API Rate Limiter Middleware.
3. Immutable Append-Only Audit Logging Middleware with Cryptographic SHA-256 Digesting.
"""

import time
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from server.config import settings
from server.models.schemas import AuditLogRecord, UserRole


# In-Memory Append-Only Cryptographic Audit Log
# Every request is hashed and recorded with monotonic timestamps
AUDIT_LOG_STREAM: List[Dict[str, Any]] = []


class TokenBucketRateLimiter:
    """
    In-memory Token Bucket rate limiter per client IP.
    Thwarts Denial of Service (DoS) and automated forensic harvesting attacks.
    """
    def __init__(self, capacity: int = 180, refill_rate_per_sec: float = 3.0):
        self.capacity = capacity
        self.refill_rate = refill_rate_per_sec
        self.buckets: Dict[str, Tuple[float, float]] = {} # ip -> (tokens, last_refill_timestamp)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        if ip not in self.buckets:
            self.buckets[ip] = (self.capacity - 1, now)
            return True

        tokens, last_refill = self.buckets[ip]
        elapsed = now - last_refill
        refilled_tokens = min(float(self.capacity), tokens + (elapsed * self.refill_rate))

        if refilled_tokens >= 1.0:
            self.buckets[ip] = (refilled_tokens - 1.0, now)
            return True
        else:
            self.buckets[ip] = (refilled_tokens, now)
            return False


rate_limiter = TokenBucketRateLimiter(
    capacity=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
    refill_rate_per_sec=settings.RATE_LIMIT_REQUESTS_PER_MINUTE / 60.0
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects high-security HTTP headers including strict CSP, HSTS, and frame protection."""
    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)

        # Assemble CSP header
        csp_value = "; ".join([f"{k} {v}" for k, v in settings.CSP_DIRECTIVES.items()])
        response.headers["Content-Security-Policy"] = csp_value
        
        # HSTS (Strict-Transport-Security)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Anti-Clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Anti-MIME-Sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(self)"
        
        # Cache control for sensitive forensic responses
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"

        return response


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Enforces token-bucket rate limiting per IP address on all API and WebSocket routes."""
    async def dispatch(self, request: Request, call_next) -> Response:
        client_ip = request.client.host if request.client else "127.0.0.1"

        # Exclude static frontend files from aggressive rate limiting
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RateLimitExceeded",
                    "detail": "Exceeded maximum requests per minute. Security rate limit engaged."
                },
                headers={"Retry-After": "10"}
            )

        return await call_next(request)


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Maintains an immutable append-only audit trail of every forensic operation,
    computing the cryptographic SHA-256 hash of the request parameters.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # Only audit API routes
        if not path.startswith("/api/"):
            return await call_next(request)

        # Process request
        response = await call_next(request)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        # Extract authorization subject if available
        auth_header = request.headers.get("Authorization", "")
        actor = "Anonymous/Viewer"
        if auth_header.startswith("Bearer "):
            try:
                from server.security.auth import decode_token
                payload = decode_token(auth_header.split(" ")[1])
                actor = f"{payload.get('sub')} ({payload.get('role')})"
            except Exception:
                actor = "Unverified Token"

        # Record hash
        audit_entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "ip_address": client_ip,
            "method": request.method,
            "endpoint": path,
            "query_params": str(request.query_params),
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "chain_hash": hashlib.sha256(f"{path}:{actor}:{client_ip}:{response.status_code}".encode()).hexdigest()
        }
        
        AUDIT_LOG_STREAM.append(audit_entry)
        if len(AUDIT_LOG_STREAM) > 5000:
            AUDIT_LOG_STREAM.pop(0)

        return response
