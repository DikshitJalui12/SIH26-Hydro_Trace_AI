"""
HydroTrace-AI Maritime Command Platform - Configuration & System Settings.
DevSecOps Configuration: Strict security tokens, CORS, CSP directives, and EEZ bounds.
"""

from typing import List, Dict, Any
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "HydroTrace-AI Maritime Command Platform"
    APP_VERSION: str = "4.0.0-PROD"
    ENV: str = "production"
    DEBUG: bool = False
    
    # Server Host & Port
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Cryptographic & Authentication Secrets
    SECRET_KEY: str = "c7d42e09a34bfa852934ecff79a10948b812ef6c4398e09f5bc3a4e9081e7d23"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    MFA_ISSUER_NAME: str = "HydroTrace-AI Naval Command"

    # Coast Guard Forensic Signing Key ID
    FORENSIC_PGP_KEY_ID: str = "ICG-KEY-2026-SHA256-ED25519-HQ-NEWDELHI"
    FORENSIC_COMMAND_TITLE: str = "INDIAN COAST GUARD MARITIME FORENSIC COMMAND"

    # Rate Limiting (Token Bucket)
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 180

    # CORS Whitelist
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Content Security Policy (CSP) Directives
    CSP_DIRECTIVES: Dict[str, str] = {
        "default-src": "'self'",
        "script-src": "'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com https://cdn.jsdelivr.net",
        "style-src": "'self' 'unsafe-inline' https://fonts.googleapis.com https://unpkg.com https://cdn.jsdelivr.net",
        "font-src": "'self' https://fonts.gstatic.com data:",
        "img-src": "'self' data: blob: https://*.tile.openstreetmap.org https://services.arcgisonline.com https://server.arcgisonline.com https://*.arcgisonline.com https://*.basemaps.cartocdn.com",
        "connect-src": "'self' ws://localhost:8000 ws://127.0.0.1:8000 wss://* http://* https://* blob: data:",
        "worker-src": "'self' blob:",
        "object-src": "'none'",
        "frame-ancestors": "'none'",
        "base-uri": "'self'"
    }

    # Indian EEZ Quadrant Boundaries (Bounding boxes for Quick-Zoom)
    INDIAN_EEZ_QUADRANTS: Dict[str, Dict[str, Any]] = {
        "arabian_sea": {
            "name": "Arabian Sea (West EEZ)",
            "center": [15.5, 68.5],
            "zoom": 6.2,
            "bounds": [[8.0, 62.0], [23.5, 74.0]],
            "critical_chokepoints": ["Gulf of Kutch", "Mumbai High Offing", "Goa Coastal Corridor"]
        },
        "bay_of_bengal": {
            "name": "Bay of Bengal (East EEZ)",
            "center": [14.0, 85.5],
            "zoom": 6.0,
            "bounds": [[9.0, 80.0], [21.5, 90.0]],
            "critical_chokepoints": ["Palk Strait", "Paradip Anchorage", "Haldia Approach"]
        },
        "lakshadweep": {
            "name": "Lakshadweep Sea & Nine Degree Channel",
            "center": [10.5, 72.8],
            "zoom": 7.0,
            "bounds": [[8.0, 71.0], [13.0, 74.5]],
            "critical_chokepoints": ["Nine Degree Channel", "Minicoy Fairway", "Cochin Approach"]
        },
        "andaman_sea": {
            "name": "Andaman & Nicobar Islands (Six Degree Channel)",
            "center": [9.0, 93.5],
            "zoom": 6.5,
            "bounds": [[6.0, 91.5], [14.0, 95.0]],
            "critical_chokepoints": ["Six Degree Channel", "Malacca Strait Gateway", "Ten Degree Channel"]
        }
    }

    model_config = {"env_file": ".env", "extra": "allow"}


settings = Settings()
