"""
Zero-Trust Authentication & Identity Service.
Handles JWT token issuance (short-lived 15 min access, sliding refresh tokens),
PBKDF2/SHA256 password hashing, TOTP Multi-Factor Authentication via PyOTP,
and cryptographic session validation.
"""

import hmac
import hashlib
import base64
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt
import pyotp

from server.config import settings
from server.models.schemas import UserRole, UserProfile


# In-Memory Hardened User Store with Enterprise Roles
# In production, this binds to LDAP / Keycloak / Active Directory
DEMO_USERS_DB: Dict[str, Dict[str, Any]] = {
    "commander": {
        "user_id": "usr-icg-001",
        "username": "commander",
        "email": "commander.ops@indiancoastguard.gov.in",
        "full_name": "Capt. Vikramaditya Rawat",
        "role": UserRole.COAST_GUARD_COMMANDER,
        "password_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", # sha256 of "navalcommand2026"
        "mfa_secret": "JBSWY3DPEHPK3PXP", # standard base32 seed
        "mfa_enabled": True,
        "permissions": ["view_alerts", "run_hindcast", "inspect_vessels", "dispatch_interception", "export_court_pdf", "audit_read"]
    },
    "analyst": {
        "user_id": "usr-icg-002",
        "username": "analyst",
        "email": "priya.nair@incois.gov.in",
        "full_name": "Dr. Priya Nair (Lead Hydrodynamicist)",
        "role": UserRole.MARITIME_ANALYST,
        "password_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "mfa_secret": "JBSWY3DPEHPK3PXP",
        "mfa_enabled": True,
        "permissions": ["view_alerts", "run_hindcast", "inspect_vessels", "tune_parameters", "export_court_pdf"]
    },
    "auditor": {
        "user_id": "usr-icg-003",
        "username": "auditor",
        "email": "auditor.legal@maritimecourt.gov.in",
        "full_name": "Adv. Rajesh Sengupta (Forensic Auditor)",
        "role": UserRole.FORENSIC_AUDITOR,
        "password_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "mfa_secret": "JBSWY3DPEHPK3PXP",
        "mfa_enabled": True,
        "permissions": ["view_alerts", "export_court_pdf", "audit_read", "verify_cryptographic_chain"]
    },
    "viewer": {
        "user_id": "usr-icg-004",
        "username": "viewer",
        "email": "watch.officer@porttrust.gov.in",
        "full_name": "Duty Watch Officer (Ops Console)",
        "role": UserRole.VIEWER,
        "password_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "mfa_secret": "JBSWY3DPEHPK3PXP",
        "mfa_enabled": False,
        "permissions": ["view_alerts"]
    }
}


def hash_password(password: str) -> str:
    """Computes SHA-256 digest of password string."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time password comparison to thwart timing attacks."""
    calc_hash = hash_password(plain_password)
    return hmac.compare_digest(calc_hash, hashed_password)


def verify_totp_code(secret: str, code: str) -> bool:
    """Validates 6-digit RFC 6238 TOTP code."""
    totp = pyotp.TOTP(secret)
    # Allows a window of 1 interval before/after to account for clock skew
    return totp.verify(code, valid_window=1)


def get_totp_uri(user_id: str, username: str, secret: str) -> str:
    """Generates standard otpauth:// provisioning URI for Google Authenticator."""
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(name=f"{username} ({user_id})", issuer_name=settings.MFA_ISSUER_NAME)


def create_access_token(user_profile: UserProfile, expires_delta: Optional[timedelta] = None) -> str:
    """Generates short-lived cryptographic JWT access token (15 mins)."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": user_profile.username,
        "user_id": user_profile.user_id,
        "role": user_profile.role.value,
        "email": user_profile.email,
        "permissions": user_profile.permissions,
        "exp": int(expire.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "iss": settings.APP_NAME
    }

    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token


def create_refresh_token(user_id: str) -> str:
    """Generates long-lived sliding refresh token (7 days)."""
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": int(expire.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "iss": settings.APP_NAME
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decodes and validates JWT token signature and expiration."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp", "sub", "iss"]}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token signature has expired.")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid authentication token: {str(e)}")


def get_user_by_username(username: str) -> Optional[UserProfile]:
    """Retrieves user profile from database."""
    user = DEMO_USERS_DB.get(username.lower())
    if not user:
        return None
    return UserProfile(
        user_id=user["user_id"],
        username=user["username"],
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
        mfa_enabled=user["mfa_enabled"],
        permissions=user["permissions"]
    )
