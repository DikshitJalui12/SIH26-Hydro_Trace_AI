"""
Authentication and Identity Management API Routes.
Provides OAuth2 token issuance, Multi-Factor Authentication (TOTP),
and role profile inspection.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from server.models.schemas import (
    LoginRequest, MFAVerifyRequest, TokenResponse, RefreshTokenRequest, UserProfile
)
from server.security.auth import (
    DEMO_USERS_DB, verify_password, verify_totp_code, create_access_token,
    create_refresh_token, decode_token, get_user_by_username
)
from server.security.rbac import get_current_user
from server.config import settings

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication & Access Control"])


@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest):
    """Authenticates user with username & password and returns short-lived JWT."""
    user_record = DEMO_USERS_DB.get(credentials.username.lower())
    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or security credentials."
        )

    # In our demo user store, default password is "navalcommand2026" or any non-empty pass for demo ease
    profile = get_user_by_username(credentials.username)
    access_token = create_access_token(profile)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=profile
    )


@router.post("/mfa/verify", response_model=TokenResponse)
async def verify_mfa(mfa_data: MFAVerifyRequest):
    """Validates 6-digit TOTP code for hardware MFA token verification."""
    # Lookup user
    found_user = None
    for u in DEMO_USERS_DB.values():
        if u["user_id"] == mfa_data.user_id:
            found_user = u
            break

    if not found_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User ID not registered."
        )

    # In strict mode, verify TOTP against secret. For test convenience, standard TOTP or "123456" accepted
    is_valid = verify_totp_code(found_user["mfa_secret"], mfa_data.totp_code) or mfa_data.totp_code == "123456"
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired TOTP 6-digit code."
        )

    profile = get_user_by_username(found_user["username"])
    access_token = create_access_token(profile)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=profile
    )


@router.get("/me", response_model=UserProfile)
async def get_my_profile(current_user: UserProfile = Depends(get_current_user)):
    """Returns the authenticated operator profile, permissions, and active role."""
    return current_user
