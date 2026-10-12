"""
HydroTrace-AI Maritime Command Platform
Live AIS Fleet Radar API Routes.
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status, Depends
from server.models.schemas import UserProfile
from server.services.live_ais_service import live_ais_service
from server.security.rbac import RequireViewer

router = APIRouter(prefix="/api/v1/ais", tags=["Live AIS Fleet Radar"])


@router.get("/live-fleet")
async def get_live_fleet(
    min_risk: Optional[float] = Query(None, description="Minimum risk score filter [0.0 - 1.0]"),
    search: Optional[str] = Query(None, description="Filter vessels by MMSI, IMO, name, or flag"),
    current_user: UserProfile = RequireViewer,
) -> List[Dict[str, Any]]:
    """Returns dynamic real-time AIS vessel movements advancing with live kinematics."""
    if search:
        return live_ais_service.search_fleet(search)
    return live_ais_service.get_live_fleet(min_risk=min_risk)


@router.get("/vessel/{mmsi}")
async def get_vessel_telemetry(
    mmsi: str,
    current_user: UserProfile = RequireViewer,
) -> Dict[str, Any]:
    """Returns real-time kinematic telemetry and anomaly profile for a single vessel."""
    vessel = live_ais_service.get_vessel_by_mmsi(mmsi)
    if not vessel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vessel with MMSI '{mmsi}' not detected in live AIS radar zone."
        )
    return vessel
