"""
HydroTrace-AI Maritime Command Platform
Live Metocean & Satellite Intel Routes.
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Query, Depends
from server.models.schemas import UserProfile
from server.services.live_metocean_service import live_metocean_service
from server.security.rbac import RequireViewer

router = APIRouter(prefix="/api/v1/weather", tags=["Live Metocean & Satellite Intel"])


@router.get("/live")
async def get_live_weather(
    lat: float = Query(1.2897, description="Latitude"),
    lon: float = Query(103.8501, description="Longitude"),
    current_user: UserProfile = RequireViewer,
) -> Dict[str, Any]:
    """Fetches real-time marine weather conditioning, wave state, winds, and surface currents."""
    return live_metocean_service.get_live_marine_weather(lat=lat, lon=lon)


@router.get("/satellite/catalog")
async def get_satellite_catalog(
    current_user: UserProfile = RequireViewer,
) -> List[Dict[str, Any]]:
    """Returns available satellite imagery categories (Sentinel-1 SAR, Sentinel-2 MSI, Optical RGB, False Color IR)."""
    return live_metocean_service.get_satellite_layer_catalog()
