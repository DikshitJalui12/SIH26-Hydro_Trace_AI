"""
Incident & Sentinel-1 Ingestion API Routes.
Provides search, filtering, and detailed multi-agent forensic inspection.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query, Depends
from server.models.schemas import IncidentSummaryCard, UnifiedIncidentPayload, UserProfile
from server.services.incident_service import incident_service
from server.security.rbac import RequireViewer
from server.config import settings

router = APIRouter(prefix="/api/v1/incidents", tags=["Incidents & Ingestion Queue"])


@router.get("", response_model=List[IncidentSummaryCard])
async def list_incidents(
    eez_zone: Optional[str] = Query(None, description="Filter by Indian EEZ zone or Chokepoint"),
    severity: Optional[str] = Query(None, description="Filter by severity (Critical, High Risk, etc.)"),
    search: Optional[str] = Query(None, description="Search query across UUID or Port Authority"),
    current_user: UserProfile = RequireViewer
):
    """Retrieves real-time ingestion queue and live Sentinel-1 satellite acquisition passes."""
    return incident_service.list_incidents(eez_zone=eez_zone, severity=severity, search=search)


@router.get("/eez/quadrants")
async def get_eez_quadrants(current_user: UserProfile = RequireViewer):
    """Returns quick-zoom bounding boxes for the 4 Indian EEZ operational quadrants."""
    return {
        "status": "success",
        "quadrants": settings.INDIAN_EEZ_QUADRANTS
    }


@router.get("/{slick_id}", response_model=UnifiedIncidentPayload)
async def get_incident_detail(
    slick_id: str,
    current_user: UserProfile = RequireViewer
):
    """Fetches full unified forensic payload combining Agent 1, Agent 2, and Agent 3 results."""
    incident = incident_service.get_incident_by_id(slick_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{slick_id}' not found in active operational registry."
        )
    return incident
