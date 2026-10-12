"""
HydroTrace-AI Maritime Command Platform - Pydantic v2 Schemas & Data Contracts.
Unifies data structures across Vision AI (Agent 1), Hydrodynamic Hindcast (Agent 2),
and AIS Attribution & Dark Vessel Detection (Agent 3).
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr


# ==========================================
# Authentication & RBAC Schemas
# ==========================================

class UserRole(str, Enum):
    VIEWER = "VIEWER"
    MARITIME_ANALYST = "MARITIME_ANALYST"
    COAST_GUARD_COMMANDER = "COAST_GUARD_COMMANDER"
    FORENSIC_AUDITOR = "FORENSIC_AUDITOR"


class UserProfile(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: str
    role: UserRole
    organization: str = "Indian Coast Guard / Maritime Law Enforcement"
    mfa_enabled: bool = True
    permissions: List[str]


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class MFAVerifyRequest(BaseModel):
    user_id: str
    totp_code: str = Field(..., min_length=6, max_length=6, pattern="^[0-9]{6}$")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: UserProfile


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# ==========================================
# Agent 1: Sentinel-1 Ingestion & Slick Detection
# ==========================================

class IngestionStatus(str, Enum):
    ACQUIRING = "Acquiring"
    CALIBRATING = "Calibrating"
    SEGMENTING = "Segmenting"
    HINDCASTING = "Hindcasting"
    COMPLETE = "Complete"
    FLAGGED_LOOKALIKE = "Flagged Lookalike"


class SeverityLevel(str, Enum):
    CRITICAL = "Critical"     # > 50 km²
    HIGH = "High Risk"        # 10 - 50 km²
    MEDIUM = "Medium"         # 2 - 10 km²
    LOW = "Low"               # < 2 km²
    LOOKALIKE = "Lookalike"   # Biogenic / Natural feature


class SlickCentroid(BaseModel):
    latitude: float
    longitude: float


class SlickPhysicalMetrics(BaseModel):
    area_km2: float
    perimeter_km: float
    mean_backscatter_db: float
    estimated_age_hours: float
    volume_m3: Optional[float] = None
    wind_damping_ratio: Optional[float] = None


class SatelliteMetadata(BaseModel):
    satellite: str = "Sentinel-1A"
    sensor: str = "C-SAR (VV + VH)"
    orbit_number: int
    pass_direction: str
    acquisition_time_utc: str
    incidence_angle_deg: float = 38.5


class IncidentSummaryCard(BaseModel):
    slick_id: str
    timestamp_utc: str
    port_authority: str
    eez_zone: str
    centroid: SlickCentroid
    area_km2: float
    status: IngestionStatus
    severity: SeverityLevel
    confidence_score: float
    suspect_count: int


# ==========================================
# Agent 2: Hydrodynamic Hindcast & Particles
# ==========================================

class SimulationDirection(str, Enum):
    HINDCAST_REWIND = "HINDCAST_REWIND"      # -dt
    FORWARD_BOOM = "FORWARD_BOOM"            # +dt


class UncertaintyConeData(BaseModel):
    centroid: SlickCentroid
    semi_major_axis_km: float
    semi_minor_axis_km: float
    orientation_degrees: float
    origin_polygon: Dict[str, Any]  # GeoJSON Polygon


class WeatherConditions(BaseModel):
    avg_current_knots: float
    avg_wind_knots: float
    wind_direction_deg: float
    sea_surface_temp_c: float


class ParticleStep(BaseModel):
    particle_id: int
    lat: float
    lon: float
    time_offset_hours: float
    current_u: float
    current_v: float


class HindcastSimulationPayload(BaseModel):
    slick_id: str
    mode: SimulationDirection
    start_utc: str
    end_utc: str
    time_steps_minutes: int
    weather: WeatherConditions
    uncertainty_cone: UncertaintyConeData
    ensemble_path: Dict[str, Any]       # GeoJSON LineString
    stochastic_particles: List[Dict[str, Any]] # Array of particle coordinates or LineStrings


# ==========================================
# Agent 3: AIS Attribution & Dark Vessels
# ==========================================

class VesselIdentityCard(BaseModel):
    imo: str
    mmsi: str
    name: str
    flag: str
    vessel_type: str
    owner: str
    operator: Optional[str] = None
    dwt: Optional[int] = None
    build_year: Optional[int] = None


class KinematicViolationsDetail(BaseModel):
    speed_drop_knots: float
    loiter_duration_mins: float
    ais_blackout_duration_mins: float
    transit_speed_knots: float
    cruising_speed_knots: float
    tortuosity_index: float
    heading_variance_deg: float
    speed_profile: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sparkline series points: [{timestamp, speed_knots, in_cone}]"
    )


class SuspectVesselDossier(BaseModel):
    rank: int
    attribution_score: float  # 0 to 100%
    vessel_identity: VesselIdentityCard
    kinematics: KinematicViolationsDetail
    track_geometry: Dict[str, Any] # GeoJSON LineString
    is_culprit: bool = False
    has_dark_gap: bool = False
    dark_gap_details: Optional[str] = None
    forensic_summary: str
    score_breakdown: Dict[str, float] = Field(
        default_factory=lambda: {
            "spatial_proximity": 0.0,
            "temporal_alignment": 0.0,
            "speed_anomaly": 0.0,
            "ais_blackout_penalty": 0.0,
            "heading_tortuosity": 0.0
        }
    )


# ==========================================
# Unified Platform Incident Response Model
# ==========================================

class UnifiedIncidentPayload(BaseModel):
    slick_id: str
    timestamp_utc: str
    eez_zone: str
    port_authority: str
    status: IngestionStatus
    severity: SeverityLevel
    satellite_metadata: SatelliteMetadata
    slick_geometry: Dict[str, Any]   # GeoJSON Polygon
    centroid: SlickCentroid
    metrics: SlickPhysicalMetrics
    confidence_score: float
    hindcast: HindcastSimulationPayload
    ranked_suspects: List[SuspectVesselDossier]
    prime_culprit: Optional[SuspectVesselDossier] = None
    sha256_hash: str


# ==========================================
# Court-Admissible Audit & DevSecOps Schemas
# ==========================================

class CryptographicAuditSeal(BaseModel):
    incident_uuid: str
    case_file_number: str
    generated_at_utc: str
    issuing_authority: str
    digital_digest_sha256: str
    pgp_key_fingerprint: str
    evidence_payload_verified: bool = True
    qr_verification_url: str


class AuditLogRecord(BaseModel):
    timestamp_utc: str
    user_id: str
    username: str
    user_role: UserRole
    action: str
    resource: str
    ip_address: str
    payload_sha256: str
    status: str
