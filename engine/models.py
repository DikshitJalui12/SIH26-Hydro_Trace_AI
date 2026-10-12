"""
Unified data models and schemas for the SIH26 Maritime Forensic System:
- Reverse Lagrangian Drift Engine (Agent 2)
- AIS Attribution & Dark Vessel Detection Engine (Agent 3)
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


# ==========================================
# Agent 2: Reverse Lagrangian Drift Engine Models
# ==========================================

class DriftMode(str, Enum):
    REVERSE_HINDCAST = "REVERSE_HINDCAST"
    FORWARD_FORECAST = "FORWARD_FORECAST"


class CentroidCoord(BaseModel):
    latitude: float = Field(..., description="Latitude in decimal degrees (WGS84)")
    longitude: float = Field(..., description="Longitude in decimal degrees (WGS84)")


class OriginUncertaintyCone(BaseModel):
    centroid: CentroidCoord
    semi_major_axis_km: float = Field(..., description="95% confidence semi-major axis in kilometers")
    semi_minor_axis_km: float = Field(..., description="95% confidence semi-minor axis in kilometers")
    orientation_degrees: float = Field(..., description="Ellipse orientation angle in degrees clockwise from North")
    origin_polygon: Dict[str, Any] = Field(..., description="GeoJSON Polygon of the 95% confidence boundary")


class HindcastWindow(BaseModel):
    start_utc: str = Field(..., description="Start of simulation window (earlier time, e.g. estimated t_0)")
    end_utc: str = Field(..., description="End of simulation window (satellite detection time T)")
    step_minutes: int = Field(default=15, description="Integration time step in minutes (default: 15)")


class WeatherSummary(BaseModel):
    avg_current_knots: float = Field(..., description="Spatiotemporally averaged ocean surface current in knots")
    avg_wind_knots: float = Field(..., description="Spatiotemporally averaged 10m wind speed in knots")
    sea_surface_temp: float = Field(..., description="Average sea surface temperature in Celsius")


class DriftOutputPayload(BaseModel):
    slick_id: str = Field(..., description="Unique identifier matching Agent 1 slick detection")
    mode: DriftMode = Field(default=DriftMode.REVERSE_HINDCAST, description="Drift simulation mode")
    hindcast_window: HindcastWindow
    origin_uncertainty_cone: OriginUncertaintyCone
    drift_path_linestring: Dict[str, Any] = Field(
        ..., description="GeoJSON FeatureCollection containing RK4 particle trajectories and ensemble path"
    )
    weather_summary: WeatherSummary
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Engine physics parameters and metadata")

    def to_ais_corridor_dict(self, delta_t_hours: float = 1.5) -> Dict[str, Any]:
        """Convert Agent 2 hindcast output into input format required by Agent 3 AIS attribution."""
        # 1 km = 0.539957 nautical miles
        radius_nm = round(self.origin_uncertainty_cone.semi_major_axis_km * 0.539957, 2)
        return {
            "slick_id": self.slick_id,
            "center_lat": self.origin_uncertainty_cone.centroid.latitude,
            "center_lon": self.origin_uncertainty_cone.centroid.longitude,
            "radius_nm": radius_nm,
            "t_origin": self.hindcast_window.start_utc,
            "delta_t_hours": delta_t_hours,
            "slick_detection_time": self.hindcast_window.end_utc
        }


# ==========================================
# Agent 3: AIS Attribution & Dark Vessel Models
# ==========================================

class AISMessage(BaseModel):
    timestamp: datetime
    mmsi: str
    lat: float
    lon: float
    sog: float = Field(..., description="Speed over ground in knots")
    cog: float = Field(..., description="Course over ground in degrees")
    heading: Optional[float] = Field(None, description="True heading in degrees")
    nav_status: Optional[int] = Field(0, description="AIS Navigational Status code")
    imo: Optional[str] = None
    vessel_name: Optional[str] = None
    vessel_type: Optional[str] = None


class SpatioTemporalCorridor(BaseModel):
    slick_id: str = Field(..., description="Identifier matching Agent 1 & Agent 2")
    center_lat: float
    center_lon: float
    radius_nm: float = Field(..., description="Uncertainty radius in nautical miles")
    t_origin: datetime = Field(..., description="Estimated discharge origin timestamp (t_0)")
    delta_t_hours: float = Field(..., description="Uncertainty window +/- delta_t in hours")
    slick_detection_time: Optional[datetime] = None
    slick_area_sqkm: Optional[float] = None
    estimated_volume_m3: Optional[float] = None


class VesselIdentity(BaseModel):
    imo: str
    mmsi: str
    name: str
    flag: str
    vessel_type: str
    owner: str
    operator: Optional[str] = None
    dwt: Optional[int] = None
    build_year: Optional[int] = None


class KinematicViolations(BaseModel):
    speed_drop_knots: float = Field(..., description="Observed speed drop below cruising speed")
    loiter_duration_mins: float = Field(..., description="Total loitering duration inside or near corridor")
    ais_blackout_duration_mins: float = Field(..., description="Duration of AIS transponder silence")
    transit_speed_knots: Optional[float] = None
    cruising_speed_knots: Optional[float] = None
    tortuosity_index: Optional[float] = None
    heading_variance_deg: Optional[float] = None


class GeoJSONLineString(BaseModel):
    type: str = "LineString"
    coordinates: List[List[float]] = Field(
        ..., description="List of [longitude, latitude, (optional epoch timestamp)]"
    )


class TrackGeometry(BaseModel):
    type: str = "Feature"
    geometry: GeoJSONLineString
    properties: Dict[str, Any] = Field(default_factory=dict)


class RankedSuspect(BaseModel):
    rank: int
    attribution_score: float = Field(..., description="Attribution score in percentage (0–100%)")
    vessel_identity: VesselIdentity
    kinematic_violations: KinematicViolations
    track_geometry: GeoJSONLineString
    forensic_summary: str
    score_components: Optional[Dict[str, float]] = None


class AttributionReport(BaseModel):
    slick_id: str
    candidates_analyzed: int
    ranked_suspects: List[RankedSuspect]
    metadata: Optional[Dict[str, Any]] = None
