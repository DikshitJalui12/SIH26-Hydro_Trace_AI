"""
Incident Fusion & Orchestration Service.
Unifies outputs from:
- Agent 1: Sentinel-1 SAR Attention U-Net Slick Detection & Morphology
- Agent 2: Lagrangian Reverse Monte Carlo RK4 Hindcast & Uncertainty Cones
- Agent 3: AIS Attribution, Kinematics & Dark Transponder Blackout Penalties
"""

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

from server.models.schemas import (
    IncidentSummaryCard, UnifiedIncidentPayload, IngestionStatus, SeverityLevel,
    SlickCentroid, SlickPhysicalMetrics, SatelliteMetadata,
    HindcastSimulationPayload, SimulationDirection, UncertaintyConeData, WeatherConditions,
    SuspectVesselDossier, VesselIdentityCard, KinematicViolationsDetail
)


WORKSPACE_ROOT = Path(__file__).parent.parent.parent


class IncidentFusionService:
    """Central repository and orchestration service for maritime incidents."""

    def __init__(self):
        self._incidents: Dict[str, UnifiedIncidentPayload] = {}
        self._incident_summaries: List[IncidentSummaryCard] = []
        self._load_precomputed_scenarios()

    def _load_precomputed_scenarios(self):
        """Loads verified case files from Agent 1, Agent 2, and Agent 3."""
        # 1. Primary Scenario: Malacca Strait Strategic Chokepoint
        malacca_report_path = WORKSPACE_ROOT / "output" / "attribution_report_SLICK-2026-MALACCA-0881.json"
        if malacca_report_path.exists():
            with open(malacca_report_path, "r", encoding="utf-8") as f:
                malacca_raw = json.load(f)
            self._ingest_malacca_scenario(malacca_raw)

        # 2. Secondary Scenario: Arabian Sea / Gulf of Kutch Offing
        hindcast_path = WORKSPACE_ROOT / "output" / "hindcast_12h.json"
        if hindcast_path.exists():
            with open(hindcast_path, "r", encoding="utf-8") as f:
                hindcast_raw = json.load(f)
            self._ingest_arabian_sea_scenario(hindcast_raw)

        # 3. Live Sentinel-1 Ingestion Queue Mock Passes
        self._generate_live_feed_cards()

    def _ingest_malacca_scenario(self, raw: dict):
        slick_id = raw.get("slick_id", "SLICK-2026-MALACCA-0881")
        
        # Center coordinates for Malacca scenario
        center_lat = 1.2100
        center_lon = 103.2700

        # Build Slick Polygon around detection point
        slick_poly = {
            "type": "Polygon",
            "coordinates": [[
                [103.264, 1.205],
                [103.278, 1.212],
                [103.284, 1.218],
                [103.276, 1.216],
                [103.268, 1.210],
                [103.264, 1.205]
            ]]
        }

        # Build 95% uncertainty ellipse polygon
        origin_poly = {
            "type": "Polygon",
            "coordinates": [[
                [103.245, 1.190],
                [103.255, 1.205],
                [103.275, 1.220],
                [103.295, 1.225],
                [103.305, 1.215],
                [103.285, 1.195],
                [103.260, 1.185],
                [103.245, 1.190]
            ]]
        }

        uncertainty_cone = UncertaintyConeData(
            centroid=SlickCentroid(latitude=center_lat, longitude=center_lon),
            semi_major_axis_km=6.0,
            semi_minor_axis_km=3.2,
            orientation_degrees=55.0,
            origin_polygon=origin_poly
        )

        weather = WeatherConditions(
            avg_current_knots=1.45,
            avg_wind_knots=14.2,
            wind_direction_deg=210.0,
            sea_surface_temp_c=28.4
        )

        # Parse ranked suspects
        ranked_suspects: List[SuspectVesselDossier] = []
        for s in raw.get("ranked_suspects", []):
            ident = s.get("vessel_identity", {})
            kine = s.get("kinematic_violations", {})
            track_geom = s.get("track_geometry", {})

            # Synthesize speed drop sparkline profile
            sparkline = []
            coords = track_geom.get("coordinates", [])
            base_speed = kine.get("cruising_speed_knots", 15.0)
            drop_speed = base_speed - kine.get("speed_drop_knots", 0.0)
            
            for idx, c in enumerate(coords[::max(1, len(coords)//15)]):
                # Midpoint dip for culprit
                if s.get("rank") == 1 and 4 <= idx <= 10:
                    spd = drop_speed + (idx % 2) * 0.4
                    in_cone = True
                else:
                    spd = base_speed + ((idx % 3) - 1) * 0.3
                    in_cone = False
                sparkline.append({
                    "timestamp": c[2] if len(c) > 2 else idx,
                    "speed_knots": round(spd, 1),
                    "in_cone": in_cone
                })

            suspect = SuspectVesselDossier(
                rank=s.get("rank", 1),
                attribution_score=s.get("attribution_score", 0.0),
                vessel_identity=VesselIdentityCard(
                    imo=ident.get("imo", "N/A"),
                    mmsi=ident.get("mmsi", "N/A"),
                    name=ident.get("name", "Unknown"),
                    flag=ident.get("flag", "Unknown"),
                    vessel_type=ident.get("vessel_type", "Tanker"),
                    owner=ident.get("owner", "Maritime Entity"),
                    operator=ident.get("operator"),
                    dwt=ident.get("dwt"),
                    build_year=ident.get("build_year")
                ),
                kinematics=KinematicViolationsDetail(
                    speed_drop_knots=kine.get("speed_drop_knots", 0.0),
                    loiter_duration_mins=kine.get("loiter_duration_mins", 0.0),
                    ais_blackout_duration_mins=kine.get("ais_blackout_duration_mins", 0.0),
                    transit_speed_knots=kine.get("transit_speed_knots", 14.0),
                    cruising_speed_knots=kine.get("cruising_speed_knots", 15.0),
                    tortuosity_index=kine.get("tortuosity_index", 1.0),
                    heading_variance_deg=kine.get("heading_variance_deg", 2.0),
                    speed_profile=sparkline
                ),
                track_geometry=track_geom,
                is_culprit=(s.get("rank") == 1),
                has_dark_gap=(kine.get("ais_blackout_duration_mins", 0) > 30),
                dark_gap_details=(
                    f"⚠️ Transponder Silence: {kine.get('ais_blackout_duration_mins')} mins inside corridor (Likely Bilge Dump Event)"
                    if kine.get("ais_blackout_duration_mins", 0) > 30 else None
                ),
                forensic_summary=s.get("forensic_summary", "Suspect transit evaluated against Lagrangian origin cone."),
                score_breakdown={
                    "spatial_proximity": 94.0 if s.get("rank") == 1 else 32.0,
                    "temporal_alignment": 92.5 if s.get("rank") == 1 else 45.0,
                    "speed_anomaly": 88.0 if s.get("rank") == 1 else 15.0,
                    "ais_blackout_penalty": 95.0 if s.get("rank") == 1 else 0.0,
                    "heading_tortuosity": 42.0 if s.get("rank") == 1 else 10.0
                }
            )
            ranked_suspects.append(suspect)

        # Build Lagrangian ensemble path
        ensemble_coords = [
            [103.2700, 1.2100],
            [103.2500, 1.1980],
            [103.2300, 1.1850],
            [103.2050, 1.1710],
            [103.1800, 1.1590]
        ]
        ensemble_path = {
            "type": "LineString",
            "coordinates": ensemble_coords
        }

        # Synthesize stochastic particle cloud for 60 FPS WebGL advection
        stochastic_particles = []
        for i in range(250): # Representative particle tracks
            p_track = []
            jitter_x = ((i % 25) - 12) * 0.003
            jitter_y = ((i // 25) - 5) * 0.002
            for pt in ensemble_coords:
                p_track.append([pt[0] + jitter_x, pt[1] + jitter_y])
            stochastic_particles.append({
                "particle_id": i,
                "coordinates": p_track
            })

        hindcast = HindcastSimulationPayload(
            slick_id=slick_id,
            mode=SimulationDirection.HINDCAST_REWIND,
            start_utc="2026-09-18T02:00:00Z",
            end_utc="2026-09-18T06:30:00Z",
            time_steps_minutes=15,
            weather=weather,
            uncertainty_cone=uncertainty_cone,
            ensemble_path=ensemble_path,
            stochastic_particles=stochastic_particles
        )

        payload = UnifiedIncidentPayload(
            slick_id=slick_id,
            timestamp_utc="2026-09-18T06:30:00Z",
            eez_zone="Andaman Sea Gateway / Malacca Straits",
            port_authority="Port Blair / Singapore Fairway Command",
            status=IngestionStatus.COMPLETE,
            severity=SeverityLevel.CRITICAL,
            satellite_metadata=SatelliteMetadata(
                satellite="Sentinel-1A",
                sensor="C-SAR (Dual-Pol VV/VH)",
                orbit_number=38412,
                pass_direction="DESCENDING",
                acquisition_time_utc="2026-09-18T06:30:00Z",
                incidence_angle_deg=39.2
            ),
            slick_geometry=slick_poly,
            centroid=SlickCentroid(latitude=center_lat, longitude=center_lon),
            metrics=SlickPhysicalMetrics(
                area_km2=14.8,
                perimeter_km=21.4,
                mean_backscatter_db=-24.6,
                estimated_age_hours=4.5,
                volume_m3=45.2,
                wind_damping_ratio=0.82
            ),
            confidence_score=0.965,
            hindcast=hindcast,
            ranked_suspects=ranked_suspects,
            prime_culprit=ranked_suspects[0] if ranked_suspects else None,
            sha256_hash=hashlib.sha256(slick_id.encode()).hexdigest()
        )

        self._incidents[slick_id] = payload
        self._incident_summaries.append(IncidentSummaryCard(
            slick_id=slick_id,
            timestamp_utc="2026-09-18T06:30:00Z",
            port_authority="Port Blair / Singapore Fairway Command",
            eez_zone="Andaman Sea / Malacca Gateway",
            centroid=SlickCentroid(latitude=center_lat, longitude=center_lon),
            area_km2=14.8,
            status=IngestionStatus.COMPLETE,
            severity=SeverityLevel.CRITICAL,
            confidence_score=0.965,
            suspect_count=len(ranked_suspects)
        ))

    def _ingest_arabian_sea_scenario(self, raw: dict):
        slick_id = raw.get("slick_id", "SLICK-SAR-20260918-0042")
        cone_raw = raw.get("origin_uncertainty_cone", {})
        weather_raw = raw.get("weather_summary", {})
        
        c_lat = cone_raw.get("centroid", {}).get("latitude", 22.5622)
        c_lon = cone_raw.get("centroid", {}).get("longitude", 68.2623)

        slick_poly = {
            "type": "Polygon",
            "coordinates": [[
                [68.135, 22.455],
                [68.155, 22.465],
                [68.160, 22.475],
                [68.140, 22.470],
                [68.135, 22.455]
            ]]
        }

        uncertainty_cone = UncertaintyConeData(
            centroid=SlickCentroid(latitude=c_lat, longitude=c_lon),
            semi_major_axis_km=cone_raw.get("semi_major_axis_km", 3.95),
            semi_minor_axis_km=cone_raw.get("semi_minor_axis_km", 2.08),
            orientation_degrees=cone_raw.get("orientation_degrees", 44.37),
            origin_polygon=cone_raw.get("origin_polygon", {})
        )

        weather = WeatherConditions(
            avg_current_knots=weather_raw.get("avg_current_knots", 0.57),
            avg_wind_knots=weather_raw.get("avg_wind_knots", 11.84),
            wind_direction_deg=195.0,
            sea_surface_temp_c=weather_raw.get("sea_surface_temp", 26.5)
        )

        drift_path_features = raw.get("drift_path_linestring", {}).get("features", [])
        ensemble_path = {"type": "LineString", "coordinates": []}
        stochastic_particles = []

        for feat in drift_path_features[:80]: # Sample 80 streamlines
            props = feat.get("properties", {})
            if props.get("type") == "ensemble_mean_path":
                ensemble_path = feat.get("geometry", {})
            else:
                stochastic_particles.append({
                    "particle_id": props.get("particle_id", 0),
                    "coordinates": feat.get("geometry", {}).get("coordinates", [])
                })

        # Synthetic Suspects for Arabian Sea / Gulf of Kutch Offing
        suspects = [
            SuspectVesselDossier(
                rank=1,
                attribution_score=82.4,
                vessel_identity=VesselIdentityCard(
                    imo="9384918",
                    mmsi="419001450",
                    name="KUTCH GLORY",
                    flag="India",
                    vessel_type="Chemical / Products Tanker",
                    owner="Gujarat Maritime Transporters Ltd",
                    operator="Kutch Energy Logistics",
                    dwt=47800,
                    build_year=2016
                ),
                kinematics=KinematicViolationsDetail(
                    speed_drop_knots=8.4,
                    loiter_duration_mins=18.0,
                    ais_blackout_duration_mins=42.0,
                    transit_speed_knots=13.6,
                    cruising_speed_knots=14.5,
                    tortuosity_index=1.12,
                    heading_variance_deg=4.8,
                    speed_profile=[
                        {"timestamp": 0, "speed_knots": 14.5, "in_cone": False},
                        {"timestamp": 1, "speed_knots": 14.2, "in_cone": False},
                        {"timestamp": 2, "speed_knots": 6.1, "in_cone": True},
                        {"timestamp": 3, "speed_knots": 5.8, "in_cone": True},
                        {"timestamp": 4, "speed_knots": 13.9, "in_cone": False}
                    ]
                ),
                track_geometry={
                    "type": "LineString",
                    "coordinates": [
                        [68.10, 22.40, 1789660000.0],
                        [68.18, 22.48, 1789664000.0],
                        [68.25, 22.55, 1789668000.0],
                        [68.32, 22.62, 1789672000.0]
                    ]
                },
                is_culprit=True,
                has_dark_gap=True,
                dark_gap_details="⚠️ Transponder Gap: 42 mins during transit through origin ellipse.",
                forensic_summary="Vessel performed an unannounced 8.4-knot deceleration followed by transponder silence directly coinciding with the 95% Lagrangian hindcast cone.",
                score_breakdown={
                    "spatial_proximity": 91.0,
                    "temporal_alignment": 88.0,
                    "speed_anomaly": 84.0,
                    "ais_blackout_penalty": 90.0,
                    "heading_tortuosity": 59.0
                }
            ),
            SuspectVesselDossier(
                rank=2,
                attribution_score=28.5,
                vessel_identity=VesselIdentityCard(
                    imo="9610992",
                    mmsi="538004210",
                    name="MARSHALL VOYAGER",
                    flag="Marshall Islands",
                    vessel_type="Bulk Carrier",
                    owner="Voyager Maritime Corp",
                    dwt=82000,
                    build_year=2014
                ),
                kinematics=KinematicViolationsDetail(
                    speed_drop_knots=0.5,
                    loiter_duration_mins=0.0,
                    ais_blackout_duration_mins=0.0,
                    transit_speed_knots=12.2,
                    cruising_speed_knots=12.3,
                    tortuosity_index=1.01,
                    heading_variance_deg=0.8,
                    speed_profile=[
                        {"timestamp": 0, "speed_knots": 12.3, "in_cone": False},
                        {"timestamp": 1, "speed_knots": 12.2, "in_cone": False},
                        {"timestamp": 2, "speed_knots": 12.2, "in_cone": False}
                    ]
                ),
                track_geometry={
                    "type": "LineString",
                    "coordinates": [
                        [68.05, 22.35, 1789660000.0],
                        [68.12, 22.42, 1789664000.0],
                        [68.20, 22.50, 1789668000.0]
                    ]
                },
                is_culprit=False,
                has_dark_gap=False,
                forensic_summary="Steady commercial bulk transit at constant cruising speed; cleared of discharge suspicion.",
                score_breakdown={
                    "spatial_proximity": 45.0,
                    "temporal_alignment": 50.0,
                    "speed_anomaly": 5.0,
                    "ais_blackout_penalty": 0.0,
                    "heading_tortuosity": 2.0
                }
            )
        ]

        payload = UnifiedIncidentPayload(
            slick_id=slick_id,
            timestamp_utc="2026-09-18T06:30:00Z",
            eez_zone="Arabian Sea (West EEZ) / Gulf of Kutch Offing",
            port_authority="Kandla / Deendayal Port Trust",
            status=IngestionStatus.COMPLETE,
            severity=SeverityLevel.HIGH,
            satellite_metadata=SatelliteMetadata(
                satellite="Sentinel-1B",
                sensor="C-SAR (VV + VH)",
                orbit_number=29104,
                pass_direction="ASCENDING",
                acquisition_time_utc="2026-09-18T06:30:00Z",
                incidence_angle_deg=41.1
            ),
            slick_geometry=slick_poly,
            centroid=SlickCentroid(latitude=22.465, longitude=68.150),
            metrics=SlickPhysicalMetrics(
                area_km2=8.6,
                perimeter_km=14.2,
                mean_backscatter_db=-22.8,
                estimated_age_hours=12.0,
                volume_m3=28.5,
                wind_damping_ratio=0.78
            ),
            confidence_score=0.912,
            hindcast=HindcastSimulationPayload(
                slick_id=slick_id,
                mode=SimulationDirection.HINDCAST_REWIND,
                start_utc="2026-09-17T18:30:00Z",
                end_utc="2026-09-18T06:30:00Z",
                time_steps_minutes=15,
                weather=weather,
                uncertainty_cone=uncertainty_cone,
                ensemble_path=ensemble_path,
                stochastic_particles=stochastic_particles
            ),
            ranked_suspects=suspects,
            prime_culprit=suspects[0],
            sha256_hash=hashlib.sha256(slick_id.encode()).hexdigest()
        )

        self._incidents[slick_id] = payload
        self._incident_summaries.append(IncidentSummaryCard(
            slick_id=slick_id,
            timestamp_utc="2026-09-18T06:30:00Z",
            port_authority="Kandla / Deendayal Port Authority",
            eez_zone="Arabian Sea / Gulf of Kutch",
            centroid=SlickCentroid(latitude=22.465, longitude=68.150),
            area_km2=8.6,
            status=IngestionStatus.COMPLETE,
            severity=SeverityLevel.HIGH,
            confidence_score=0.912,
            suspect_count=len(suspects)
        ))

    def _generate_live_feed_cards(self):
        """Adds Sentinel-1 real-time acquisition passes to the ingestion queue."""
        feeds = [
            {
                "slick_id": "S1A-PASS-20260919-BOB-014",
                "timestamp_utc": "2026-09-19T04:15:00Z",
                "port_authority": "Paradip Naval Base",
                "eez_zone": "Bay of Bengal (East EEZ)",
                "lat": 19.82,
                "lon": 87.14,
                "area_km2": 32.4,
                "status": IngestionStatus.SEGMENTING,
                "severity": SeverityLevel.HIGH,
                "confidence_score": 0.88,
                "suspect_count": 3
            },
            {
                "slick_id": "S1B-PASS-20260919-LAK-008",
                "timestamp_utc": "2026-09-19T05:22:00Z",
                "port_authority": "Kochi Southern Command",
                "eez_zone": "Lakshadweep / Nine Degree Channel",
                "lat": 9.94,
                "lon": 73.65,
                "area_km2": 54.1,
                "status": IngestionStatus.CALIBRATING,
                "severity": SeverityLevel.CRITICAL,
                "confidence_score": 0.94,
                "suspect_count": 0
            },
            {
                "slick_id": "S1A-LOOKALIKE-20260918-GOA",
                "timestamp_utc": "2026-09-18T22:10:00Z",
                "port_authority": "Mormugao Port Trust",
                "eez_zone": "Arabian Sea / Goa Coast",
                "lat": 15.25,
                "lon": 73.40,
                "area_km2": 4.1,
                "status": IngestionStatus.FLAGGED_LOOKALIKE,
                "severity": SeverityLevel.LOOKALIKE,
                "confidence_score": 0.22,
                "suspect_count": 0
            }
        ]

        for item in feeds:
            self._incident_summaries.append(IncidentSummaryCard(
                slick_id=item["slick_id"],
                timestamp_utc=item["timestamp_utc"],
                port_authority=item["port_authority"],
                eez_zone=item["eez_zone"],
                centroid=SlickCentroid(latitude=item["lat"], longitude=item["lon"]),
                area_km2=item["area_km2"],
                status=item["status"],
                severity=item["severity"],
                confidence_score=item["confidence_score"],
                suspect_count=item["suspect_count"]
            ))

    def list_incidents(
        self,
        eez_zone: Optional[str] = None,
        severity: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[IncidentSummaryCard]:
        """Filters and searches incidents in the ingestion queue."""
        results = self._incident_summaries
        if eez_zone:
            results = [r for r in results if eez_zone.lower() in r.eez_zone.lower()]
        if severity:
            results = [r for r in results if severity.lower() == r.severity.value.lower()]
        if search:
            q = search.lower()
            results = [
                r for r in results
                if q in r.slick_id.lower() or q in r.port_authority.lower() or q in r.eez_zone.lower()
            ]
        return results

    def get_incident_by_id(self, slick_id: str) -> Optional[UnifiedIncidentPayload]:
        """Returns complete unified forensic payload for a given incident UUID."""
        return self._incidents.get(slick_id)


incident_service = IncidentFusionService()
