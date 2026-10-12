"""
AIS Attribution Engine Pipeline Coordinator.
Executes end-to-end spatio-temporal querying, kinematic profiling,
dark vessel transponder analysis, registry risk scoring, and JSON reporting.
"""

from datetime import timedelta
from typing import List, Dict, Any, Optional
from engine.models import (
    SpatioTemporalCorridor,
    AttributionReport,
    RankedSuspect,
    KinematicViolations,
    GeoJSONLineString,
    VesselIdentity
)
from engine.telemetry_store import TelemetryStore
from engine.kinematics import KinematicsProfiler
from engine.dark_vessel import DarkVesselProfiler
from engine.registry import VesselRegistry
from engine.attribution_scorer import AttributionScorer


class AISAttributionPipeline:
    """
    End-to-end attribution pipeline connecting historical AIS storage,
    spatio-temporal uncertainty filtering, and multi-factor forensic scoring.
    """

    def __init__(
        self,
        telemetry_store: Optional[TelemetryStore] = None,
        registry: Optional[VesselRegistry] = None,
        scorer: Optional[AttributionScorer] = None
    ):
        self.store = telemetry_store or TelemetryStore()
        self.registry = registry or VesselRegistry()
        self.scorer = scorer or AttributionScorer()

    def process_corridor(
        self,
        corridor: SpatioTemporalCorridor,
        weights: Optional[Dict[str, float]] = None
    ) -> AttributionReport:
        """
        Execute full forensic attribution investigation for a slick corridor.
        """
        if weights:
            self.scorer = AttributionScorer(weights=weights)

        # 1. Spatio-temporal candidate querying
        candidate_mmsis = self.store.query_candidate_mmsis(corridor)

        # Extended analysis window to capture approach cruising speed and departure
        t_extended_start = corridor.t_origin - timedelta(hours=corridor.delta_t_hours * 2.5)
        t_extended_end = corridor.t_origin + timedelta(hours=corridor.delta_t_hours * 2.5)

        suspect_evaluations = []

        # 2. Forensic profiling for each candidate
        for mmsi in candidate_mmsis:
            trajectory = self.store.get_vessel_trajectory(
                mmsi=mmsi,
                t_start=t_extended_start,
                t_end=t_extended_end
            )

            if not trajectory:
                continue

            # Kinematics analysis (speed drops, loitering, zigzagging)
            kinematics = KinematicsProfiler.analyze_trajectory(trajectory, corridor)

            # Dark vessel & dropout analysis (ITU-R M.1371 ping rates, intentional blackouts)
            dropout = DarkVesselProfiler.analyze_dropout(trajectory, corridor)

            # Merge blackout duration into kinematic violations
            blackout_mins = dropout.get("corridor_blackout_mins", 0.0) or dropout.get("max_blackout_mins", 0.0)
            kinematics.ais_blackout_duration_mins = blackout_mins

            # Registry lookup & MARPOL Annex I profile risk
            sample_pt = trajectory[0]
            identity = self.registry.lookup(mmsi=mmsi, fallback_telemetry=sample_pt)
            profile_risk = self.registry.calculate_vessel_profile_risk(identity)

            # Multi-factor attribution scoring
            score_result = self.scorer.score_candidate(
                identity=identity,
                violations=kinematics,
                dropout_analysis=dropout,
                trajectory=trajectory,
                corridor=corridor,
                vessel_profile_risk=profile_risk
            )

            # Build GeoJSON LineString geometry with timestamped coordinates: [lon, lat, epoch_timestamp]
            coords = [
                [
                    round(pt["lon"], 5),
                    round(pt["lat"], 5),
                    int(pt["timestamp"].timestamp())
                ]
                for pt in trajectory
            ]
            geometry = GeoJSONLineString(coordinates=coords)

            suspect_evaluations.append({
                "attribution_score": score_result["attribution_score"],
                "vessel_identity": identity,
                "kinematic_violations": kinematics,
                "track_geometry": geometry,
                "forensic_summary": score_result["forensic_summary"],
                "score_components": score_result["components"]
            })

        # 3. Sort suspects descending by attribution score
        suspect_evaluations.sort(key=lambda s: s["attribution_score"], reverse=True)

        # 4. Assign ranks and package into ranked suspects
        ranked_suspects: List[RankedSuspect] = []
        for idx, item in enumerate(suspect_evaluations, start=1):
            ranked_suspects.append(RankedSuspect(
                rank=idx,
                attribution_score=item["attribution_score"],
                vessel_identity=item["vessel_identity"],
                kinematic_violations=item["kinematic_violations"],
                track_geometry=item["track_geometry"],
                forensic_summary=item["forensic_summary"],
                score_components=item["score_components"]
            ))

        return AttributionReport(
            slick_id=corridor.slick_id,
            candidates_analyzed=len(candidate_mmsis),
            ranked_suspects=ranked_suspects,
            metadata={
                "corridor_center": [corridor.center_lat, corridor.center_lon],
                "corridor_radius_nm": corridor.radius_nm,
                "corridor_t_origin": corridor.t_origin.isoformat(),
                "delta_t_hours": corridor.delta_t_hours
            }
        )
