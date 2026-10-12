"""
Multi-Factor Attribution Scoring Engine & Forensic Summarizer.
Combines spatio-temporal proximity, kinematic anomalies (speed drops / loitering),
deliberate Class-A AIS transponder blackouts, and MARPOL Annex I registry risk profiles.
"""

import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from engine.models import (
    SpatioTemporalCorridor,
    VesselIdentity,
    KinematicViolations,
    RankedSuspect,
    GeoJSONLineString
)
from engine.telemetry_store import TelemetryStore


class AttributionScorer:
    """
    Computes multi-factor attribution scores (0 to 100%) using the equation:
    Score = w1 * SpatialProximity + w2 * SpeedAnomaly + w3 * LoiteringTime +
            w4 * TransponderGap + w5 * VesselProfileRisk
    """

    DEFAULT_WEIGHTS = {
        "spatial_proximity": 0.25,
        "speed_anomaly": 0.25,
        "loitering_time": 0.15,
        "transponder_gap": 0.20,
        "vessel_profile_risk": 0.15
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        # Ensure weights sum to 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def calculate_spatial_proximity_score(
        self,
        trajectory: List[Dict[str, Any]],
        corridor: SpatioTemporalCorridor
    ) -> float:
        """
        Calculate normalized spatial-temporal proximity score [0.0, 1.0].
        Considers minimum distance to corridor origin and temporal alignment.
        """
        if not trajectory:
            return 0.0

        min_dist_nm = float("inf")
        closest_pt_time = None

        for pt in trajectory:
            dist = TelemetryStore.haversine_distance_nm(
                corridor.center_lat, corridor.center_lon, pt["lat"], pt["lon"]
            )
            if dist < min_dist_nm:
                min_dist_nm = dist
                closest_pt_time = pt["timestamp"]

        if min_dist_nm > corridor.radius_nm * 1.5:
            return 0.0

        # Distance decay factor: 1.0 if right at center, 0.0 at corridor edge
        dist_factor = max(0.0, 1.0 - (min_dist_nm / max(0.1, corridor.radius_nm)))

        # Temporal decay factor: 1.0 at exact t_origin
        time_factor = 1.0
        if closest_pt_time is not None:
            def _parse_ts(ts):
                if isinstance(ts, datetime):
                    return ts
                if isinstance(ts, str):
                    clean = ts.replace("Z", "+00:00")
                    return datetime.fromisoformat(clean)
                return ts

            try:
                t_pt = _parse_ts(closest_pt_time)
                t_corr = _parse_ts(corridor.t_origin)
                if t_pt.tzinfo is None and t_corr.tzinfo is not None:
                    t_pt = t_pt.replace(tzinfo=timezone.utc)
                elif t_pt.tzinfo is not None and t_corr.tzinfo is None:
                    t_corr = t_corr.replace(tzinfo=timezone.utc)
                dt_hours = abs((t_pt - t_corr).total_seconds()) / 3600.0
                time_factor = max(0.0, 1.0 - (dt_hours / max(1.0, corridor.delta_t_hours * 1.5)))
            except Exception:
                time_factor = 0.5

        return round(0.7 * dist_factor + 0.3 * time_factor, 3)

    def calculate_speed_anomaly_score(
        self,
        violations: KinematicViolations
    ) -> float:
        """
        Calculate speed anomaly score [0.0, 1.0].
        Emphasizes drops from cruising speed (13-18 kts) into the 4-7 knot
        slow underway bilge/slop pumping envelope under MARPOL Annex I.
        """
        speed_drop = violations.speed_drop_knots
        transit_speed = violations.transit_speed_knots or (
            violations.cruising_speed_knots - violations.speed_drop_knots
            if violations.cruising_speed_knots else 12.0
        )

        # Baseline score proportional to speed drop (max out at 10 knots drop)
        drop_score = min(1.0, speed_drop / 10.0)

        # Severe penalty if transit speed falls directly into the 4.0 - 7.5 knot pumping window
        pumping_window_bonus = 0.0
        if 3.5 <= transit_speed <= 7.5 and speed_drop >= 4.0:
            pumping_window_bonus = 0.35
        elif transit_speed < 3.5 and speed_drop >= 5.0:
            pumping_window_bonus = 0.20  # Nearly stopped or drifting

        total_score = min(1.0, drop_score * 0.65 + pumping_window_bonus)
        return round(total_score, 3)

    def calculate_loitering_score(
        self,
        violations: KinematicViolations
    ) -> float:
        """
        Calculate loitering & course maneuvering score [0.0, 1.0].
        Accounts for time spent at reduced speed and track tortuosity.
        """
        # 45 minutes of loitering in commercial fairway is highly abnormal
        loiter_mins = violations.loiter_duration_mins
        time_score = min(1.0, loiter_mins / 45.0)

        # Tortuosity factor: straight transit = 1.0, zigzag / loops >= 1.5
        tortuosity = violations.tortuosity_index or 1.0
        tortuosity_score = min(1.0, max(0.0, (tortuosity - 1.0) / 0.8))

        # Heading variance factor
        heading_var = violations.heading_variance_deg or 0.0
        heading_score = min(1.0, heading_var / 60.0)

        composite = 0.50 * time_score + 0.30 * tortuosity_score + 0.20 * heading_score
        return round(min(1.0, composite), 3)

    def calculate_transponder_gap_score(
        self,
        dropout_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate AIS transponder blackout score [0.0, 1.0].
        Heavily penalizes vessels going dark prior to entering the discharge envelope.
        """
        corridor_blackout = dropout_analysis.get("corridor_blackout_mins", 0.0)
        max_blackout = dropout_analysis.get("max_blackout_mins", 0.0)
        envelope_blackout_flag = dropout_analysis.get("envelope_blackout_flag", False)
        penalty_factor = dropout_analysis.get("penalty_factor", 1.0)

        # 40+ minutes of transponder blackout is heavily suspicious
        raw_gap_score = min(1.0, max(corridor_blackout, max_blackout) / 45.0)

        if envelope_blackout_flag:
            # Vessel went silent right before/during envelope and re-appeared after
            return round(min(1.0, max(0.85, raw_gap_score * penalty_factor)), 3)
        else:
            return round(min(1.0, raw_gap_score * penalty_factor), 3)

    def score_candidate(
        self,
        identity: VesselIdentity,
        violations: KinematicViolations,
        dropout_analysis: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        corridor: SpatioTemporalCorridor,
        vessel_profile_risk: float
    ) -> Dict[str, Any]:
        """
        Compute comprehensive attribution score and component breakdown.
        """
        s_spatial = self.calculate_spatial_proximity_score(trajectory, corridor)
        s_speed = self.calculate_speed_anomaly_score(violations)
        s_loiter = self.calculate_loitering_score(violations)
        s_transponder = self.calculate_transponder_gap_score(dropout_analysis)
        s_profile = vessel_profile_risk

        # Multi-Factor Attribution Formula
        raw_score = (
            self.weights["spatial_proximity"] * s_spatial +
            self.weights["speed_anomaly"] * s_speed +
            self.weights["loitering_time"] * s_loiter +
            self.weights["transponder_gap"] * s_transponder +
            self.weights["vessel_profile_risk"] * s_profile
        )

        final_score_pct = round(min(100.0, max(0.0, raw_score * 100.0)), 1)

        components = {
            "spatial_proximity": round(s_spatial * 100.0, 1),
            "speed_anomaly": round(s_speed * 100.0, 1),
            "loitering_time": round(s_loiter * 100.0, 1),
            "transponder_gap": round(s_transponder * 100.0, 1),
            "vessel_profile_risk": round(s_profile * 100.0, 1)
        }

        forensic_summary = self.generate_forensic_summary(
            identity=identity,
            violations=violations,
            dropout_analysis=dropout_analysis,
            corridor=corridor,
            score_pct=final_score_pct,
            components=components
        )

        return {
            "attribution_score": final_score_pct,
            "components": components,
            "forensic_summary": forensic_summary
        }

    @staticmethod
    def generate_forensic_summary(
        identity: VesselIdentity,
        violations: KinematicViolations,
        dropout_analysis: Dict[str, Any],
        corridor: SpatioTemporalCorridor,
        score_pct: float,
        components: Dict[str, float]
    ) -> str:
        """
        Generate detailed, expert natural-language forensic analysis
        referencing MARPOL 73/78 Annex I regulations, AIS kinematics, and transponder behavior.
        """
        marpol_ref = "MARPOL 73/78 Annex I Regulation 34 (Cargo Area / Tank Washings)" \
            if "Tanker" in identity.vessel_type else "MARPOL 73/78 Annex I Regulation 15 (Machinery Space Bilge Discharges)"

        speed_drop = violations.speed_drop_knots
        transit_spd = violations.transit_speed_knots or 0.0
        cruise_spd = violations.cruising_speed_knots or 0.0
        loiter_mins = violations.loiter_duration_mins
        blackout_mins = dropout_analysis.get("corridor_blackout_mins", 0.0) or dropout_analysis.get("max_blackout_mins", 0.0)
        envelope_blackout = dropout_analysis.get("envelope_blackout_flag", False)

        lines = []
        if score_pct >= 75.0:
            lines.append(
                f"HIGH PROBABILITY SUSPECT (Score: {score_pct}%). "
                f"Vessel {identity.name} (IMO: {identity.imo}, MMSI: {identity.mmsi}, Flag: {identity.flag}, Type: {identity.vessel_type}) "
                f"exhibits an acute signature of illicit oily discharge under {marpol_ref}."
            )
        elif score_pct >= 50.0:
            lines.append(
                f"MEDIUM PROBABILITY SUSPECT (Score: {score_pct}%). "
                f"Vessel {identity.name} (IMO: {identity.imo}, Flag: {identity.flag}, Type: {identity.vessel_type}) "
                f"intersected the spill corridor with notable operational anomalies."
            )
        else:
            lines.append(
                f"LOW PROBABILITY / CLEARED (Score: {score_pct}%). "
                f"Vessel {identity.name} (IMO: {identity.imo}, Flag: {identity.flag}, Type: {identity.vessel_type}) "
                f"maintained routine commercial transit kinematics with nominal transponder integrity."
            )

        # Kinematic breakdown
        if speed_drop >= 5.0 and 3.5 <= transit_spd <= 7.5:
            lines.append(
                f"Kinematic Violation: Exhibited an abrupt, unprompted deceleration of {speed_drop:.1f} kts "
                f"(from cruising speed of {cruise_spd:.1f} kts down to {transit_spd:.1f} kts) inside origin envelope {corridor.slick_id}. "
                f"This speed range ({transit_spd:.1f} kts) corresponds directly to the optimal slow underway bilge/slop pumping "
                f"window utilized to bypass 15 ppm Oily Water Separators (OWS) or discharge oily waste without creating conspicuous wake sheen."
            )
        elif speed_drop >= 3.0:
            lines.append(
                f"Kinematic Observation: Recorded a speed reduction of {speed_drop:.1f} kts (from {cruise_spd:.1f} to {transit_spd:.1f} kts)."
            )

        # Loiter / course maneuvers
        if loiter_mins >= 15.0 or (violations.tortuosity_index and violations.tortuosity_index > 1.25):
            lines.append(
                f"Maneuvering Anomaly: Vessel loitered for {loiter_mins:.1f} minutes in the envelope with a tortuosity index of "
                f"{violations.tortuosity_index or 1.0:.2f} and heading variance of {violations.heading_variance_deg or 0.0:.1f}°, "
                f"incompatible with standard commercial passage planning."
            )

        # Transponder blackout
        if envelope_blackout:
            lines.append(
                f"Transponder Tampering: Detected a deliberate Class-A AIS blackout spanning {blackout_mins:.1f} minutes. "
                f"The vessel transmitted normally upon approach, disabled its transponder immediately prior to entering the discharge envelope, "
                f"and re-engaged AIS after clearing the zone, in clear violation of SOLAS Chapter V Regulation 19 and ITU-R M.1371 standards."
            )
        elif blackout_mins >= 15.0:
            lines.append(
                f"AIS Transmission Gap: Observed a non-transmitting window of {blackout_mins:.1f} minutes near the envelope."
            )

        # Risk profile & ownership
        lines.append(
            f"Vessel Profile Risk: Registered owner '{identity.owner}', operated under {identity.flag} registry. "
            f"Annex I discharge capability rated at {components['vessel_profile_risk']:.1f}%."
        )

        return " ".join(lines)
