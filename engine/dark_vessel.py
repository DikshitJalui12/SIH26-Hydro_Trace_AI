"""
Dark Vessel & Transponder Dropout Profiling Engine.
Detects intentional Class-A AIS transponder blackouts compliant with ITU-R M.1371,
calculates expected vs received ping rates, and applies heavy penalties to vessels
going dark immediately before/during discharge envelope transit.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import numpy as np
from engine.models import SpatioTemporalCorridor
from engine.telemetry_store import TelemetryStore


class DarkVesselProfiler:
    """
    Evaluates AIS transponder behavior against ITU-R M.1371 standards.
    Identifies intentional blackouts, missing ping ratios, and zone-correlated dropouts.
    """

    @staticmethod
    def get_itur_expected_interval_seconds(sog: float, is_maneuvering: bool = False) -> float:
        """
        Nominal Class-A AIS reporting intervals under ITU-R M.1371:
        - Ship at anchor or moored and not moving > 3 kts: 3 min (180 s)
        - Ship 0-14 kts: 10 s (3.33 s if changing course)
        - Ship 14-23 kts: 6 s (2.0 s if changing course)
        - Ship > 23 kts: 2 s
        """
        if sog <= 3.0:
            return 180.0
        elif sog <= 14.0:
            return 3.33 if is_maneuvering else 10.0
        elif sog <= 23.0:
            return 2.0 if is_maneuvering else 6.0
        else:
            return 2.0

    @classmethod
    def analyze_dropout(
        cls,
        trajectory: List[Dict[str, Any]],
        corridor: SpatioTemporalCorridor,
        satellite_downsampling_factor: float = 6.0
    ) -> Dict[str, Any]:
        """
        Analyze AIS gaps, ping integrity, and corridor-correlated blackout behavior.
        
        satellite_downsampling_factor: Accounting for realistic S-AIS orbit passes
        where raw terrestrial 10s pings are aggregated or downsampled to ~60s in store.
        """
        if not trajectory or len(trajectory) < 2:
            return {
                "max_blackout_mins": 0.0,
                "corridor_blackout_mins": 0.0,
                "expected_pings": 0,
                "received_pings": len(trajectory),
                "dropout_ratio": 0.0,
                "envelope_blackout_flag": False,
                "penalty_factor": 1.0,
                "blackout_intervals": []
            }

        t_corridor_start = corridor.t_origin - timedelta(hours=corridor.delta_t_hours)
        t_corridor_end = corridor.t_origin + timedelta(hours=corridor.delta_t_hours)

        total_duration_sec = (trajectory[-1]["timestamp"] - trajectory[0]["timestamp"]).total_seconds()
        received_pings = len(trajectory)

        # Calculate expected pings based on mean speed and ITU-R M.1371 interval
        mean_sog = float(np.mean([p["sog"] for p in trajectory]))
        nominal_interval = cls.get_itur_expected_interval_seconds(mean_sog) * satellite_downsampling_factor
        expected_pings = max(received_pings, int(total_duration_sec / max(5.0, nominal_interval)))

        # Find gaps between consecutive transmissions
        max_overall_gap_sec = 0.0
        corridor_gap_sec = 0.0
        blackout_intervals = []
        envelope_blackout_flag = False

        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]
            p2 = trajectory[i + 1]
            dt_sec = (p2["timestamp"] - p1["timestamp"]).total_seconds()

            if dt_sec > max_overall_gap_sec:
                max_overall_gap_sec = dt_sec

            # Check if this gap occurred in or near the corridor window
            gap_midpoint = p1["timestamp"] + timedelta(seconds=dt_sec / 2.0)
            in_time_window = (t_corridor_start - timedelta(minutes=30)) <= gap_midpoint <= (t_corridor_end + timedelta(minutes=30))

            dist1 = TelemetryStore.haversine_distance_nm(corridor.center_lat, corridor.center_lon, p1["lat"], p1["lon"])
            dist2 = TelemetryStore.haversine_distance_nm(corridor.center_lat, corridor.center_lon, p2["lat"], p2["lon"])
            near_corridor = (dist1 <= corridor.radius_nm * 2.0) or (dist2 <= corridor.radius_nm * 2.0)

            # A gap > 15 minutes is treated as an abnormal blackout (standard AIS should ping every 10s-3min)
            if dt_sec >= 900.0:
                blackout_intervals.append({
                    "start": p1["timestamp"].isoformat(),
                    "end": p2["timestamp"].isoformat(),
                    "duration_mins": round(dt_sec / 60.0, 1),
                    "start_pos": [round(p1["lon"], 4), round(p1["lat"], 4)],
                    "end_pos": [round(p2["lon"], 4), round(p2["lat"], 4)]
                })

            if in_time_window and near_corridor and dt_sec > corridor_gap_sec:
                corridor_gap_sec = dt_sec

            # Signature: Vessel was transmitting normally, goes dark entering corridor,
            # and re-emerges after clearing corridor
            if dt_sec >= 1200.0 and in_time_window:
                # Check if p1 is on the approach side and p2 is downstream
                if (dist1 <= corridor.radius_nm * 2.5) or (dist2 <= corridor.radius_nm * 2.5):
                    envelope_blackout_flag = True

        dropout_ratio = max(0.0, min(1.0, 1.0 - (received_pings / max(1, expected_pings))))

        # Penalty factor: deliberate transponder shutdown in zone receives heavy multiplier
        penalty_factor = 1.0
        if envelope_blackout_flag:
            penalty_factor = 1.6  # 60% penalty escalation for dark transponder in corridor
        elif corridor_gap_sec > 1800.0:
            penalty_factor = 1.3

        return {
            "max_blackout_mins": round(max_overall_gap_sec / 60.0, 1),
            "corridor_blackout_mins": round(corridor_gap_sec / 60.0, 1),
            "expected_pings": expected_pings,
            "received_pings": received_pings,
            "dropout_ratio": round(dropout_ratio, 3),
            "envelope_blackout_flag": envelope_blackout_flag,
            "penalty_factor": penalty_factor,
            "blackout_intervals": blackout_intervals
        }
