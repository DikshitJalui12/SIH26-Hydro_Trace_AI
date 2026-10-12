"""
Kinematic Anomaly Profiling Engine.
Analyzes velocity profiles, speed drops, loitering loops, and course tortuosity
indicative of illicit MARPOL Annex I underway oily bilge or slop discharges.
"""

import math
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
from engine.models import SpatioTemporalCorridor, KinematicViolations
from engine.telemetry_store import TelemetryStore


class KinematicsProfiler:
    """
    Profiles vessel kinematics for anomalies indicative of MARPOL Annex I violations:
    - Speed drops from cruising speed (14-18 kts) to 4-7 kts (typical underway pumping speed)
    - Loitering loops and zigzagging incompatible with commercial navigation plans
    - Course tortuosity and heading variance
    """

    # Typical operational range for illegal bilge pumping underway: 4.0 to 7.5 knots
    PUMPING_SPEED_MIN = 3.5
    PUMPING_SPEED_MAX = 7.5

    @staticmethod
    def _circular_mean_and_variance(angles_deg: List[float]) -> Tuple[float, float]:
        """Compute circular mean and angular standard deviation in degrees."""
        if not angles_deg:
            return 0.0, 0.0
        rads = np.radians(angles_deg)
        sin_sum = np.sum(np.sin(rads))
        cos_sum = np.sum(np.cos(rads))
        mean_rad = math.atan2(sin_sum, cos_sum)
        mean_deg = (math.degrees(mean_rad) + 360.0) % 360.0

        r = math.sqrt(sin_sum**2 + cos_sum**2) / len(angles_deg)
        r = min(1.0, max(0.0, r))
        # Circular standard deviation in degrees
        circ_std = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(r)))) if r > 1e-6 else 180.0
        return mean_deg, circ_std

    @classmethod
    def analyze_trajectory(
        cls,
        trajectory: List[Dict[str, Any]],
        corridor: SpatioTemporalCorridor
    ) -> KinematicViolations:
        """
        Perform kinematic anomaly profiling across the full trajectory relative to the corridor.
        """
        if not trajectory:
            return KinematicViolations(
                speed_drop_knots=0.0,
                loiter_duration_mins=0.0,
                ais_blackout_duration_mins=0.0,
                transit_speed_knots=0.0,
                cruising_speed_knots=0.0,
                tortuosity_index=1.0,
                heading_variance_deg=0.0
            )

        inside_points = []
        outside_points = []

        for pt in trajectory:
            dist = TelemetryStore.haversine_distance_nm(
                corridor.center_lat, corridor.center_lon, pt["lat"], pt["lon"]
            )
            # Check if point falls within spatial corridor and within temporal window
            time_diff_hours = abs((pt["timestamp"] - corridor.t_origin).total_seconds()) / 3600.0
            if dist <= corridor.radius_nm and time_diff_hours <= (corridor.delta_t_hours * 1.5):
                inside_points.append(pt)
            else:
                outside_points.append(pt)

        # Baseline Cruising Speed calculation
        if outside_points:
            outside_sogs = [p["sog"] for p in outside_points if p["sog"] > 2.0]
            cruising_speed = float(np.median(outside_sogs)) if outside_sogs else 14.0
        else:
            # If all points are inside, take the 90th percentile of speed as the cruising speed
            all_sogs = [p["sog"] for p in trajectory if p["sog"] > 2.0]
            cruising_speed = float(np.percentile(all_sogs, 90)) if all_sogs else 14.0

        # Corridor transit speed
        if inside_points:
            inside_sogs = [p["sog"] for p in inside_points]
            transit_speed = float(np.mean(inside_sogs))
            min_transit_speed = float(np.min(inside_sogs))
        else:
            transit_speed = cruising_speed
            min_transit_speed = cruising_speed

        # Speed drop calculation
        # If vessel was cruising at 15 kts and slowed down to 5.5 kts during corridor transit:
        speed_drop = max(0.0, cruising_speed - min_transit_speed)

        # Loitering duration inside corridor: time intervals where vessel speed was low (< 8.0 kts)
        loiter_minutes = 0.0
        if len(inside_points) >= 2:
            for i in range(len(inside_points) - 1):
                p1 = inside_points[i]
                p2 = inside_points[i + 1]
                dt_mins = (p2["timestamp"] - p1["timestamp"]).total_seconds() / 60.0
                # If dt is reasonable (< 3 hours) and speed is in loitering / pumping range
                if 0 < dt_mins <= 180.0:
                    avg_speed = 0.5 * (p1["sog"] + p2["sog"])
                    if avg_speed <= cls.PUMPING_SPEED_MAX:
                        loiter_minutes += dt_mins

        # Course tortuosity and heading variance inside or near the corridor
        tortuosity_pts = inside_points if len(inside_points) >= 3 else trajectory
        tortuosity_index = 1.0
        heading_variance = 0.0

        if len(tortuosity_pts) >= 2:
            actual_track_length_nm = 0.0
            for i in range(len(tortuosity_pts) - 1):
                actual_track_length_nm += TelemetryStore.haversine_distance_nm(
                    tortuosity_pts[i]["lat"], tortuosity_pts[i]["lon"],
                    tortuosity_pts[i + 1]["lat"], tortuosity_pts[i + 1]["lon"]
                )

            net_displacement_nm = TelemetryStore.haversine_distance_nm(
                tortuosity_pts[0]["lat"], tortuosity_pts[0]["lon"],
                tortuosity_pts[-1]["lat"], tortuosity_pts[-1]["lon"]
            )

            if net_displacement_nm > 0.05:
                tortuosity_index = actual_track_length_nm / net_displacement_nm
            else:
                tortuosity_index = max(1.0, actual_track_length_nm * 5.0)

            cogs = [p["cog"] for p in tortuosity_pts if p["cog"] is not None]
            _, heading_variance = cls._circular_mean_and_variance(cogs)

        return KinematicViolations(
            speed_drop_knots=round(speed_drop, 2),
            loiter_duration_mins=round(loiter_minutes, 1),
            ais_blackout_duration_mins=0.0,  # Will be populated by Dark Vessel Profiler
            transit_speed_knots=round(transit_speed, 2),
            cruising_speed_knots=round(cruising_speed, 2),
            tortuosity_index=round(tortuosity_index, 2),
            heading_variance_deg=round(heading_variance, 2)
        )
