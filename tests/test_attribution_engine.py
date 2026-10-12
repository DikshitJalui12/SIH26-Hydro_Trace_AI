"""
Unit and Integration Test Suite for AIS Attribution Engine.
Validates ITU-R M.1371 compliance, MARPOL 73/78 Annex I kinematics,
spatial indexing, multi-factor scoring, and JSON schema conformity.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.models import SpatioTemporalCorridor, AISMessage
from engine.telemetry_store import TelemetryStore
from engine.kinematics import KinematicsProfiler
from engine.dark_vessel import DarkVesselProfiler
from engine.registry import VesselRegistry
from engine.attribution_scorer import AttributionScorer
from engine.pipeline import AISAttributionPipeline
from data.synthetic_ais_generator import generate_scenario_telemetry


class TestAISAttributionEngine(unittest.TestCase):

    def setUp(self):
        self.store = TelemetryStore()
        self.corridor = SpatioTemporalCorridor(
            slick_id="TEST-SLICK-001",
            center_lat=1.2100,
            center_lon=103.2700,
            radius_nm=6.0,
            t_origin=datetime.fromisoformat("2026-09-18T02:00:00"),
            delta_t_hours=1.5
        )

    def test_haversine_distance_calculation(self):
        """Verify Haversine distance matches known maritime distances."""
        # Distance between (1.0, 103.0) and (1.0, 104.0) at equator is ~60 NM
        dist = TelemetryStore.haversine_distance_nm(1.0, 103.0, 1.0, 104.0)
        self.assertAlmostEqual(dist, 59.98, delta=0.5)

        # Distance to same point is 0
        dist_zero = TelemetryStore.haversine_distance_nm(1.21, 103.27, 1.21, 103.27)
        self.assertEqual(dist_zero, 0.0)

    def test_duckdb_spatial_query_filtering(self):
        """Test candidate MMSI spatial and temporal query filtering."""
        records = generate_scenario_telemetry()
        self.store.ingest_records(records)

        candidates = self.store.query_candidate_mmsis(self.corridor)
        # Primary suspect 563001230, tanker 354112000, container 477210980, bulk 636018920 should be captured
        self.assertIn("563001230", candidates)
        self.assertIn("477210980", candidates)
        self.assertIn("636018920", candidates)

    def test_kinematics_speed_drop_profiling(self):
        """Verify speed drop detection flags transitions into 4-7 knot pumping window."""
        t_base = datetime.fromisoformat("2026-09-18T00:00:00")
        trajectory = []
        # Cruising at 16 kts outside corridor
        for i in range(10):
            trajectory.append({
                "timestamp": t_base + timedelta(minutes=i * 5),
                "lat": 1.0 + i * 0.01,
                "lon": 103.0 + i * 0.01,
                "sog": 16.0,
                "cog": 45.0
            })
        # Slowing down to 5.0 kts inside corridor
        for i in range(10, 15):
            trajectory.append({
                "timestamp": t_base + timedelta(minutes=i * 5),
                "lat": 1.21 + (i - 10) * 0.001,
                "lon": 103.27 + (i - 10) * 0.001,
                "sog": 5.0,
                "cog": 45.0
            })

        violations = KinematicsProfiler.analyze_trajectory(trajectory, self.corridor)
        self.assertGreaterEqual(violations.speed_drop_knots, 10.0)
        self.assertAlmostEqual(violations.cruising_speed_knots, 16.0, delta=1.0)
        self.assertAlmostEqual(violations.transit_speed_knots, 5.0, delta=1.0)
        self.assertGreater(violations.loiter_duration_mins, 15.0)

    def test_dark_vessel_itur_ping_intervals(self):
        """Verify ITU-R M.1371 dynamic ping interval logic."""
        # At anchor <= 3 kts
        self.assertEqual(DarkVesselProfiler.get_itur_expected_interval_seconds(1.5), 180.0)
        # Slow speed <= 14 kts
        self.assertEqual(DarkVesselProfiler.get_itur_expected_interval_seconds(10.0), 10.0)
        self.assertEqual(DarkVesselProfiler.get_itur_expected_interval_seconds(10.0, is_maneuvering=True), 3.33)
        # Medium speed 14-23 kts
        self.assertEqual(DarkVesselProfiler.get_itur_expected_interval_seconds(18.0), 6.0)
        self.assertEqual(DarkVesselProfiler.get_itur_expected_interval_seconds(18.0, is_maneuvering=True), 2.0)
        # Fast speed > 23 kts
        self.assertEqual(DarkVesselProfiler.get_itur_expected_interval_seconds(25.0), 2.0)

    def test_dark_vessel_corridor_blackout_detection(self):
        """Verify transponder blackout is detected when vessel goes dark in corridor."""
        t_base = datetime.fromisoformat("2026-09-18T01:00:00")
        trajectory = [
            {"timestamp": t_base, "lat": 1.15, "lon": 103.15, "sog": 15.0, "cog": 60.0},
            {"timestamp": t_base + timedelta(minutes=15), "lat": 1.19, "lon": 103.22, "sog": 15.0, "cog": 60.0},
            # 50-minute transponder blackout spanning entry and exit of envelope
            {"timestamp": t_base + timedelta(minutes=65), "lat": 1.25, "lon": 103.35, "sog": 14.5, "cog": 60.0},
            {"timestamp": t_base + timedelta(minutes=80), "lat": 1.28, "lon": 103.40, "sog": 14.5, "cog": 60.0}
        ]
        dropout = DarkVesselProfiler.analyze_dropout(trajectory, self.corridor)
        self.assertGreaterEqual(dropout["max_blackout_mins"], 49.0)
        self.assertTrue(dropout["envelope_blackout_flag"])
        self.assertGreater(dropout["penalty_factor"], 1.0)

    def test_imo_checksum_validation(self):
        """Verify official IMO 7-digit check digit algorithm."""
        # Valid IMO numbers
        self.assertTrue(VesselRegistry.validate_imo("9421831"))
        self.assertTrue(VesselRegistry.validate_imo("IMO 9684122"))
        self.assertTrue(VesselRegistry.validate_imo("9312042"))

        # Invalid IMO numbers
        self.assertFalse(VesselRegistry.validate_imo("9421832"))  # Bad checksum
        self.assertFalse(VesselRegistry.validate_imo("12345"))    # Too short
        self.assertFalse(VesselRegistry.validate_imo("ABCDEFG"))  # Non-digit

    def test_attribution_scoring_normalization(self):
        """Verify attribution scores are strictly normalized within [0.0, 100.0] range."""
        scorer = AttributionScorer()
        registry = VesselRegistry()

        # Build extreme case (100% violation on all counts)
        identity = registry.lookup("563001230")
        violations = KinematicsProfiler.analyze_trajectory([], self.corridor)
        violations.speed_drop_knots = 12.0
        violations.transit_speed_knots = 5.0
        violations.loiter_duration_mins = 60.0
        violations.tortuosity_index = 2.0
        violations.heading_variance_deg = 80.0

        dropout = {
            "corridor_blackout_mins": 50.0,
            "max_blackout_mins": 50.0,
            "envelope_blackout_flag": True,
            "penalty_factor": 1.6
        }

        # Trajectory right at corridor center
        traj = [{"timestamp": self.corridor.t_origin, "lat": self.corridor.center_lat, "lon": self.corridor.center_lon}]

        res = scorer.score_candidate(
            identity=identity,
            violations=violations,
            dropout_analysis=dropout,
            trajectory=traj,
            corridor=self.corridor,
            vessel_profile_risk=1.0
        )
        self.assertGreaterEqual(res["attribution_score"], 80.0)
        self.assertLessEqual(res["attribution_score"], 100.0)

    def test_end_to_end_pipeline_output_format(self):
        """Verify pipeline output strictly adheres to requested JSON payload format."""
        records = generate_scenario_telemetry()
        self.store.ingest_records(records)
        pipeline = AISAttributionPipeline(telemetry_store=self.store)

        report = pipeline.process_corridor(self.corridor)
        data = report.model_dump()

        # Verify top-level keys
        self.assertEqual(data["slick_id"], "TEST-SLICK-001")
        self.assertIsInstance(data["candidates_analyzed"], int)
        self.assertGreater(data["candidates_analyzed"], 0)
        self.assertIsInstance(data["ranked_suspects"], list)

        # Verify ranked suspect structure
        top_suspect = data["ranked_suspects"][0]
        self.assertEqual(top_suspect["rank"], 1)
        self.assertIsInstance(top_suspect["attribution_score"], float)
        self.assertGreaterEqual(top_suspect["attribution_score"], 70.0)

        # Verify vessel_identity schema
        identity = top_suspect["vessel_identity"]
        for key in ["imo", "mmsi", "name", "flag", "vessel_type", "owner"]:
            self.assertIn(key, identity)

        # Verify kinematic_violations schema
        violations = top_suspect["kinematic_violations"]
        for key in ["speed_drop_knots", "loiter_duration_mins", "ais_blackout_duration_mins"]:
            self.assertIn(key, violations)
            self.assertIsInstance(violations[key], (int, float))

        # Verify track_geometry schema (GeoJSON LineString)
        geom = top_suspect["track_geometry"]
        self.assertEqual(geom["type"], "LineString")
        self.assertIsInstance(geom["coordinates"], list)
        self.assertGreater(len(geom["coordinates"]), 0)
        self.assertEqual(len(geom["coordinates"][0]), 3)

        # Verify forensic_summary
        self.assertIsInstance(top_suspect["forensic_summary"], str)
        self.assertIn("MARPOL 73/78 Annex I", top_suspect["forensic_summary"])
        self.assertIn("OCEAN APEX", top_suspect["vessel_identity"]["name"])


if __name__ == "__main__":
    unittest.main()
