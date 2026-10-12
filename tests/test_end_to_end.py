"""
End-to-End Integration Tests for Vision AI Tier.
Validates the complete pipeline on synthetic Sentinel-1 GRD imagery,
checking adherence to output JSON schema, wind gating, and metrics.
"""
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from data.sample_generator import generate_synthetic_sar_scene
from vision_ai.config import PipelineConfig
from vision_ai.pipeline import VisionAITier

class TestEndToEndPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_tif = Path(cls.temp_dir.name) / "s1_test_scene.tif"
        
        # Generate synthetic Sentinel-1 GRD scene
        generate_synthetic_sar_scene(
            str(cls.test_tif),
            height=256,
            width=256,
            center_lat=25.2,
            center_lon=55.3,
            pixel_size_deg=0.0001
        )

        config = PipelineConfig()
        config.preprocessing.tile_size = 256
        config.preprocessing.tile_overlap = 32
        config.segmentation.base_filters = 16
        config.segmentation.probability_threshold = 0.40
        config.segmentation.min_slick_area_km2 = 0.01

        cls.pipeline = VisionAITier(config=config)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_complete_pipeline_payload_schema(self):
        # Run with optimal wind conditions (U=4.0, V=2.0 -> ~4.47 m/s)
        results = self.pipeline.process_scene(
            product_path=self.test_tif,
            simulated_u10=4.0,
            simulated_v10=2.0,
            discard_on_wind_fail=True
        )

        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1, "Expected at least 1 slick detected")

        slick = results[0]
        
        # 1. slick_id (UUID)
        self.assertIn("slick_id", slick)
        # Verify valid UUID string
        val_uuid = uuid.UUID(slick["slick_id"])
        self.assertEqual(str(val_uuid), slick["slick_id"])

        # 2. timestamp_utc (ISO 8601)
        self.assertIn("timestamp_utc", slick)
        self.assertIsInstance(slick["timestamp_utc"], str)

        # 3. satellite_metadata (orbit, sensor, pass_direction)
        self.assertIn("satellite_metadata", slick)
        sat_meta = slick["satellite_metadata"]
        self.assertIn("orbit", sat_meta)
        self.assertIn("sensor", sat_meta)
        self.assertIn("pass_direction", sat_meta)

        # 4. centroid: { latitude: float, longitude: float }
        self.assertIn("centroid", slick)
        centroid = slick["centroid"]
        self.assertIn("latitude", centroid)
        self.assertIn("longitude", centroid)
        self.assertAlmostEqual(centroid["latitude"], 25.2, delta=0.1)
        self.assertAlmostEqual(centroid["longitude"], 55.3, delta=0.1)

        # 5. geometry: GeoJSON Polygon
        self.assertIn("geometry", slick)
        geom = slick["geometry"]
        self.assertEqual(geom["type"], "Polygon")
        self.assertIn("coordinates", geom)
        self.assertGreaterEqual(len(geom["coordinates"][0]), 4) # Closed polygon has >= 4 points

        # 6. metrics: { area_km2, perimeter_km, mean_backscatter_db, estimated_age_hours }
        self.assertIn("metrics", slick)
        metrics = slick["metrics"]
        self.assertIn("area_km2", metrics)
        self.assertIn("perimeter_km", metrics)
        self.assertIn("mean_backscatter_db", metrics)
        self.assertIn("estimated_age_hours", metrics)
        self.assertGreater(metrics["area_km2"], 0.0)
        self.assertGreater(metrics["perimeter_km"], 0.0)
        self.assertLess(metrics["mean_backscatter_db"], -10.0) # Dark radar return
        self.assertGreater(metrics["estimated_age_hours"], 0.0)

        # 7. confidence_score: float (0.0 to 1.0)
        self.assertIn("confidence_score", slick)
        score = slick["confidence_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_wind_gating_discards_under_calm_water(self):
        # Calm water condition (wind speed < 2 m/s)
        results = self.pipeline.process_scene(
            product_path=self.test_tif,
            simulated_u10=1.0,
            simulated_v10=0.5,
            discard_on_wind_fail=True
        )
        self.assertEqual(len(results), 0, "Calm water should discard all detections")

if __name__ == "__main__":
    unittest.main()
