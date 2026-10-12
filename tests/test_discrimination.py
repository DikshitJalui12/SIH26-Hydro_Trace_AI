"""
Unit tests for ERA5 Wind Gating, Polarimetric Features, and Lookalike Classification.
"""
import unittest
import numpy as np

from vision_ai.discrimination.era5_wind_gate import ERA5WindGater
from vision_ai.discrimination.polarization_features import PolarimetricFeatureExtractor, PolarimetricFeatures
from vision_ai.discrimination.lookalike_classifier import LookalikeClassifier, SlickType

class TestDiscrimination(unittest.TestCase):

    def setUp(self):
        self.wind_gater = ERA5WindGater()
        self.feature_extractor = PolarimetricFeatureExtractor()
        self.classifier = LookalikeClassifier()

    def test_wind_gating_low_wind_rejection(self):
        # Calm water condition: U=1.0 m/s, V=0.5 m/s -> W = 1.12 m/s (< 2 m/s)
        result = self.wind_gater.evaluate_wind_gate(
            lat=25.0, lon=55.0, timestamp_utc="2026-09-18T00:00:00Z", u10=1.0, v10=0.5
        )
        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, "REJECTED_CALM_WATER")

    def test_wind_gating_high_wind_rejection(self):
        # Storm condition: U=10.0 m/s, V=10.0 m/s -> W = 14.14 m/s (> 12 m/s)
        result = self.wind_gater.evaluate_wind_gate(
            lat=25.0, lon=55.0, timestamp_utc="2026-09-18T00:00:00Z", u10=10.0, v10=10.0
        )
        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, "REJECTED_HIGH_WIND")

    def test_wind_gating_optimal_window(self):
        # Optimal condition: U=4.0 m/s, V=3.0 m/s -> W = 5.0 m/s (2 <= W <= 12)
        result = self.wind_gater.evaluate_wind_gate(
            lat=25.0, lon=55.0, timestamp_utc="2026-09-18T00:00:00Z", u10=4.0, v10=3.0
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.status, "OPTIMAL_WIND_WINDOW")
        self.assertAlmostEqual(result.wind_speed_ms, 5.0, places=1)

    def test_biogenic_film_rejection(self):
        # Biogenic film mock features: mild damping (< 5 dB), high homogeneity (> 0.85), diffuse edge (< 0.15)
        features = PolarimetricFeatures(
            mean_vv_db=-15.0, mean_vh_db=-24.0,
            sea_vv_db=-11.0, sea_vh_db=-21.0,
            vv_damping_db=4.0, vh_damping_db=3.0,
            damping_ratio=1.33, vv_vh_ratio_db=9.0,
            boundary_gradient=0.08, glcm_contrast=0.05,
            glcm_homogeneity=0.92, glcm_energy=0.88
        )
        wind = self.wind_gater.evaluate_wind_gate(
            lat=25.0, lon=55.0, timestamp_utc="2026-09-18T00:00:00Z", u10=4.0, v10=3.0
        )
        classification = self.classifier.classify_slick(features, wind, area_km2=1.5)
        self.assertEqual(classification.slick_type, SlickType.BIOGENIC_FILM)
        self.assertFalse(classification.is_genuine_oil)

    def test_genuine_mineral_oil_detection(self):
        # Mineral oil mock features: strong damping (8.5 dB), sharp boundary gradient (0.35), moderate homogeneity
        features = PolarimetricFeatures(
            mean_vv_db=-20.5, mean_vh_db=-27.0,
            sea_vv_db=-12.0, sea_vh_db=-21.0,
            vv_damping_db=8.5, vh_damping_db=6.0,
            damping_ratio=1.42, vv_vh_ratio_db=6.5,
            boundary_gradient=0.35, glcm_contrast=0.45,
            glcm_homogeneity=0.65, glcm_energy=0.40
        )
        wind = self.wind_gater.evaluate_wind_gate(
            lat=25.0, lon=55.0, timestamp_utc="2026-09-18T00:00:00Z", u10=4.0, v10=3.0
        )
        classification = self.classifier.classify_slick(features, wind, area_km2=2.5)
        self.assertEqual(classification.slick_type, SlickType.MINERAL_OIL)
        self.assertTrue(classification.is_genuine_oil)
        self.assertGreater(classification.confidence_score, 0.70)

if __name__ == "__main__":
    unittest.main()
