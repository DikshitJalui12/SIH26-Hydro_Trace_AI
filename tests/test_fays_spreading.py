"""
Unit tests for Fay's 3-Phase Spreading Theory and Age Estimation.
"""
import unittest

from vision_ai.discrimination.fays_spreading import FaysSpreadingEstimator, SpreadingRegime

class TestFaysSpreading(unittest.TestCase):

    def setUp(self):
        self.estimator = FaysSpreadingEstimator()

    def test_regime_transitions(self):
        # 10 m^3 spill
        t12, t23, r12, r23 = self.estimator.compute_regime_transitions(volume_m3=10.0)
        
        self.assertGreater(t12, 0.0)
        self.assertGreater(t23, t12)
        self.assertGreater(r12, 0.0)
        self.assertGreater(r23, r12)

    def test_age_estimation_realistic_bounds(self):
        # 0.5 km^2 surface area
        res = self.estimator.estimate_age_from_area(area_km2=0.5)
        
        self.assertGreater(res.estimated_age_hours, 0.0)
        self.assertLessEqual(res.estimated_age_hours, 720.0) # Within 30 days (clamped at 30-day max)
        self.assertIn(res.spreading_regime, [
            SpreadingRegime.GRAVITY_INERTIA,
            SpreadingRegime.GRAVITY_VISCOUS,
            SpreadingRegime.SURFACE_TENSION_VISCOUS
        ])
        self.assertGreater(res.effective_radius_m, 100.0)

    def test_regime_progression_with_increasing_area(self):
        # Small area (fresh spill) -> large area (weathered sheen)
        small_slick = self.estimator.estimate_age_from_area(area_km2=0.005, volume_m3=100.0)
        large_slick = self.estimator.estimate_age_from_area(area_km2=5.0, volume_m3=100.0)
        
        self.assertLess(small_slick.estimated_age_hours, large_slick.estimated_age_hours)
        self.assertLess(small_slick.effective_radius_m, large_slick.effective_radius_m)

if __name__ == "__main__":
    unittest.main()
