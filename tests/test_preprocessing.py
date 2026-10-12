"""
Unit tests for SAR Ingestion, Radiometric Calibration, Speckle Filtering, and Tiling.
"""
import unittest
import numpy as np

from vision_ai.ingestion.preprocessing import RadiometricCalibrator
from vision_ai.ingestion.speckle_filter import RefinedLeeFilter, GammaMapFilter
from vision_ai.ingestion.normalization import SARNormalizer, SARTiler

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        self.calibrator = RadiometricCalibrator()
        self.lee_filter = RefinedLeeFilter(window_size=7, enl=4.4)
        self.gamma_filter = GammaMapFilter(window_size=7, enl=4.4)
        self.normalizer = SARNormalizer()
        self.tiler = SARTiler(tile_size=128, overlap=32)

    def test_radiometric_calibration(self):
        # Synthetic DN array with nominal values around 150
        dn = np.full((50, 50), 150.0, dtype=np.float32)
        sigma0_lin, sigma0_db = self.calibrator.calibrate_band(dn)
        
        self.assertEqual(sigma0_lin.shape, (50, 50))
        self.assertEqual(sigma0_db.shape, (50, 50))
        # 150 / 300 = 0.5, squared = 0.25 -> 10*log10(0.25) ~= -6.02 dB
        self.assertAlmostEqual(float(sigma0_lin[0, 0]), 0.25, places=2)
        self.assertAlmostEqual(float(sigma0_db[0, 0]), -6.02, places=1)

    def test_refined_lee_filter_variance_reduction(self):
        # Create noisy image with multiplicative speckle
        np.random.seed(42)
        clean = np.full((100, 100), 0.1, dtype=np.float32)
        speckle = np.random.gamma(shape=4.0, scale=0.25, size=(100, 100)).astype(np.float32)
        noisy = clean * speckle
        
        filtered = self.lee_filter.filter(noisy)
        
        self.assertEqual(filtered.shape, noisy.shape)
        # Filtered variance should be significantly less than noisy variance
        var_noisy = np.var(noisy)
        var_filtered = np.var(filtered)
        self.assertLess(var_filtered, var_noisy)
        self.assertTrue(np.all(filtered > 0.0))

    def test_gamma_map_filter(self):
        np.random.seed(42)
        noisy = np.random.uniform(0.05, 0.25, size=(80, 80)).astype(np.float32)
        filtered = self.gamma_filter.filter(noisy)
        self.assertEqual(filtered.shape, noisy.shape)
        self.assertLess(np.var(filtered), np.var(noisy))

    def test_normalization_bounds(self):
        vv_db = np.array([[-35.0, -30.0, -15.0, 0.0, 5.0]], dtype=np.float32)
        vh_db = np.array([[-40.0, -35.0, -20.0, -5.0, 0.0]], dtype=np.float32)
        
        norm_dual = self.normalizer.normalize_dual_pol(vv_db, vh_db)
        self.assertEqual(norm_dual.shape, (2, 1, 5))
        self.assertGreaterEqual(float(np.min(norm_dual)), 0.0)
        self.assertLessEqual(float(np.max(norm_dual)), 1.0)

    def test_tiling_and_stitching(self):
        # Test full cycle of slicing and recombining
        canvas = np.ones((2, 200, 200), dtype=np.float32) * 0.8
        tiles = list(self.tiler.generate_tiles(canvas))
        self.assertGreater(len(tiles), 1)

        # Mock predictions for each tile
        tile_preds = []
        for tile, y, y_end, x, x_end in tiles:
            # Predict 1-channel probability
            mock_pred = np.full((128, 128), 0.75, dtype=np.float32)
            tile_preds.append((mock_pred, y, y_end, x, x_end))

        stitched = self.tiler.stitch_tiles(tile_preds, (200, 200))
        self.assertEqual(stitched.shape, (200, 200))
        # Values should be blended smoothly around 0.75
        self.assertAlmostEqual(float(np.mean(stitched)), 0.75, places=2)

if __name__ == "__main__":
    unittest.main()
