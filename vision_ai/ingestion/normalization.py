"""
SAR Normalization and Sliding Window Tiling Engine.
Prepares calibrated multi-polarization (VV/VH) SAR arrays for Deep Learning segmentation.
"""
from typing import Generator, List, Tuple
import numpy as np

class SARNormalizer:
    """
    Normalizes calibrated SAR backscatter in decibels (dB) into [0.0, 1.0] ranges
    suitable for deep learning architectures.
    """
    
    def __init__(
        self,
        vv_range: Tuple[float, float] = (-30.0, 0.0),
        vh_range: Tuple[float, float] = (-35.0, -5.0)
    ):
        self.vv_min, self.vv_max = vv_range
        self.vh_min, self.vh_max = vh_range

    def normalize_band(self, band_db: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Min-max normalization with clipping."""
        clamped = np.clip(band_db, min_val, max_val)
        norm = (clamped - min_val) / (max_val - min_val + 1e-7)
        return norm.astype(np.float32)

    def normalize_dual_pol(self, vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
        """
        Stacks and normalizes VV and VH decibel backscatter into a 2-channel tensor (2, H, W).
        """
        vv_norm = self.normalize_band(vv_db, self.vv_min, self.vv_max)
        vh_norm = self.normalize_band(vh_db, self.vh_min, self.vh_max)
        return np.stack([vv_norm, vh_norm], axis=0)


class SARTiler:
    """
    Slices large SAR rasters into overlapping patches for DL inference
    and recombines them using smooth Hann window blending to eliminate seam artifacts.
    """
    
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        
        # 2D blending weight window (Hann window)
        w1d = np.hanning(tile_size)
        # Avoid zero weights at borders
        w1d = np.clip(w1d, 0.1, 1.0)
        self.blend_window = np.outer(w1d, w1d).astype(np.float32)

    def generate_tiles(
        self, array: np.ndarray
    ) -> Generator[Tuple[np.ndarray, int, int, int, int], None, None]:
        """
        Yields (tile_array, y_start, y_end, x_start, x_end).
        Supports input shapes: (C, H, W) or (H, W).
        """
        is_chw = (array.ndim == 3)
        h = array.shape[1] if is_chw else array.shape[0]
        w = array.shape[2] if is_chw else array.shape[1]

        y_points = list(range(0, max(1, h - self.tile_size + 1), self.stride))
        if y_points[-1] + self.tile_size < h:
            y_points.append(h - self.tile_size)
            
        x_points = list(range(0, max(1, w - self.tile_size + 1), self.stride))
        if x_points[-1] + self.tile_size < w:
            x_points.append(w - self.tile_size)
            
        # Handle case where image is smaller than tile_size
        if h < self.tile_size or w < self.tile_size:
            pad_h = max(0, self.tile_size - h)
            pad_w = max(0, self.tile_size - w)
            if is_chw:
                padded = np.pad(array, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            else:
                padded = np.pad(array, ((0, pad_h), (0, pad_w)), mode='reflect')
            yield padded, 0, h, 0, w
            return

        for y in y_points:
            for x in x_points:
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                
                # Extract tile
                if is_chw:
                    tile = array[:, y:y_end, x:x_end]
                else:
                    tile = array[y:y_end, x:x_end]
                    
                # Pad if partial edge tile
                th = tile.shape[1] if is_chw else tile.shape[0]
                tw = tile.shape[2] if is_chw else tile.shape[1]
                if th < self.tile_size or tw < self.tile_size:
                    ph = self.tile_size - th
                    pw = self.tile_size - tw
                    if is_chw:
                        tile = np.pad(tile, ((0, 0), (0, ph), (0, pw)), mode='reflect')
                    else:
                        tile = np.pad(tile, ((0, ph), (0, pw)), mode='reflect')
                        
                yield tile, y, y_end, x, x_end

    def stitch_tiles(
        self, tile_predictions: List[Tuple[np.ndarray, int, int, int, int]], full_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Stitches predicted probability tiles back into the full canvas with smooth blending.
        """
        h, w = full_shape
        stitched = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)

        for tile, y, y_end, x, x_end in tile_predictions:
            th = y_end - y
            tw = x_end - x
            tile_sub = tile[:th, :tw]
            weight_sub = self.blend_window[:th, :tw]
            
            stitched[y:y_end, x:x_end] += tile_sub * weight_sub
            weight_sum[y:y_end, x:x_end] += weight_sub

        # Avoid division by zero
        stitched = np.divide(stitched, np.maximum(weight_sum, 1e-7))
        return np.clip(stitched, 0.0, 1.0)
