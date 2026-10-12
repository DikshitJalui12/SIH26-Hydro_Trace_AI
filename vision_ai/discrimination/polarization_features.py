"""
Polarimetric SAR Feature Extraction and Textural Analysis.
Calculates VV/VH cross-polarization damping ratios, backscatter contrast,
and Gray-Level Co-occurrence Matrix (GLCM) textural parameters.
"""
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.ndimage import binary_dilation, sobel

@dataclass
class PolarimetricFeatures:
    """Extracted polarimetric and textural indicators for candidate oil slick."""
    mean_vv_db: float
    mean_vh_db: float
    sea_vv_db: float
    sea_vh_db: float
    vv_damping_db: float      # D_vv = sigma0_sea_vv_db - sigma0_slick_vv_db
    vh_damping_db: float      # D_vh = sigma0_sea_vh_db - sigma0_slick_vh_db
    damping_ratio: float      # D_vv / D_vh
    vv_vh_ratio_db: float     # sigma0_vv_db - sigma0_vh_db inside slick
    boundary_gradient: float  # Mean Sobel gradient at slick boundary
    glcm_contrast: float      # GLCM textural contrast
    glcm_homogeneity: float   # GLCM textural homogeneity
    glcm_energy: float        # GLCM energy (uniformity)

class PolarimetricFeatureExtractor:
    """
    Computes physical polarimetric and textural metrics distinguishing mineral oils from biogenic films.
    """
    
    def __init__(self, glcm_levels: int = 16):
        self.glcm_levels = glcm_levels

    def extract_features(
        self,
        vv_db: np.ndarray,
        vh_db: np.ndarray,
        slick_mask: np.ndarray
    ) -> PolarimetricFeatures:
        """
        Extracts polarimetric damping and textural metrics given calibrated VV/VH arrays
        and a boolean mask of the candidate slick.
        """
        mask_bool = slick_mask.astype(bool)
        if not np.any(mask_bool):
            # Empty slick fallback
            return PolarimetricFeatures(
                mean_vv_db=-15.0, mean_vh_db=-25.0,
                sea_vv_db=-15.0, sea_vh_db=-25.0,
                vv_damping_db=0.0, vh_damping_db=0.0,
                damping_ratio=1.0, vv_vh_ratio_db=10.0,
                boundary_gradient=0.0, glcm_contrast=0.0,
                glcm_homogeneity=1.0, glcm_energy=1.0
            )

        # 1. Background clean sea reference extraction: dilated ring around slick
        dilated = binary_dilation(mask_bool, iterations=15)
        sea_ring = dilated & (~mask_bool)
        
        # If ring is empty (slick covers entire tile), fallback to non-slick area
        if not np.any(sea_ring):
            sea_ring = ~mask_bool
        if not np.any(sea_ring):
            sea_ring = mask_bool

        # 2. Mean decibel backscatters
        slick_vv = float(np.mean(vv_db[mask_bool]))
        slick_vh = float(np.mean(vh_db[mask_bool]))
        sea_vv = float(np.mean(vv_db[sea_ring]))
        sea_vh = float(np.mean(vh_db[sea_ring]))

        # 3. Damping in decibels (contrast)
        vv_damping_db = max(0.0, sea_vv - slick_vv)
        vh_damping_db = max(0.0, sea_vh - slick_vh)
        
        damping_ratio = vv_damping_db / (vh_damping_db + 1e-5)
        vv_vh_ratio_db = slick_vv - slick_vh

        # 4. Boundary gradient (steepness of edge transition)
        sobel_h = sobel(vv_db, axis=0)
        sobel_w = sobel(vv_db, axis=1)
        grad_mag = np.hypot(sobel_h, sobel_w)
        
        boundary = binary_dilation(mask_bool, iterations=2) & (~mask_bool)
        boundary_gradient = float(np.mean(grad_mag[boundary])) if np.any(boundary) else 0.0

        # 5. Textural metrics via normalized co-occurrence inside slick
        contrast, homogeneity, energy = self._compute_glcm_metrics(vv_db, mask_bool)

        return PolarimetricFeatures(
            mean_vv_db=round(slick_vv, 2),
            mean_vh_db=round(slick_vh, 2),
            sea_vv_db=round(sea_vv, 2),
            sea_vh_db=round(sea_vh, 2),
            vv_damping_db=round(vv_damping_db, 2),
            vh_damping_db=round(vh_damping_db, 2),
            damping_ratio=round(damping_ratio, 3),
            vv_vh_ratio_db=round(vv_vh_ratio_db, 2),
            boundary_gradient=round(boundary_gradient, 3),
            glcm_contrast=round(contrast, 3),
            glcm_homogeneity=round(homogeneity, 3),
            glcm_energy=round(energy, 3)
        )

    def _compute_glcm_metrics(
        self, image_db: np.ndarray, mask: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Fast discrete co-occurrence matrix on quantized radar intensities within slick patch.
        """
        # Quantize decibels [-35, 0] into [0, glcm_levels - 1]
        quantized = np.clip((image_db + 35.0) / 35.0 * (self.glcm_levels - 1), 0, self.glcm_levels - 1).astype(int)
        
        # Horizontal pairs inside mask
        m1 = mask[:, :-1] & mask[:, 1:]
        if not np.any(m1):
            return 0.1, 0.9, 0.8
            
        p_row = quantized[:, :-1][m1]
        p_col = quantized[:, 1:][m1]
        
        # Co-occurrence histogram
        glcm, _, _ = np.histogram2d(p_row, p_col, bins=self.glcm_levels, range=[[0, self.glcm_levels], [0, self.glcm_levels]])
        glcm_sum = np.sum(glcm)
        if glcm_sum == 0:
            return 0.1, 0.9, 0.8
        p = glcm / glcm_sum
        
        i_idx, j_idx = np.indices((self.glcm_levels, self.glcm_levels))
        
        # GLCM Contrast
        contrast = float(np.sum(p * np.square(i_idx - j_idx)))
        # GLCM Homogeneity
        homogeneity = float(np.sum(p / (1.0 + np.abs(i_idx - j_idx))))
        # GLCM Energy
        energy = float(np.sum(np.square(p)))
        
        return contrast, homogeneity, energy
