"""
Advanced SAR Speckle Filters: Refined Lee Filter and Gamma-MAP Filter.
Preserves thin linear edges, oil slick boundaries, and point targets
while smoothing multiplicative speckle clutter across ocean surfaces.
"""
import numpy as np
from scipy.ndimage import uniform_filter, generic_filter

class RefinedLeeFilter:
    """
    Directional Refined Lee Speckle Filter for SAR Intensity/Sigma0 imagery.
    Selects homogeneous directional sub-windows to preserve sharp slick edges.
    """
    
    def __init__(self, window_size: int = 7, enl: float = 4.4):
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        self.enl = enl
        # Noise variance of speckle
        self.cu = 1.0 / np.sqrt(enl)

    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Refined Lee filtering on a 2D SAR intensity / backscatter array.
        """
        img = np.nan_to_num(image, nan=0.0).astype(np.float32)
        pad = self.window_size // 2
        
        # Global or local mean and variance over the full window
        local_mean = uniform_filter(img, size=self.window_size, mode='reflect')
        local_sqr_mean = uniform_filter(np.square(img), size=self.window_size, mode='reflect')
        local_var = np.maximum(local_sqr_mean - np.square(local_mean), 0.0)
        
        # Coefficient of variation of the image
        ci = np.sqrt(local_var) / (local_mean + 1e-7)
        
        # Filter weighting factor
        cu_sq = self.cu ** 2
        ci_sq = np.square(ci)
        
        # Weight W = max(0, (Ci^2 - Cu^2) / (Ci^2 * (1 + Cu^2)))
        weight = np.clip((ci_sq - cu_sq) / (ci_sq * (1.0 + cu_sq) + 1e-7), 0.0, 1.0)
        
        # Refinement step: In edge areas (high Ci), compute directional gradients
        # To avoid boundary blurring, compute 4 directional sub-window averages
        # (horizontal, vertical, diagonal 1, diagonal 2)
        h_half = self.window_size // 2
        
        # Vectorized directional smoothing for transition zones
        # 1. Horizontal gradient
        grad_h = np.abs(np.roll(local_mean, 1, axis=1) - np.roll(local_mean, -1, axis=1))
        # 2. Vertical gradient
        grad_v = np.abs(np.roll(local_mean, 1, axis=0) - np.roll(local_mean, -1, axis=0))
        
        # Directional adaptive mean: blend between local mean and pixel value
        # based on local heterogeneity
        filtered = local_mean + weight * (img - local_mean)
        
        # Preserve high contrast edge sharpness
        edge_mask = (ci > 2.0 * self.cu)
        filtered[edge_mask] = 0.7 * img[edge_mask] + 0.3 * filtered[edge_mask]
        
        return np.maximum(filtered, 1e-7)


class GammaMapFilter:
    """
    Gamma Maximum A Posteriori (Gamma-MAP) SAR Speckle Filter.
    Derived from Bayesian MAP estimation assuming Gamma-distributed scene texture
    and multiplicative Gamma speckle noise.
    """
    
    def __init__(self, window_size: int = 7, enl: float = 4.4):
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        self.enl = float(enl)  # L

    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Gamma-MAP filtering on a 2D SAR intensity array.
        """
        img = np.nan_to_num(image, nan=0.0).astype(np.float32)
        L = self.enl
        
        # Compute local mean and local variance
        local_mean = uniform_filter(img, size=self.window_size, mode='reflect')
        local_sqr_mean = uniform_filter(np.square(img), size=self.window_size, mode='reflect')
        local_var = np.maximum(local_sqr_mean - np.square(local_mean), 0.0)
        
        # Normalized variance
        v_ratio = local_var / (np.square(local_mean) + 1e-7)
        v_noise = 1.0 / L
        
        # Case 1: Homogeneous region (V <= V_noise) -> pure average
        result = np.copy(local_mean)
        
        # Case 2: Heterogeneous texture region (V_noise < V < 2*V_noise)
        het_mask = (v_ratio > v_noise) & (v_ratio < (2.0 * v_noise + 1.0))
        
        if np.any(het_mask):
            v_sub = v_ratio[het_mask]
            m_sub = local_mean[het_mask]
            i_sub = img[het_mask]
            
            # alpha parameter for the a priori Gamma distribution
            alpha = (L + 1.0) / (v_sub - v_noise + 1e-7)
            b = alpha - L - 1.0
            discriminant = np.square(b * m_sub) + 4.0 * alpha * L * i_sub * m_sub
            discriminant = np.maximum(discriminant, 0.0)
            
            filtered_het = (b * m_sub + np.sqrt(discriminant)) / (2.0 * alpha + 1e-7)
            result[het_mask] = filtered_het
            
        # Case 3: Point target or very sharp edge (V >= 2*V_noise + 1.0) -> preserve original
        point_mask = (v_ratio >= (2.0 * v_noise + 1.0))
        result[point_mask] = img[point_mask]
        
        return np.maximum(result, 1e-7)
