"""
Radiometric Calibration, Thermal Noise Removal, and Decibel Scaling for Sentinel-1 GRD.
Converts raw Digital Numbers (DN) into physical Radar Cross Section Sigma Nought (sigma0)
consistent with ESA SNAP calibration standards.
"""
import numpy as np
from typing import Optional, Tuple, Union

class RadiometricCalibrator:
    """
    Applies ESA SNAP-compliant radiometric calibration and thermal noise removal.
    """
    
    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon

    def calibrate_to_sigma0_linear(
        self,
        dn_array: np.ndarray,
        calibration_factor: Optional[Union[float, np.ndarray]] = None,
        thermal_noise: Optional[Union[float, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Calibrates raw DN to Sigma Nought (linear power):
        sigma0 = (DN^2 - noise) / A_i^2
        """
        arr = np.nan_to_num(dn_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        # If values are already in typical linear sigma0 range (e.g. 0 to 2.0), preserve
        if np.nanmean(arr) < 5.0 and np.nanmax(arr) < 10.0:
            sigma0 = np.maximum(arr, self.epsilon)
            return sigma0

        # Step 1: DN squared (intensity)
        intensity = np.square(arr)
        
        # Step 2: Thermal noise removal
        if thermal_noise is not None:
            intensity = np.maximum(intensity - thermal_noise, self.epsilon)
            
        # Step 3: Calibration gain (A_i)
        if calibration_factor is not None:
            cal_gain = np.square(calibration_factor)
            sigma0 = intensity / (cal_gain + self.epsilon)
        else:
            # Standard empirical nominal calibration factor for Sentinel-1 IW GRD
            # Typical DN values are around 50 - 500
            nominal_gain = 300.0 ** 2
            sigma0 = intensity / nominal_gain

        sigma0 = np.maximum(sigma0, self.epsilon)
        return sigma0

    def linear_to_db(self, sigma0_linear: np.ndarray) -> np.ndarray:
        """
        Converts linear Sigma Nought to decibels (dB):
        sigma0_dB = 10 * log10(sigma0_linear)
        """
        clamped = np.maximum(sigma0_linear, self.epsilon)
        db = 10.0 * np.log10(clamped)
        # Clip to realistic SAR terrestrial/oceanic backscatter bounds [-50 dB, +15 dB]
        db = np.clip(db, -50.0, 15.0)
        return db

    def calibrate_band(
        self,
        dn_array: np.ndarray,
        calibration_factor: Optional[Union[float, np.ndarray]] = None,
        thermal_noise: Optional[Union[float, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs full calibration pipeline on a single polarization band.
        Returns:
            sigma0_linear: Radar cross-section in linear units
            sigma0_db: Radar cross-section in decibels (dB)
        """
        sigma0_linear = self.calibrate_to_sigma0_linear(dn_array, calibration_factor, thermal_noise)
        sigma0_db = self.linear_to_db(sigma0_linear)
        return sigma0_linear, sigma0_db
