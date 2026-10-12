"""
Configuration and physical constants for the Vision AI Tier Oil Spill Detection System.
"""
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PreprocessingConfig:
    """SAR Radiometric and Speckle Filtering Configuration."""
    # Decibel normalization ranges [min_db, max_db]
    vv_db_range: Tuple[float, float] = (-30.0, 0.0)
    vh_db_range: Tuple[float, float] = (-35.0, -5.0)
    
    # Speckle filtering
    speckle_filter_type: str = "refined_lee"  # 'refined_lee' or 'gamma_map'
    filter_window_size: int = 7
    equivalent_number_of_looks: float = 4.4  # Nominal ENL for Sentinel-1 IW GRD
    
    # Tiling configuration for DL inference
    tile_size: int = 512
    tile_overlap: int = 64
    batch_size: int = 4

@dataclass
class WindGatingConfig:
    """Atmospheric Wind Gating Criteria (ECMWF ERA5 10m wind)."""
    min_wind_speed_ms: float = 2.0   # Below this: specular reflection / calm water false alarms
    max_wind_speed_ms: float = 12.0  # Above this: wave breaking / oil dispersion
    optimal_wind_range: Tuple[float, float] = (3.0, 10.0)

@dataclass
class DiscriminationConfig:
    """Lookalike and Surfactant Discrimination Thresholds."""
    # Cross-polarization damping ratio (VV_damping / VH_damping)
    # Mineral oil strongly damps both polarizations, with characteristic damping ratio
    min_damping_ratio_db: float = 3.0   # Min backscatter drop (contrast) in dB
    biogenic_damping_max_db: float = 7.0 # Biogenic films rarely damp > 6-7 dB
    
    # GLCM Textural contrast thresholds
    min_boundary_gradient: float = 0.15 # Mineral oils have sharp boundary gradients
    max_biogenic_homogeneity: float = 0.85 # Biogenic slicks are excessively smooth and diffuse

@dataclass
class FaysSpreadingConfig:
    """Hydrodynamic constants for Fay's 3-Phase Spreading Theory."""
    # Fluid physical properties
    seawater_density: float = 1025.0    # kg/m^3
    crude_oil_density: float = 860.0    # kg/m^3 (medium crude)
    gravity: float = 9.80665            # m/s^2
    seawater_kinematic_viscosity: float = 1.05e-6  # m^2/s at 20 deg C
    surface_tension_coeff: float = 0.025 # N/m (interfacial spreading coefficient sigma)
    
    # Fay's empirical scaling coefficients (k1, k2, k3)
    k1_gravity_inertia: float = 1.14
    k2_gravity_viscous: float = 0.98
    k3_surface_tension_viscous: float = 1.60
    
    # Typical nominal initial spill volume assumed when calculating age if unmeasured
    nominal_oil_thickness_microns: float = 10.0 # micrometers (1e-5 m) for dark radar slick

@dataclass
class SegmentationConfig:
    """Deep Learning Model Configuration."""
    model_architecture: str = "attention_unet" # 'attention_unet'
    in_channels: int = 2  # [VV, VH] or [VV, VH, VV-VH]
    num_classes: int = 1  # Binary segmentation
    base_filters: int = 32
    probability_threshold: float = 0.50
    min_slick_area_km2: float = 0.05  # Filter out microscopic noise patches < 50,000 m^2

@dataclass
class PipelineConfig:
    """Master Pipeline Configuration."""
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    wind_gating: WindGatingConfig = WindGatingConfig()
    discrimination: DiscriminationConfig = DiscriminationConfig()
    fays: FaysSpreadingConfig = FaysSpreadingConfig()
    segmentation: SegmentationConfig = SegmentationConfig()
    device: str = "cpu"  # 'cpu' or 'cuda'
