"""
Monte Carlo particle seeding and Brownian turbulent random-walk dispersion.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely import contains_xy
from .geodesics import R_EARTH


def sample_particles_in_polygon(
    geojson_geom: Dict[str, Any],
    num_particles: int = 10000,
    seed: int | None = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly seed stochastic particles across an input GeoJSON Polygon or MultiPolygon.
    
    Returns:
        lats: np.ndarray of shape (num_particles,)
        lons: np.ndarray of shape (num_particles,)
    """
    if seed is not None:
        np.random.seed(seed)
        
    geom = shape(geojson_geom)
    if not geom.is_valid:
        geom = geom.buffer(0) # fix non-simple self-intersecting geometries
        
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    
    sampled_lons: List[float] = []
    sampled_lats: List[float] = []
    
    # Batch rejection sampling with vectorized GEOS contains_xy
    batch_size = max(num_particles * 2, 20000)
    
    while len(sampled_lons) < num_particles:
        cand_lons = np.random.uniform(min_lon, max_lon, batch_size)
        cand_lats = np.random.uniform(min_lat, max_lat, batch_size)
        
        # Vectorized containment test (C-level GEOS in Shapely 2.0+)
        inside_mask = contains_xy(geom, cand_lons, cand_lats)
        valid_lons = cand_lons[inside_mask]
        valid_lats = cand_lats[inside_mask]
        
        sampled_lons.extend(valid_lons.tolist())
        sampled_lats.extend(valid_lats.tolist())
        
    final_lons = np.array(sampled_lons[:num_particles], dtype=np.float64)
    final_lats = np.array(sampled_lats[:num_particles], dtype=np.float64)
    
    return final_lats, final_lons


def apply_turbulent_diffusion(
    lats: np.ndarray,
    lons: np.ndarray,
    dt_s: float,
    kh: float = 10.0, # 10 m²/s horizontal eddy diffusivity
    rng: np.random.Generator | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject Brownian turbulent random-walk dispersion:
    dX_turb = sqrt(2 * Kh * |dt|) * Normal(0, 1)
    
    Converts metric diffusion into spherical degree displacements.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    sigma_metric = np.sqrt(2.0 * kh * abs(dt_s))
    
    # 2D Gaussian standard normal
    dx_m = rng.normal(0.0, sigma_metric, size=lats.shape)
    dy_m = rng.normal(0.0, sigma_metric, size=lats.shape)
    
    # Convert metric displacement to angular displacement (degrees)
    lat_rad = np.radians(lats)
    dlat_deg = (dy_m / R_EARTH) * (180.0 / np.pi)
    dlon_deg = (dx_m / (R_EARTH * np.cos(lat_rad))) * (180.0 / np.pi)
    
    return lats + dlat_deg, lons + dlon_deg
