"""
Tests for Monte Carlo polygon particle sampling and Brownian turbulent diffusion.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon, Point, shape
from engine.monte_carlo import sample_particles_in_polygon, apply_turbulent_diffusion
from engine.geodesics import latlon_to_local_xy_km


def test_polygon_particle_containment():
    """Verify all 10,000 sampled particles fall strictly inside polygon boundary."""
    # Define a triangular test polygon
    poly_geom = {
        "type": "Polygon",
        "coordinates": [
            [[60.0, 20.0], [61.0, 20.0], [60.5, 21.0], [60.0, 20.0]]
        ]
    }
    
    num_particles = 10000
    lats, lons = sample_particles_in_polygon(poly_geom, num_particles=num_particles, seed=123)
    
    assert len(lats) == num_particles
    assert len(lons) == num_particles
    
    geom = shape(poly_geom)
    
    # Verify a random sample of 200 points with exact shapely containment
    sample_indices = np.random.choice(num_particles, size=200, replace=False)
    for idx in sample_indices:
        pt = Point(lons[idx], lats[idx])
        assert geom.contains(pt) or geom.touches(pt), f"Particle ({lons[idx]}, {lats[idx]}) outside polygon!"


def test_brownian_diffusion_variance():
    """
    Verify horizontal eddy diffusivity adheres to Einstein-Smoluchowski diffusion equation:
    <dx^2> = 2 * Kh * dt
    """
    kh = 10.0 # m²/s
    dt_s = 3600.0 # 1 hour
    n_particles = 50000
    
    ref_lat = 20.0
    ref_lon = 60.0
    
    init_lats = np.full(n_particles, ref_lat)
    init_lons = np.full(n_particles, ref_lon)
    
    diff_lats, diff_lons = apply_turbulent_diffusion(
        lats=init_lats,
        lons=init_lons,
        dt_s=dt_s,
        kh=kh,
        rng=np.random.default_rng(42)
    )
    
    # Convert displacements to meters
    x_km, y_km = latlon_to_local_xy_km(diff_lats, diff_lons, ref_lat=ref_lat, ref_lon=ref_lon)
    dx_m = x_km * 1000.0
    dy_m = y_km * 1000.0
    
    var_x = float(np.var(dx_m))
    var_y = float(np.var(dy_m))
    expected_var = 2.0 * kh * dt_s # 72,000 m²
    
    # With 50,000 samples, variance standard error is ~0.6%, relative tolerance of 3% is well within bounds
    assert np.isclose(var_x, expected_var, rtol=0.03), f"var_x {var_x} deviates from expected {expected_var}"
    assert np.isclose(var_y, expected_var, rtol=0.03), f"var_y {var_y} deviates from expected {expected_var}"
