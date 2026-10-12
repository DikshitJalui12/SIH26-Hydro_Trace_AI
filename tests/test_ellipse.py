"""
Tests for 95% confidence spatial error ellipse and uncertainty cones.
"""

import numpy as np
import pytest
from shapely.geometry import shape, Point
from engine.uncertainty import calculate_95_confidence_ellipse, SCALE_95
from engine.geodesics import local_xy_km_to_latlon


def test_ellipse_known_covariance():
    """
    Test ellipse math against known analytical bivariate normal distribution.
    sigma_x = 3.0 km, sigma_y = 1.5 km
    """
    ref_lat = 15.0
    ref_lon = 70.0
    n_samples = 40000
    
    rng = np.random.default_rng(999)
    sigma_x_km = 3.0
    sigma_y_km = 1.5
    
    x_km = rng.normal(0.0, sigma_x_km, size=n_samples)
    y_km = rng.normal(0.0, sigma_y_km, size=n_samples)
    
    lats, lons = local_xy_km_to_latlon(x_km, y_km, ref_lat=ref_lat, ref_lon=ref_lon)
    
    cone = calculate_95_confidence_ellipse(lats, lons)
    
    expected_major_km = SCALE_95 * sigma_x_km # ~ 7.343 km
    expected_minor_km = SCALE_95 * sigma_y_km # ~ 3.672 km
    
    assert np.isclose(cone.semi_major_axis_km, expected_major_km, rtol=0.03)
    assert np.isclose(cone.semi_minor_axis_km, expected_minor_km, rtol=0.03)
    
    # Check that ~95% of points fall inside the resulting GeoJSON polygon
    poly = shape(cone.origin_polygon)
    
    # Subsample 1000 points to test containment
    test_idx = np.random.choice(n_samples, size=1000, replace=False)
    inside_count = 0
    for idx in test_idx:
        if poly.contains(Point(lons[idx], lats[idx])):
            inside_count += 1
            
    containment_pct = inside_count / 1000.0
    # Expected is 0.95 (+/- 0.02)
    assert 0.93 <= containment_pct <= 0.97, f"Containment pct {containment_pct} out of expected 95% range"
