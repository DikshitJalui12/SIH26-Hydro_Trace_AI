"""
Geodesic transformations, spherical kinematics, and Coriolis leeway rotation.
"""

import numpy as np

# WGS84 Ellipsoid constants
WGS84_A = 6378137.0         # semi-major axis (meters)
WGS84_B = 6356752.314245    # semi-minor axis (meters)
WGS84_E2 = 1.0 - (WGS84_B**2 / WGS84_A**2) # eccentricity squared
R_EARTH = 6371000.0         # mean spherical radius (meters)


def latlon_to_metric_derivs(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    u_mps: np.ndarray,
    v_mps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert horizontal metric velocities (u_east, v_north) in m/s
    into spherical angular velocities (dlon/dt, dlat/dt) in degrees per second.
    Uses local meridional and parallel radii of curvature (WGS84 approximation).
    """
    lat_rad = np.radians(lat_deg)
    
    # Radii of curvature
    sin_lat = np.sin(lat_rad)
    den = np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    m_radius = (WGS84_A * (1.0 - WGS84_E2)) / (den**3)     # Meridian radius
    n_radius = WGS84_A / den                               # Prime vertical radius
    p_radius = np.maximum(n_radius * np.cos(lat_rad), 1.0) # Parallel radius (clamped to prevent div by 0 at poles)
    
    # dlat/dt and dlon/dt in deg/s
    dlat_dt = (v_mps / m_radius) * (180.0 / np.pi)
    dlon_dt = (u_mps / p_radius) * (180.0 / np.pi)
    
    return dlat_dt, dlon_dt


def compute_leeway_deflection(
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    latitude: np.ndarray,
    alpha: float = 0.03,
    deflection_angle_deg: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute aerodynamic surface leeway wind drift vector.
    
    Parameters:
    - u_wind, v_wind: 10m atmospheric wind components (m/s)
    - latitude: latitude in degrees (used to determine Coriolis deflection direction)
    - alpha: aerodynamic leeway drag factor (default: 0.03, i.e. 3% surface drift)
    - deflection_angle_deg: optional explicit deflection angle in degrees.
      If None, defaults to Coriolis/Ekman leeway deflection:
      +15° (deflection to the right of wind in Northern Hemisphere)
      -15° (deflection to the left of wind in Southern Hemisphere)
    
    Returns:
    - u_leeway, v_leeway in m/s
    """
    if deflection_angle_deg is None:
        # 15 degrees right in Northern Hemisphere, left in Southern Hemisphere
        # At equator (near zero), deflection transitions smoothly
        theta_deg = 15.0 * np.sign(latitude)
    else:
        theta_deg = np.full_like(u_wind, deflection_angle_deg, dtype=float)
        
    theta_rad = np.radians(theta_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    
    # Clockwise rotation:
    # u_rot = u * cos(theta) + v * sin(theta)
    # v_rot = -u * sin(theta) + v * cos(theta)
    u_rot = u_wind * cos_t + v_wind * sin_t
    v_rot = -u_wind * sin_t + v_wind * cos_t
    
    return alpha * u_rot, alpha * v_rot


def haversine_distance_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray
) -> float | np.ndarray:
    """
    Calculate great-circle distance between two points (or arrays of points) on Earth.
    """
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    
    return (R_EARTH * c) / 1000.0


def latlon_to_local_xy_km(
    lats: np.ndarray,
    lons: np.ndarray,
    ref_lat: float,
    ref_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert spherical lat/lon coordinates into local tangent plane metric coordinates (x_km, y_km)
    centered at (ref_lat, ref_lon) using equirectangular projection with latitude scaling.
    Suitable for local spatial statistics and covariance calculation.
    """
    ref_lat_rad = np.radians(ref_lat)
    y_km = (np.radians(lats - ref_lat) * R_EARTH) / 1000.0
    x_km = (np.radians(lons - ref_lon) * R_EARTH * np.cos(ref_lat_rad)) / 1000.0
    return x_km, y_km


def local_xy_km_to_latlon(
    x_km: np.ndarray,
    y_km: np.ndarray,
    ref_lat: float,
    ref_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Invert local tangent plane coordinates (x_km, y_km) back to spherical (lat, lon) in degrees.
    """
    ref_lat_rad = np.radians(ref_lat)
    lats = ref_lat + np.degrees((y_km * 1000.0) / R_EARTH)
    lons = ref_lon + np.degrees((x_km * 1000.0) / (R_EARTH * np.cos(ref_lat_rad)))
    return lats, lons
