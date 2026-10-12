"""
Spatial uncertainty quantification, covariance decomposition, and 95% error ellipse computation.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
from .geodesics import latlon_to_local_xy_km, local_xy_km_to_latlon
from .models import CentroidCoord, OriginUncertaintyCone

# Chi-squared critical scale factor for 2-DOF 95% confidence level
# F(s^2) = 1 - exp(-s^2/2) = 0.95 => s = sqrt(-2 * ln(0.05)) ~= 2.44774688578
CHI2_2DOF_95 = 5.991464547107979
SCALE_95 = np.sqrt(CHI2_2DOF_95)


def calculate_95_confidence_ellipse(
    lats: np.ndarray,
    lons: np.ndarray,
    num_boundary_points: int = 64
) -> OriginUncertaintyCone:
    """
    Compute 95% spatial confidence ellipse from 2D particle ensemble.
    
    1. Metric projection centered at ensemble centroid
    2. Covariance matrix estimation & eigen-decomposition
    3. Semi-major and semi-minor axes scaled by chi2(2, 0.95)
    4. Orientation angle in degrees clockwise from True North
    5. GeoJSON Polygon representation of the bounding ellipse
    """
    mean_lat = float(np.mean(lats))
    mean_lon = float(np.mean(lons))
    
    # Project to local metric tangent plane (km)
    x_km, y_km = latlon_to_local_xy_km(lats, lons, ref_lat=mean_lat, ref_lon=mean_lon)
    
    # Covariance matrix (2x2)
    coords = np.vstack([x_km, y_km])
    cov = np.cov(coords)
    
    # Eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort in descending order (lambda_1 >= lambda_2)
    sort_indices = np.argsort(eigenvalues)[::-1]
    lambda_1 = max(float(eigenvalues[sort_indices[0]]), 1e-6)
    lambda_2 = max(float(eigenvalues[sort_indices[1]]), 1e-6)
    v1 = eigenvectors[:, sort_indices[0]] # primary axis [v1_x, v1_y]
    v2 = eigenvectors[:, sort_indices[1]] # secondary axis [v2_x, v2_y]
    
    # Semi-axes in kilometers at 95% confidence
    semi_major_km = SCALE_95 * np.sqrt(lambda_1)
    semi_minor_km = SCALE_95 * np.sqrt(lambda_2)
    
    # Orientation: angle of semi-major axis clockwise from North
    # v1_x is East (dx), v1_y is North (dy)
    angle_from_north_rad = np.arctan2(v1[0], v1[1])
    orientation_deg = float(np.degrees(angle_from_north_rad)) % 360.0
    
    # Discretize ellipse boundary
    theta = np.linspace(0, 2 * np.pi, num_boundary_points)
    circle_pts = np.vstack([semi_major_km * np.cos(theta), semi_minor_km * np.sin(theta)])
    
    # Rotate by eigenvector matrix: [x, y]^T = [v1, v2] * [a*cos(t), b*sin(t)]^T
    rot_matrix = np.column_stack([v1, v2])
    ellipse_metric = rot_matrix @ circle_pts
    
    # Convert metric back to spherical coordinates
    ell_lats, ell_lons = local_xy_km_to_latlon(
        ellipse_metric[0], ellipse_metric[1], ref_lat=mean_lat, ref_lon=mean_lon
    )
    
    # Close the polygon ring (first point = last point)
    ell_coords = [[round(float(lo), 6), round(float(la), 6)] for lo, la in zip(ell_lons, ell_lats)]
    if ell_coords[0] != ell_coords[-1]:
        ell_coords.append(ell_coords[0])
        
    origin_polygon = {
        "type": "Polygon",
        "coordinates": [ell_coords]
    }
    
    return OriginUncertaintyCone(
        centroid=CentroidCoord(latitude=round(mean_lat, 6), longitude=round(mean_lon, 6)),
        semi_major_axis_km=round(semi_major_km, 3),
        semi_minor_axis_km=round(semi_minor_km, 3),
        orientation_degrees=round(orientation_deg, 2),
        origin_polygon=origin_polygon
    )


def compute_trajectory_features(
    time_history_lats: np.ndarray, # shape (n_steps, n_particles)
    time_history_lons: np.ndarray, # shape (n_steps, n_particles)
    time_labels_iso: List[str],
    num_sample_tracks: int = 50
) -> Dict[str, Any]:
    """
    Construct GeoJSON FeatureCollection containing:
    1. Ensemble Mean Drift Path (LineString)
    2. Upper and Lower corridor envelopes / 95% corridor bounds
    3. Representative sample streamlines illustrating dispersal
    """
    n_steps, n_particles = time_history_lats.shape
    features = []
    
    # 1. Ensemble Mean Trajectory
    mean_lats = np.mean(time_history_lats, axis=1)
    mean_lons = np.mean(time_history_lons, axis=1)
    mean_coords = [[round(float(lo), 6), round(float(la), 6)] for lo, la in zip(mean_lons, mean_lats)]
    
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": mean_coords
        },
        "properties": {
            "type": "ensemble_mean_path",
            "start_time": time_labels_iso[0],
            "end_time": time_labels_iso[-1],
            "points_count": n_steps
        }
    })
    
    # 2. Representative sample particle tracks (subsampled to keep GeoJSON lightweight)
    subsample_idx = np.linspace(0, n_particles - 1, min(num_sample_tracks, n_particles), dtype=int)
    for p_idx in subsample_idx:
        p_lats = time_history_lats[:, p_idx]
        p_lons = time_history_lons[:, p_idx]
        p_coords = [[round(float(lo), 6), round(float(la), 6)] for lo, la in zip(p_lons, p_lats)]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": p_coords
            },
            "properties": {
                "type": "stochastic_particle_streamline",
                "particle_id": int(p_idx)
            }
        })
        
    return {
        "type": "FeatureCollection",
        "features": features
    }
