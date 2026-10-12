"""
4th-Order Runge-Kutta (RK4) numerical time integrator with adaptive time-stepping.
Supports both backward (hindcast) and forward (forecast) advective trajectories.
"""

from typing import Tuple, Callable
import numpy as np
from .environment import EnvironmentalProvider
from .geodesics import latlon_to_metric_derivs, compute_leeway_deflection


class RK4Integrator:
    """
    Vectorized 4th-Order Runge-Kutta ODE solver on spherical WGS84 geodesics.
    """

    def __init__(
        self,
        env_provider: EnvironmentalProvider,
        alpha_leeway: float = 0.03,
        deflection_angle_deg: float | None = None,
        base_step_minutes: float = 15.0,
        min_step_minutes: float = 1.0,
        spatial_error_tol_m: float = 10.0
    ):
        self.env = env_provider
        self.alpha_leeway = alpha_leeway
        self.deflection_angle_deg = deflection_angle_deg
        self.base_step_s = base_step_minutes * 60.0
        self.min_step_s = min_step_minutes * 60.0
        self.spatial_error_tol_m = spatial_error_tol_m

    def compute_velocity_field(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        t_epoch_s: float,
        is_reverse: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate advection velocity vector:
        Forward: v = u_curr + alpha * W_wind(deflected)
        Reverse: v = - (u_curr + alpha * W_wind(deflected))
        Returns:
            dlat_dt, dlon_dt in degrees per second
        """
        u_curr, v_curr, u_wind, v_wind, _ = self.env.sample_vectors(lat, lon, t_epoch_s)
        
        # Leeway with Coriolis deflection
        u_lee, v_lee = compute_leeway_deflection(
            u_wind=u_wind,
            v_wind=v_wind,
            latitude=lat,
            alpha=self.alpha_leeway,
            deflection_angle_deg=self.deflection_angle_deg
        )
        
        # Combined hydrodynamic and aerodynamic drift
        u_total = u_curr + u_lee
        v_total = v_curr + v_lee
        
        if is_reverse:
            # Reverse Lagrangian backtracking: dx/dt = - (u + alpha*W)
            u_total = -u_total
            v_total = -v_total
            
        dlat_dt, dlon_dt = latlon_to_metric_derivs(lat, lon, u_total, v_total)
        return dlat_dt, dlon_dt

    def rk4_step(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        t_epoch_s: float,
        dt_s: float,
        is_reverse: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single classical 4th-Order Runge-Kutta step over interval dt_s.
        """
        # k1
        k1_lat, k1_lon = self.compute_velocity_field(lat, lon, t_epoch_s, is_reverse)
        
        # k2
        lat_k2 = lat + 0.5 * dt_s * k1_lat
        lon_k2 = lon + 0.5 * dt_s * k1_lon
        t_k2 = t_epoch_s + 0.5 * dt_s
        k2_lat, k2_lon = self.compute_velocity_field(lat_k2, lon_k2, t_k2, is_reverse)
        
        # k3
        lat_k3 = lat + 0.5 * dt_s * k2_lat
        lon_k3 = lon + 0.5 * dt_s * k2_lon
        t_k3 = t_epoch_s + 0.5 * dt_s
        k3_lat, k3_lon = self.compute_velocity_field(lat_k3, lon_k3, t_k3, is_reverse)
        
        # k4
        lat_k4 = lat + dt_s * k3_lat
        lon_k4 = lon + dt_s * k3_lon
        t_k4 = t_epoch_s + dt_s
        k4_lat, k4_lon = self.compute_velocity_field(lat_k4, lon_k4, t_k4, is_reverse)
        
        # RK4 weighted average
        lat_next = lat + (dt_s / 6.0) * (k1_lat + 2.0 * k2_lat + 2.0 * k3_lat + k4_lat)
        lon_next = lon + (dt_s / 6.0) * (k1_lon + 2.0 * k2_lon + 2.0 * k3_lon + k4_lon)
        
        return lat_next, lon_next

    def adaptive_step(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        t_epoch_s: float,
        dt_target_s: float,
        is_reverse: bool
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Adaptive time-stepping via step-doubling error estimation.
        Takes one full step dt, vs two half steps (dt/2).
        If estimated spatial error exceeds tolerance, halves step down to min_step_s.
        """
        current_dt = dt_target_s
        
        while abs(current_dt) >= self.min_step_s:
            # 1 full step
            lat_full, lon_full = self.rk4_step(lat, lon, t_epoch_s, current_dt, is_reverse)
            
            # 2 half steps
            half_dt = current_dt / 2.0
            lat_h1, lon_h1 = self.rk4_step(lat, lon, t_epoch_s, half_dt, is_reverse)
            lat_half, lon_half = self.rk4_step(lat_h1, lon_h1, t_epoch_s + half_dt, half_dt, is_reverse)
            
            # Estimate spatial discrepancy in meters (~111,000 m per deg)
            dlat_m = (lat_half - lat_full) * 111139.0
            dlon_m = (lon_half - lon_full) * 111139.0 * np.cos(np.radians(lat))
            max_error_m = np.max(np.hypot(dlat_m, dlon_m))
            
            if max_error_m <= self.spatial_error_tol_m or abs(current_dt) <= self.min_step_s * 1.05:
                # Accept step using higher-order Richardson extrapolation estimate
                # x_accurate = x_half + (x_half - x_full) / 15
                lat_accepted = lat_half + (lat_half - lat_full) / 15.0
                lon_accepted = lon_half + (lon_half - lon_full) / 15.0
                return lat_accepted, lon_accepted, current_dt
            else:
                # Halve step size
                current_dt /= 2.0
                
        # If reached min step, return result from min step
        return lat_half, lon_half, current_dt
