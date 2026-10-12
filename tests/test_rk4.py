"""
Unit tests for RK4 numerical integrator and geodesic kinematics.
"""

import numpy as np
import pytest
from engine.environment import EnvironmentalProvider
from engine.integrator import RK4Integrator
from engine.geodesics import (
    latlon_to_metric_derivs,
    compute_leeway_deflection,
    haversine_distance_km
)
from engine.models import WeatherSummary


class MockConstantFlowEnv(EnvironmentalProvider):
    """Uniform steady current and wind for analytical verification."""
    def __init__(self, u_curr=0.5, v_curr=0.5, u_wind=10.0, v_wind=0.0, sst=25.0):
        self.u_curr = u_curr
        self.v_curr = v_curr
        self.u_wind = u_wind
        self.v_wind = v_wind
        self.sst = sst

    def sample_vectors(self, lat, lon, time_epoch_s):
        n = len(lat)
        return (
            np.full(n, self.u_curr),
            np.full(n, self.v_curr),
            np.full(n, self.u_wind),
            np.full(n, self.v_wind),
            np.full(n, self.sst)
        )

    def get_weather_summary(self, bbox, t_start_s, t_end_s):
        return WeatherSummary(
            avg_current_knots=0.97,
            avg_wind_knots=19.44,
            sea_surface_temp=self.sst
        )


def test_coriolis_deflection_signs():
    u_w = np.array([0.0])
    v_w = np.array([10.0]) # Wind blowing due North
    
    # Northern hemisphere (lat > 0): should deflect to the right (positive u_lee > 0, East)
    u_lee_nh, v_lee_nh = compute_leeway_deflection(u_w, v_w, latitude=np.array([20.0]), alpha=0.03)
    assert u_lee_nh[0] > 0.0, "Northern Hemisphere leeway should deflect clockwise (Eastward)"
    assert v_lee_nh[0] > 0.0
    
    # Southern hemisphere (lat < 0): should deflect to the left (negative u_lee < 0, West)
    u_lee_sh, v_lee_sh = compute_leeway_deflection(u_w, v_w, latitude=np.array([-20.0]), alpha=0.03)
    assert u_lee_sh[0] < 0.0, "Southern Hemisphere leeway should deflect counter-clockwise (Westward)"


def test_rk4_time_reversibility():
    """
    Test that stepping forward by dt then backward by dt returns to initial position.
    """
    env = MockConstantFlowEnv()
    integrator = RK4Integrator(env, alpha_leeway=0.03, base_step_minutes=15.0)
    
    init_lat = np.array([20.0])
    init_lon = np.array([65.0])
    t0 = 1700000000.0
    dt_s = 900.0 # 15 minutes
    
    # Step forward
    fwd_lat, fwd_lon = integrator.rk4_step(init_lat, init_lon, t0, dt_s, is_reverse=False)
    
    # Step backward from resulting position
    back_lat, back_lon = integrator.rk4_step(fwd_lat, fwd_lon, t0 + dt_s, -dt_s, is_reverse=False)
    
    dist_m = haversine_distance_km(init_lat, init_lon, back_lat, back_lon) * 1000.0
    assert dist_m < 0.01, f"RK4 reversibility error too high: {dist_m} meters"
