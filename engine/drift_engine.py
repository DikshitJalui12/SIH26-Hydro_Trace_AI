"""
Reverse Lagrangian Drift Engine orchestrator.
Handles dual-mode operation: Reverse Hindcasting and Forward Containment Forecasting.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import numpy as np
from shapely.geometry import shape

from .models import (
    DriftMode,
    DriftOutputPayload,
    HindcastWindow,
    WeatherSummary,
    OriginUncertaintyCone
)
from .environment import EnvironmentalProvider, RealisticOceanicAtmosphericModel
from .integrator import RK4Integrator
from .monte_carlo import sample_particles_in_polygon, apply_turbulent_diffusion
from .uncertainty import calculate_95_confidence_ellipse, compute_trajectory_features


class ReverseLagrangianDriftEngine:
    """
    Physical Oceanographic Inverse Transport Modeling Engine.
    Executes 10,000-particle stochastic Lagrangian backtracking and forward forecasting.
    """

    def __init__(
        self,
        env_provider: Optional[EnvironmentalProvider] = None,
        alpha_leeway: float = 0.03,
        deflection_angle_deg: Optional[float] = None,
        step_minutes: int = 15,
        num_particles: int = 10000,
        eddy_diffusivity_kh: float = 10.0,
        seed: Optional[int] = 42
    ):
        self.env_provider = env_provider
        self.alpha_leeway = alpha_leeway
        self.deflection_angle_deg = deflection_angle_deg
        self.step_minutes = step_minutes
        self.num_particles = num_particles
        self.eddy_diffusivity_kh = eddy_diffusivity_kh
        self.seed = seed

    def run(
        self,
        slick_geojson: Dict[str, Any],
        detection_time_iso: str,
        duration_hours: float = 12.0,
        mode: DriftMode = DriftMode.REVERSE_HINDCAST,
        slick_id: Optional[str] = None
    ) -> DriftOutputPayload:
        """
        Execute Lagrangian drift simulation.
        
        Parameters:
        - slick_geojson: GeoJSON Feature or Geometry of the detected slick
        - detection_time_iso: ISO 8601 timestamp of satellite acquisition (t = T)
        - duration_hours: Simulation duration (e.g. 12h for hindcast, up to 48h for forecast)
        - mode: REVERSE_HINDCAST (default) or FORWARD_FORECAST
        - slick_id: Optional identifier matching Agent 1 detection
        """
        # Parse GeoJSON geometry
        if "geometry" in slick_geojson:
            geom_dict = slick_geojson["geometry"]
            props = slick_geojson.get("properties", {})
            inferred_slick_id = props.get("slick_id", "SLICK-SAR-001")
        else:
            geom_dict = slick_geojson
            inferred_slick_id = "SLICK-SAR-001"
            
        final_slick_id = slick_id or inferred_slick_id
        
        # Parse timestamp (support UTC ISO strings)
        t_detect = datetime.fromisoformat(detection_time_iso.replace("Z", "+00:00"))
        if t_detect.tzinfo is None:
            t_detect = t_detect.replace(tzinfo=timezone.utc)
            
        t_detect_epoch_s = t_detect.timestamp()
        
        # Setup environmental provider if not already configured
        geom_obj = shape(geom_dict)
        min_lon, min_lat, max_lon, max_lat = geom_obj.bounds
        
        # Estimate drift bounding box (at ~1.5 knots, 48h drift is max ~140 km ~= 1.3 degrees)
        drift_margin_deg = (duration_hours / 24.0) * 1.2 + 0.5
        expanded_bbox = (
            min_lon - drift_margin_deg,
            min_lat - drift_margin_deg,
            max_lon + drift_margin_deg,
            max_lat + drift_margin_deg
        )
        
        t_span_s = duration_hours * 3600.0
        if mode == DriftMode.REVERSE_HINDCAST:
            t_start_epoch_s = t_detect_epoch_s - t_span_s
            t_end_epoch_s = t_detect_epoch_s
            dt_s = -(self.step_minutes * 60.0)
            is_reverse = True
        else:
            t_start_epoch_s = t_detect_epoch_s
            t_end_epoch_s = t_detect_epoch_s + t_span_s
            dt_s = (self.step_minutes * 60.0)
            is_reverse = False
            
        env = self.env_provider
        if env is None:
            # Build high-resolution 1/12° CMEMS / ERA5 hydrodynamic reanalysis provider
            base_model = RealisticOceanicAtmosphericModel()
            env = base_model.to_gridded_provider(
                bbox=expanded_bbox,
                t_start_s=min(t_start_epoch_s, t_end_epoch_s),
                t_end_s=max(t_start_epoch_s, t_end_epoch_s),
                spatial_res_deg=1.0 / 12.0
            )
            
        # Seed 10,000 Monte Carlo stochastic particles within slick polygon
        init_lats, init_lons = sample_particles_in_polygon(
            geojson_geom=geom_dict,
            num_particles=self.num_particles,
            seed=self.seed
        )
        
        # Initialize RK4 Integrator
        integrator = RK4Integrator(
            env_provider=env,
            alpha_leeway=self.alpha_leeway,
            deflection_angle_deg=self.deflection_angle_deg,
            base_step_minutes=self.step_minutes
        )
        
        # Time-stepping setup
        n_steps = int(round(t_span_s / (self.step_minutes * 60.0)))
        
        # Particle trajectories over time: shape (n_steps + 1, num_particles)
        history_lats = np.zeros((n_steps + 1, self.num_particles), dtype=np.float64)
        history_lons = np.zeros((n_steps + 1, self.num_particles), dtype=np.float64)
        time_labels_iso = []
        
        current_lats = init_lats.copy()
        current_lons = init_lons.copy()
        current_t_s = t_detect_epoch_s
        
        history_lats[0] = current_lats
        history_lons[0] = current_lons
        time_labels_iso.append(datetime.fromtimestamp(current_t_s, tz=timezone.utc).isoformat())
        
        rng = np.random.default_rng(self.seed)
        
        # Integration loop
        for step in range(1, n_steps + 1):
            # 1. 4th-Order Runge-Kutta advection step
            step_dt = dt_s
            next_lats, next_lons = integrator.rk4_step(
                lat=current_lats,
                lon=current_lons,
                t_epoch_s=current_t_s,
                dt_s=step_dt,
                is_reverse=is_reverse
            )
            
            # 2. Horizontal Brownian turbulent random-walk dispersion (Kh = 10 m^2/s)
            diff_lats, diff_lons = apply_turbulent_diffusion(
                lats=next_lats,
                lons=next_lons,
                dt_s=abs(step_dt),
                kh=self.eddy_diffusivity_kh,
                rng=rng
            )
            
            current_lats = diff_lats
            current_lons = diff_lons
            current_t_s += step_dt
            
            history_lats[step] = current_lats
            history_lons[step] = current_lons
            time_labels_iso.append(datetime.fromtimestamp(current_t_s, tz=timezone.utc).isoformat())
            
        # 3. Compute 95% Confidence Spatial Ellipse at terminal origin / forecast state
        terminal_lats = history_lats[-1]
        terminal_lons = history_lons[-1]
        uncertainty_cone = calculate_95_confidence_ellipse(terminal_lats, terminal_lons)
        
        # 4. Construct trajectory corridor features (FeatureCollection)
        drift_path_fc = compute_trajectory_features(
            time_history_lats=history_lats,
            time_history_lons=history_lons,
            time_labels_iso=time_labels_iso,
            num_sample_tracks=50
        )
        
        # 5. Compute Weather Summary over regional domain and duration
        weather = env.get_weather_summary(
            bbox=expanded_bbox,
            t_start_s=min(t_start_epoch_s, t_end_epoch_s),
            t_end_s=max(t_start_epoch_s, t_end_epoch_s)
        )
        
        # Construct HindcastWindow
        start_utc_str = datetime.fromtimestamp(t_start_epoch_s, tz=timezone.utc).isoformat()
        end_utc_str = datetime.fromtimestamp(t_end_epoch_s, tz=timezone.utc).isoformat()
        
        window = HindcastWindow(
            start_utc=start_utc_str,
            end_utc=end_utc_str,
            step_minutes=self.step_minutes
        )
        
        metadata = {
            "particles_count": self.num_particles,
            "eddy_diffusivity_kh_m2s": self.eddy_diffusivity_kh,
            "wind_leeway_coefficient_alpha": self.alpha_leeway,
            "integration_scheme": "Runge-Kutta-4th-Order (RK4)",
            "advection_equation": "dx/dt = -(u_curr + alpha*W_wind)" if is_reverse else "dx/dt = +(u_curr + alpha*W_wind)",
            "coriolis_deflection": "automatic_latitudinal" if self.deflection_angle_deg is None else f"{self.deflection_angle_deg} deg",
            "confidence_level": "95% (chi2_2dof=5.991)"
        }
        
        return DriftOutputPayload(
            slick_id=final_slick_id,
            mode=mode,
            hindcast_window=window,
            origin_uncertainty_cone=uncertainty_cone,
            drift_path_linestring=drift_path_fc,
            weather_summary=weather,
            metadata=metadata
        )
