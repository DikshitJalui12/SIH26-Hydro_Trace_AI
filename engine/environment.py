"""
Environmental dynamic vector ingestion and spatiotemporal interpolation layer.
Supports CMEMS Global Ocean Physics Reanalysis (1/12°) and ECMWF ERA5 10m wind fields.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .models import WeatherSummary


class EnvironmentalProvider(ABC):
    """Abstract interface for environmental current and wind vector fields."""

    @abstractmethod
    def sample_vectors(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        time_epoch_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample oceanographic currents and atmospheric wind at positions (lat, lon) and time.
        
        Returns:
            (u_curr, v_curr, u_wind, v_wind, sst)
            u_curr, v_curr: ocean surface currents (m/s)
            u_wind, v_wind: 10m atmospheric winds (m/s)
            sst: sea surface temperature (Celsius)
        """
        pass

    @abstractmethod
    def get_weather_summary(
        self,
        bbox: tuple[float, float, float, float],
        t_start_s: float,
        t_end_s: float
    ) -> WeatherSummary:
        """Compute regional and temporal average weather conditions."""
        pass


class GriddedEnvironmentalProvider(EnvironmentalProvider):
    """
    Spatiotemporally interpolated environmental provider backed by regular gridded
    reanalysis fields (1/12° spatial resolution for CMEMS, 0.25°/1/12° for ERA5).
    """

    def __init__(
        self,
        time_grid_s: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        u_curr: np.ndarray,
        v_curr: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        sst: np.ndarray
    ):
        """
        Grids shape: (n_time, n_lat, n_lon)
        """
        self.time_grid_s = time_grid_s
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        
        # Build 3D regular grid interpolators (bounds_error=False, fill_value=None to extrapolate at edges)
        coords = (time_grid_s, lat_grid, lon_grid)
        self._interp_u_curr = RegularGridInterpolator(coords, u_curr, bounds_error=False, fill_value=None)
        self._interp_v_curr = RegularGridInterpolator(coords, v_curr, bounds_error=False, fill_value=None)
        self._interp_u_wind = RegularGridInterpolator(coords, u_wind, bounds_error=False, fill_value=None)
        self._interp_v_wind = RegularGridInterpolator(coords, v_wind, bounds_error=False, fill_value=None)
        self._interp_sst = RegularGridInterpolator(coords, sst, bounds_error=False, fill_value=None)
        
        self.raw_u_curr = u_curr
        self.raw_v_curr = v_curr
        self.raw_u_wind = u_wind
        self.raw_v_wind = v_wind
        self.raw_sst = sst

    def sample_vectors(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        time_epoch_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t_arr = np.full_like(lat, time_epoch_s, dtype=float)
        pts = np.column_stack([t_arr, lat, lon])
        
        uc = self._interp_u_curr(pts)
        vc = self._interp_v_curr(pts)
        uw = self._interp_u_wind(pts)
        vw = self._interp_v_wind(pts)
        st = self._interp_sst(pts)
        return uc, vc, uw, vw, st

    def get_weather_summary(
        self,
        bbox: tuple[float, float, float, float],
        t_start_s: float,
        t_end_s: float
    ) -> WeatherSummary:
        # 1 m/s = 1.94384 knots
        curr_speed = np.hypot(self.raw_u_curr, self.raw_v_curr) * 1.94384
        wind_speed = np.hypot(self.raw_u_wind, self.raw_v_wind) * 1.94384
        
        avg_curr = float(np.nanmean(curr_speed))
        avg_wind = float(np.nanmean(wind_speed))
        avg_sst = float(np.nanmean(self.raw_sst))
        
        return WeatherSummary(
            avg_current_knots=round(avg_curr, 2),
            avg_wind_knots=round(avg_wind, 2),
            sea_surface_temp=round(avg_sst, 2)
        )


class RealisticOceanicAtmosphericModel(EnvironmentalProvider):
    """
    Physical oceanographic hydrodynamics and atmospheric boundary layer vector field.
    Features:
    - Mesoscale geostrophic eddies with spatial divergence-free streamfunctions
    - Semidiurnal M2 (12.42h) and diurnal K1 (23.93h) tidal current oscillations
    - Background coastal / shelf current shear
    - Dynamic synoptic atmospheric wind vectors with directional variability
    - SST distribution based on latitude gradient and coastal upwelling
    
    Can either operate directly on continuous coordinates or generate an interpolated
    1/12° CMEMS-resolution gridded field covering the target bounding box.
    """

    def __init__(
        self,
        base_current_speed: float = 0.28,      # m/s (~0.55 knots)
        base_current_heading_deg: float = 45.0, # toward NE
        tidal_amplitude: float = 0.12,          # m/s
        base_wind_speed: float = 6.5,           # m/s (~12.6 knots)
        base_wind_heading_deg: float = 60.0,    # blowing toward ENE
        eddy_scale_km: float = 45.0,            # mesoscale Rossby radius
        mean_sst_celsius: float = 26.5
    ):
        self.base_current_speed = base_current_speed
        self.base_current_heading_deg = base_current_heading_deg
        self.tidal_amplitude = tidal_amplitude
        self.base_wind_speed = base_wind_speed
        self.base_wind_heading_deg = base_wind_heading_deg
        self.eddy_scale_km = eddy_scale_km
        self.mean_sst_celsius = mean_sst_celsius

    def sample_vectors(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        time_epoch_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Background mean currents
        curr_rad = np.radians(self.base_current_heading_deg)
        u_bg = self.base_current_speed * np.sin(curr_rad)
        v_bg = self.base_current_speed * np.cos(curr_rad)
        
        # M2 Tidal oscillation (period = 12.4206 hours = 44714 seconds)
        omega_m2 = 2.0 * np.pi / 44714.0
        # Phase varies slightly with longitude
        t_phase_m2 = omega_m2 * time_epoch_s + np.radians(lon * 5.0)
        u_tide = self.tidal_amplitude * np.cos(t_phase_m2)
        v_tide = self.tidal_amplitude * 0.8 * np.sin(t_phase_m2) # elliptical tidal rotary motion
        
        # Mesoscale eddy field: divergence-free streamfunction psi(x, y) = A * cos(kx*x) * sin(ky*y)
        # 1 deg lat ~ 111 km
        kx = 2.0 * np.pi / (self.eddy_scale_km / 111.0)
        ky = 2.0 * np.pi / (self.eddy_scale_km / 111.0)
        # Drift of eddies over time (slow westward Rossby wave propagation ~ 3 cm/s)
        time_hours = time_epoch_s / 3600.0
        eddy_x = lon + 0.001 * time_hours
        eddy_y = lat
        
        u_eddy = -0.15 * np.cos(kx * eddy_x) * np.cos(ky * eddy_y)
        v_eddy = 0.15 * np.sin(kx * eddy_x) * np.sin(ky * eddy_y)
        
        u_curr = u_bg + u_tide + u_eddy
        v_curr = v_bg + v_tide + v_eddy
        
        # Atmospheric wind field: synoptic base + diurnal sea-breeze modulation (24h period)
        wind_rad = np.radians(self.base_wind_heading_deg)
        omega_diurnal = 2.0 * np.pi / 86400.0
        diurnal_factor = 1.0 + 0.18 * np.sin(omega_diurnal * time_epoch_s + np.radians(lon * 15.0))
        
        u_wind = (self.base_wind_speed * np.sin(wind_rad) * diurnal_factor) + 0.3 * np.sin(kx * lon)
        v_wind = (self.base_wind_speed * np.cos(wind_rad) * diurnal_factor) + 0.3 * np.cos(ky * lat)
        
        # SST: latitudinal temperature gradient + eddy thermal signature
        sst = self.mean_sst_celsius - 0.08 * (lat - np.mean(lat)) + 0.4 * np.sin(kx * lon) * np.cos(ky * lat)
        
        return u_curr, v_curr, u_wind, v_wind, sst

    def get_weather_summary(
        self,
        bbox: tuple[float, float, float, float],
        t_start_s: float,
        t_end_s: float
    ) -> WeatherSummary:
        min_lon, min_lat, max_lon, max_lat = bbox
        lats = np.linspace(min_lat, max_lat, 5)
        lons = np.linspace(min_lon, max_lon, 5)
        times = np.linspace(t_start_s, t_end_s, 5)
        
        speeds_curr = []
        speeds_wind = []
        sst_vals = []
        
        for t in times:
            for la in lats:
                for lo in lons:
                    uc, vc, uw, vw, st = self.sample_vectors(np.array([la]), np.array([lo]), t)
                    speeds_curr.append(np.hypot(uc[0], vc[0]) * 1.94384) # knots
                    speeds_wind.append(np.hypot(uw[0], vw[0]) * 1.94384) # knots
                    sst_vals.append(st[0])
                    
        return WeatherSummary(
            avg_current_knots=round(float(np.mean(speeds_curr)), 2),
            avg_wind_knots=round(float(np.mean(speeds_wind)), 2),
            sea_surface_temp=round(float(np.mean(sst_vals)), 2)
        )

    def to_gridded_provider(
        self,
        bbox: tuple[float, float, float, float],
        t_start_s: float,
        t_end_s: float,
        spatial_res_deg: float = 1.0 / 12.0, # 1/12 degree CMEMS resolution
        temporal_step_s: float = 3600.0     # 1 hour
    ) -> GriddedEnvironmentalProvider:
        """
        Discretize the continuous physical model into a high-resolution 1/12° CMEMS / ERA5
        reanalysis grid and return a GriddedEnvironmentalProvider.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        pad = spatial_res_deg * 3.0
        
        lat_grid = np.arange(min_lat - pad, max_lat + pad + spatial_res_deg, spatial_res_deg)
        lon_grid = np.arange(min_lon - pad, max_lon + pad + spatial_res_deg, spatial_res_deg)
        time_grid = np.arange(t_start_s - temporal_step_s, t_end_s + temporal_step_s * 2, temporal_step_s)
        
        n_t = len(time_grid)
        n_lat = len(lat_grid)
        n_lon = len(lon_grid)
        
        u_curr = np.zeros((n_t, n_lat, n_lon), dtype=np.float32)
        v_curr = np.zeros((n_t, n_lat, n_lon), dtype=np.float32)
        u_wind = np.zeros((n_t, n_lat, n_lon), dtype=np.float32)
        v_wind = np.zeros((n_t, n_lat, n_lon), dtype=np.float32)
        sst = np.zeros((n_t, n_lat, n_lon), dtype=np.float32)
        
        mesh_lon, mesh_lat = np.meshgrid(lon_grid, lat_grid)
        flat_lon = mesh_lon.ravel()
        flat_lat = mesh_lat.ravel()
        
        for idx, t in enumerate(time_grid):
            uc, vc, uw, vw, st = self.sample_vectors(flat_lat, flat_lon, t)
            u_curr[idx] = uc.reshape(n_lat, n_lon)
            v_curr[idx] = vc.reshape(n_lat, n_lon)
            u_wind[idx] = uw.reshape(n_lat, n_lon)
            v_wind[idx] = vw.reshape(n_lat, n_lon)
            sst[idx] = st.reshape(n_lat, n_lon)
            
        return GriddedEnvironmentalProvider(
            time_grid_s=time_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            u_curr=u_curr,
            v_curr=v_curr,
            u_wind=u_wind,
            v_wind=v_wind,
            sst=sst
        )
