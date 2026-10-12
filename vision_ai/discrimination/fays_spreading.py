"""
Fay's 3-Phase Hydrodynamic Spreading Theory and Oil Slick Weathering Age Estimator.
Implements analytical hydrodynamic spreading regimes:
1. Gravity-Inertia Regime
2. Gravity-Viscous Regime
3. Surface Tension-Viscous Regime
Inverts observed SAR surface area to calculate slick age (hours) and weathering stage.
"""
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from ..config import FaysSpreadingConfig

class SpreadingRegime(str, Enum):
    GRAVITY_INERTIA = "gravity-inertia"
    GRAVITY_VISCOUS = "gravity-viscous"
    SURFACE_TENSION_VISCOUS = "surface-tension-viscous"

@dataclass
class FaysAgeEstimate:
    """Estimated hydrodynamic age and weathering characteristics of the oil slick."""
    estimated_age_hours: float
    estimated_age_seconds: float
    spreading_regime: SpreadingRegime
    weathering_stage: str
    effective_radius_m: float
    estimated_volume_m3: float

class FaysSpreadingEstimator:
    """
    Computes oil slick growth and inverts observed SAR morphological area
    to estimate spill age using Fay's spreading theory.
    """
    
    def __init__(self, config: Optional[FaysSpreadingConfig] = None):
        self.config = config or FaysSpreadingConfig()
        
        # Reduced gravity: delta * g
        rho_w = self.config.seawater_density
        rho_o = self.config.crude_oil_density
        self.delta = (rho_w - rho_o) / rho_w
        self.delta_g = self.delta * self.config.gravity
        self.nu_w = self.config.seawater_kinematic_viscosity
        self.sigma = self.config.surface_tension_coeff

    def compute_regime_transitions(self, volume_m3: float) -> Tuple[float, float, float, float]:
        """
        Computes transition times t12, t23 and radii r12, r23 between Fay's spreading phases.
        Returns: (t12_sec, t23_sec, r12_m, r23_m)
        """
        k1 = self.config.k1_gravity_inertia
        k2 = self.config.k2_gravity_viscous
        k3 = self.config.k3_surface_tension_viscous

        # Transition t12: gravity-inertia to gravity-viscous
        t12 = (k2 / k1)**4 * (volume_m3 / (self.delta_g * self.nu_w + 1e-12))**(1.0 / 3.0)
        r12 = k1 * (self.delta_g * volume_m3)**(0.25) * (t12**0.5)

        # Transition t23: gravity-viscous to surface-tension-viscous
        rho_w = self.config.seawater_density
        num = (rho_w * self.delta_g)**2 * (volume_m3**(5.0 / 3.0))
        den = (self.nu_w**0.2) * (self.sigma**1.2) + 1e-12
        t23 = (k2 / k3)**2.4 * (num / den)
        
        c3 = (self.sigma**2 / (rho_w**2 * self.nu_w + 1e-12))**0.25
        r23 = k3 * c3 * (t23**0.75)

        return t12, t23, r12, r23

    def estimate_age_from_area(
        self,
        area_km2: float,
        volume_m3: Optional[float] = None,
        slick_thickness_microns: Optional[float] = None
    ) -> FaysAgeEstimate:
        """
        Inverts observed slick surface area to estimate age (in hours) and weathering regime.
        """
        area_m2 = area_km2 * 1e6
        effective_radius_m = math.sqrt(max(1.0, area_m2 / math.pi))
        
        thickness_m = (slick_thickness_microns or self.config.nominal_oil_thickness_microns) * 1e-6
        if volume_m3 is None:
            # Estimate volume from area and nominal thickness
            volume_m3 = area_m2 * thickness_m
            
        t12, t23, r12, r23 = self.compute_regime_transitions(volume_m3)
        
        k1 = self.config.k1_gravity_inertia
        k2 = self.config.k2_gravity_viscous
        k3 = self.config.k3_surface_tension_viscous
        rho_w = self.config.seawater_density

        # Determine spreading regime based on effective radius
        if effective_radius_m <= r12 or r12 <= 0:
            regime = SpreadingRegime.GRAVITY_INERTIA
            # r = k1 * (delta * g * V)^(1/4) * t^(1/2)
            # t = [ r / (k1 * (delta * g * V)^(1/4)) ]^2
            denom = k1 * (self.delta_g * volume_m3)**0.25 + 1e-7
            t_sec = (effective_radius_m / denom)**2
            weathering_stage = "Initial Discharge (Gravity-Inertial Spreading)"

        elif effective_radius_m <= r23 or r23 <= r12:
            regime = SpreadingRegime.GRAVITY_VISCOUS
            # r = k2 * (delta * g * V^2 / nu^(1/2))^(1/6) * t^(1/4)
            # t = [ r / (k2 * ...) ]^4
            factor = (self.delta_g * (volume_m3**2) / (math.sqrt(self.nu_w) + 1e-12))**(1.0 / 6.0)
            denom = k2 * factor + 1e-7
            t_sec = (effective_radius_m / denom)**4
            weathering_stage = "Intermediate Spreading (Gravity-Viscous Dissipation)"

        else:
            regime = SpreadingRegime.SURFACE_TENSION_VISCOUS
            # r = k3 * (sigma^2 / (rho_w^2 * nu))^(1/4) * t^(3/4)
            # t = [ r / (k3 * ...) ]^(4/3)
            factor = (self.sigma**2 / (rho_w**2 * self.nu_w + 1e-12))**0.25
            denom = k3 * factor + 1e-7
            t_sec = (effective_radius_m / denom)**(4.0 / 3.0)
            weathering_stage = "Mature Weathered Sheen (Surface Tension-Viscous Dissipation)"

        # Ensure realistic lower and upper physical bounds
        t_sec = max(60.0, min(t_sec, 30.0 * 24.0 * 3600.0))  # Between 1 minute and 30 days
        age_hours = round(t_sec / 3600.0, 2)

        return FaysAgeEstimate(
            estimated_age_hours=age_hours,
            estimated_age_seconds=round(t_sec, 1),
            spreading_regime=regime,
            weathering_stage=weathering_stage,
            effective_radius_m=round(effective_radius_m, 1),
            estimated_volume_m3=round(volume_m3, 2)
        )
