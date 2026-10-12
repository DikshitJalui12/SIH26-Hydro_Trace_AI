"""
Maritime Forensic Analytics & Numerical Modeling Engine
- Reverse Lagrangian Drift Engine (Agent 2)
- AIS Attribution & Dark Vessel Detection Engine (Agent 3)
"""

__version__ = "1.0.0"

from .models import (
    DriftMode,
    DriftOutputPayload,
    OriginUncertaintyCone,
    HindcastWindow,
    WeatherSummary,
    AISMessage,
    SpatioTemporalCorridor,
    VesselIdentity,
    KinematicViolations,
    RankedSuspect,
    AttributionReport
)
from .drift_engine import ReverseLagrangianDriftEngine
from .environment import (
    EnvironmentalProvider,
    GriddedEnvironmentalProvider,
    RealisticOceanicAtmosphericModel
)
from .integrator import RK4Integrator
from .monte_carlo import sample_particles_in_polygon, apply_turbulent_diffusion
from .uncertainty import calculate_95_confidence_ellipse
from .visualizer import generate_interactive_map_html
