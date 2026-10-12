"""
SAR Lookalike Discrimination Engine.
Distinguishes genuine crude/mineral oil spills from biogenic films,
grease ice, and calm water areas using multi-attribute radar physics.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .era5_wind_gate import WindGatingResult
from .polarization_features import PolarimetricFeatures
from ..config import DiscriminationConfig

class SlickType(str, Enum):
    MINERAL_OIL = "MINERAL_OIL_SLICK"
    BIOGENIC_FILM = "LOOKALIKE_BIOGENIC_FILM"
    CALM_WATER = "LOOKALIKE_CALM_WATER"
    GREASE_ICE = "LOOKALIKE_GREASE_ICE"
    INDETERMINATE = "INDETERMINATE"

@dataclass
class SlickClassification:
    """Detailed discrimination assessment for candidate oil slick."""
    slick_type: SlickType
    is_genuine_oil: bool
    confidence_score: float  # 0.0 to 1.0
    rationale: List[str] = field(default_factory=list)

class LookalikeClassifier:
    """
    Evaluates polarimetric damping, textural gradients, and atmospheric wind fields
    to eliminate natural ocean lookalikes.
    """
    
    def __init__(self, config: Optional[DiscriminationConfig] = None):
        self.config = config or DiscriminationConfig()

    def classify_slick(
        self,
        features: PolarimetricFeatures,
        wind_result: WindGatingResult,
        area_km2: float
    ) -> SlickClassification:
        """
        Classifies candidate dark feature using SAR physical discrimination rules.
        """
        reasons: List[str] = []
        base_confidence = 0.50

        # Check 1: Wind Gating Physics
        if not wind_result.is_valid:
            if wind_result.status == "REJECTED_CALM_WATER":
                return SlickClassification(
                    slick_type=SlickType.CALM_WATER,
                    is_genuine_oil=False,
                    confidence_score=0.10,
                    rationale=[wind_result.reason, "Rejected: Specular reflection in low-wind zone."]
                )
            elif wind_result.status == "REJECTED_HIGH_WIND":
                return SlickClassification(
                    slick_type=SlickType.INDETERMINATE,
                    is_genuine_oil=False,
                    confidence_score=0.15,
                    rationale=[wind_result.reason, "Rejected: Extreme wind dispersion."]
                )
        else:
            reasons.append(f"Wind speed {wind_result.wind_speed_ms:.1f} m/s provides optimal Bragg sea background.")
            base_confidence += 0.15

        # Check 2: Polarimetric Damping Contrast (VV damping)
        # Mineral oil typically shows > 3.0 to 10.0 dB backscatter attenuation
        if features.vv_damping_db >= self.config.min_damping_ratio_db:
            reasons.append(f"Significant VV backscatter attenuation ({features.vv_damping_db:.1f} dB >= {self.config.min_damping_ratio_db} dB).")
            base_confidence += 0.15
        else:
            reasons.append(f"Weak VV backscatter attenuation ({features.vv_damping_db:.1f} dB < {self.config.min_damping_ratio_db} dB).")
            base_confidence -= 0.20

        # Check 3: Boundary Sharpness (Sobel gradient)
        # Mineral oil has a sharp physicochemical boundary with clean water;
        # Biogenic films are monomolecular and diffuse gradually into ocean surface
        if features.boundary_gradient >= self.config.min_boundary_gradient:
            reasons.append(f"Sharp boundary gradient ({features.boundary_gradient:.2f}) characteristic of viscous petroleum.")
            base_confidence += 0.10
        else:
            reasons.append(f"Diffuse boundary gradient ({features.boundary_gradient:.2f}) typical of biogenic surfactant.")
            base_confidence -= 0.15

        # Check 4: Textural Homogeneity (GLCM)
        # Biogenic films have extremely high homogeneity (> 0.85) without internal thickness variations
        is_biogenic = (
            features.glcm_homogeneity > self.config.max_biogenic_homogeneity
            and features.vv_damping_db < self.config.biogenic_damping_max_db
            and features.boundary_gradient < self.config.min_boundary_gradient
        )

        if is_biogenic:
            reasons.append("High textural homogeneity and mild damping indicate monomolecular biogenic film.")
            return SlickClassification(
                slick_type=SlickType.BIOGENIC_FILM,
                is_genuine_oil=False,
                confidence_score=round(max(0.15, min(0.35, base_confidence)), 2),
                rationale=reasons
            )

        # Check 5: Extremely low temperatures / polar cross-pol (Grease Ice check)
        # Grease ice has high VH cross-polarization backscatter relative to open water
        if features.vh_damping_db < 0.5 and features.vv_damping_db > 5.0:
            reasons.append("Anomalous cross-polarization signature consistent with grease ice slurry.")
            return SlickClassification(
                slick_type=SlickType.GREASE_ICE,
                is_genuine_oil=False,
                confidence_score=0.20,
                rationale=reasons
            )

        # Genuine Mineral Oil Spills
        final_confidence = round(max(0.20, min(0.98, base_confidence)), 2)
        is_genuine = final_confidence >= 0.60
        slick_type = SlickType.MINERAL_OIL if is_genuine else SlickType.INDETERMINATE

        if is_genuine:
            reasons.append("Multi-polarization damping and morphological boundary confirm genuine mineral oil spill.")

        return SlickClassification(
            slick_type=slick_type,
            is_genuine_oil=is_genuine,
            confidence_score=final_confidence,
            rationale=reasons
        )
