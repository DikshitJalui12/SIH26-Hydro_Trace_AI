"""
Global Vessel Registry & MARPOL Risk Profiler.
Resolves IMO, MMSI, vessel type, flag state, and owner details,
and assesses inherent MARPOL Annex I discharge risk factors.
"""

from typing import Dict, Any, Optional
from engine.models import VesselIdentity


class VesselRegistry:
    """
    Simulated Global Maritime Registry (IHS Markit / Equasis / GISIS equivalent)
    with MARPOL Annex I vessel type risk profiling and flag state safety performance.
    """

    # Vessel type baseline risk for illegal oily discharge under MARPOL Annex I
    TYPE_RISK_MAP = {
        "Crude Oil Tanker": 1.00,
        "Chemical/Products Tanker": 0.95,
        "Product Tanker": 0.95,
        "Oil/Chemical Tanker": 0.95,
        "Bulk Carrier": 0.75,
        "Ore Carrier": 0.75,
        "Container Ship": 0.65,
        "General Cargo": 0.60,
        "Vehicle Carrier": 0.50,
        "Tug / Offshore Support": 0.30,
        "Fishing Vessel": 0.25,
        "Passenger Ship": 0.15,
        "Unknown": 0.50
    }

    # Paris MoU / Tokyo MoU Flag State Risk Ratings (High Risk / Medium Risk / Quality)
    FLAG_RISK_MAP = {
        "Togo": 1.25,
        "Comoros": 1.25,
        "Cameroon": 1.25,
        "Guyana": 1.20,
        "Palau": 1.15,
        "Panama": 1.05,
        "Liberia": 1.00,
        "Marshall Islands": 1.00,
        "Singapore": 0.95,
        "Bahamas": 1.00,
        "Malta": 1.00,
        "Hong Kong": 0.95,
        "Norway": 0.90,
        "United Kingdom": 0.90
    }

    def __init__(self):
        # In-memory registry database seeded with realistic commercial vessels
        self._registry: Dict[str, Dict[str, Any]] = {
            "563001230": {
                "imo": "9421831",
                "mmsi": "563001230",
                "name": "OCEAN APEX",
                "flag": "Panama",
                "vessel_type": "Crude Oil Tanker",
                "owner": "Apex Maritime Holdings SA",
                "operator": "PetroTrans Global Shipping Ltd",
                "dwt": 158500,
                "build_year": 2011
            },
            "636018920": {
                "imo": "9312042",
                "mmsi": "636018920",
                "name": "NORDIC GIANT",
                "flag": "Liberia",
                "vessel_type": "Bulk Carrier",
                "owner": "Nordic Bulk Carriers AS",
                "operator": "Scandic Ocean Lines",
                "dwt": 82000,
                "build_year": 2008
            },
            "477210980": {
                "imo": "9684122",
                "mmsi": "477210980",
                "name": "PACIFIC BRIDGE",
                "flag": "Hong Kong",
                "vessel_type": "Container Ship",
                "owner": "Orient Express Line Corp",
                "operator": "Pacific Freight Logistics",
                "dwt": 114000,
                "build_year": 2016
            },
            "525004110": {
                "imo": "9182344",
                "mmsi": "525004110",
                "name": "SEA FALCON",
                "flag": "Togo",
                "vessel_type": "Chemical/Products Tanker",
                "owner": "Volga Marine Services Ltd",
                "operator": "Black Sea Bunkering SA",
                "dwt": 47000,
                "build_year": 2002
            },
            "533009840": {
                "imo": "9503342",
                "mmsi": "533009840",
                "name": "MALACCA LEADER",
                "flag": "Singapore",
                "vessel_type": "Tug / Offshore Support",
                "owner": "Singa Marine Logistics Pte Ltd",
                "operator": "Singa Port Services",
                "dwt": 3200,
                "build_year": 2014
            },
            "354112000": {
                "imo": "9287651",
                "mmsi": "354112000",
                "name": "AURORA VOYAGER",
                "flag": "Comoros",
                "vessel_type": "Crude Oil Tanker",
                "owner": "Poseidon Horizon Ltd",
                "operator": "Unknown Offshore Entity",
                "dwt": 105000,
                "build_year": 2004
            }
        }

    @staticmethod
    def validate_imo(imo: str) -> bool:
        """Validate IMO 7-digit number using official checksum algorithm."""
        cleaned = imo.replace("IMO", "").strip()
        if not (cleaned.isdigit() and len(cleaned) == 7):
            return False
        digits = [int(d) for d in cleaned]
        checksum = sum(digits[i] * (7 - i) for i in range(6))
        return (checksum % 10) == digits[6]

    def lookup(self, mmsi: str, fallback_telemetry: Optional[Dict[str, Any]] = None) -> VesselIdentity:
        """Resolve vessel identity by MMSI, falling back to telemetry metadata if absent."""
        if mmsi in self._registry:
            entry = self._registry[mmsi]
            return VesselIdentity(
                imo=entry["imo"],
                mmsi=entry["mmsi"],
                name=entry["name"],
                flag=entry["flag"],
                vessel_type=entry["vessel_type"],
                owner=entry["owner"],
                operator=entry.get("operator"),
                dwt=entry.get("dwt"),
                build_year=entry.get("build_year")
            )

        # Fallback from AIS telemetry
        fallback_telemetry = fallback_telemetry or {}
        imo = fallback_telemetry.get("imo") or "UNKNOWN"
        name = fallback_telemetry.get("vessel_name") or f"MMSI-{mmsi}"
        v_type = fallback_telemetry.get("vessel_type") or "Unknown"

        return VesselIdentity(
            imo=str(imo),
            mmsi=str(mmsi),
            name=str(name),
            flag="Unknown",
            vessel_type=str(v_type),
            owner="Unknown Commercial Operator"
        )

    def calculate_vessel_profile_risk(self, identity: VesselIdentity) -> float:
        """
        Calculate inherent vessel risk score (0.0 to 1.0) based on
        MARPOL Annex I discharge capacity, flag state safety index, and vessel age.
        """
        base_risk = self.TYPE_RISK_MAP.get(identity.vessel_type, 0.50)
        flag_multiplier = self.FLAG_RISK_MAP.get(identity.flag, 1.00)

        # Age risk factor (vessels > 15 years old have higher OWS failure / piping bypass rates)
        age_multiplier = 1.0
        if identity.build_year:
            age = max(0, 2026 - identity.build_year)
            if age > 20:
                age_multiplier = 1.20
            elif age > 15:
                age_multiplier = 1.10

        composite_risk = min(1.0, base_risk * flag_multiplier * age_multiplier)
        return round(composite_risk, 3)
