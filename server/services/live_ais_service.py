"""
HydroTrace-AI Maritime Command Platform
Live AIS Fleet Radar Service
Manages real-time dynamic AIS vessel movements, kinematic track advancement, and anomaly surveillance.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional


class LiveAISFleetService:
    """Simulates and tracks live AIS dynamic vessel movements in the Malacca Strait & Indian EEZ."""

    def __init__(self):
        self._last_tick = time.time()
        # Seed realistic vessels around Malacca Strait & Andaman Sea
        self._fleet: Dict[str, Dict[str, Any]] = {
            "352001928": {
                "mmsi": "352001928",
                "imo": "9412345",
                "name": "PACIFIC VOYAGER",
                "type": "Crude Oil Tanker (VLCC)",
                "flag": "Panama",
                "flag_code": "PA",
                "dwt": 298500,
                "length_m": 333,
                "beam_m": 60,
                "draft_m": 20.5,
                "lat": 1.3410,
                "lon": 103.7820,
                "sog_kts": 8.4,
                "cog_deg": 284.0,
                "heading_deg": 285,
                "nav_status": "Underway Using Engine",
                "destination": "SINGAPORE -> FUJAIRAH",
                "eta": "2026-09-24 18:00 UTC",
                "risk_score": 0.94,
                "risk_tier": "PRIME SUSPECT (MARPOL VIOLATION)",
                "anomalies": [
                    "AIS Blackout: 02h 15m duration in discharge zone",
                    "Speed Drop: Decelerated from 14.8 to 8.2 kts during transit",
                    "Hydrodynamic Plume Coincidence: 98.4% spatial-temporal match"
                ],
                "last_ping_offset": 4,  # seconds ago
            },
            "636019842": {
                "mmsi": "636019842",
                "imo": "9387410",
                "name": "OCEAN STAR",
                "type": "Chemical / Oil Products Tanker",
                "flag": "Liberia",
                "flag_code": "LR",
                "dwt": 50500,
                "length_m": 183,
                "beam_m": 32,
                "draft_m": 12.2,
                "lat": 1.4120,
                "lon": 103.8890,
                "sog_kts": 12.6,
                "cog_deg": 105.0,
                "heading_deg": 106,
                "nav_status": "Underway Using Engine",
                "destination": "PORT KLANG -> JURONG",
                "eta": "2026-09-19 16:30 UTC",
                "risk_score": 0.38,
                "risk_tier": "SECONDARY (UPSTREAM CORRIDOR)",
                "anomalies": [
                    "Constant velocity maintained",
                    "No AIS dropout reported"
                ],
                "last_ping_offset": 8,
            },
            "211832000": {
                "mmsi": "211832000",
                "imo": "9729000",
                "name": "HAMBURG EXPRESS",
                "type": "Ultra Large Container Vessel (ULCV)",
                "flag": "Germany",
                "flag_code": "DE",
                "dwt": 199000,
                "length_m": 400,
                "beam_m": 59,
                "draft_m": 15.8,
                "lat": 1.2250,
                "lon": 103.9550,
                "sog_kts": 17.2,
                "cog_deg": 242.0,
                "heading_deg": 242,
                "nav_status": "Underway Using Engine",
                "destination": "SHANGHAI -> ROTTERDAM",
                "eta": "2026-10-04 06:00 UTC",
                "risk_score": 0.08,
                "risk_tier": "BENIGN TRANSIT",
                "anomalies": [],
                "last_ping_offset": 2,
            },
            "413289000": {
                "mmsi": "413289000",
                "imo": "9618822",
                "name": "YANGTZE FORTUNE",
                "type": "Bulk Carrier (Capesize)",
                "flag": "China",
                "flag_code": "CN",
                "dwt": 176000,
                "length_m": 292,
                "beam_m": 45,
                "draft_m": 18.1,
                "lat": 1.1890,
                "lon": 103.7120,
                "sog_kts": 11.4,
                "cog_deg": 292.0,
                "heading_deg": 290,
                "nav_status": "Underway Using Engine",
                "destination": "HEDLAND -> TIANJIN",
                "eta": "2026-09-28 22:00 UTC",
                "risk_score": 0.12,
                "risk_tier": "BENIGN TRANSIT",
                "anomalies": [],
                "last_ping_offset": 5,
            },
            "563001240": {
                "mmsi": "563001240",
                "imo": "9812400",
                "name": "SINGAPORE GUARDIAN",
                "type": "Coast Guard / Law Enforcement Cutter",
                "flag": "Singapore",
                "flag_code": "SG",
                "dwt": 2400,
                "length_m": 85,
                "beam_m": 14,
                "draft_m": 4.5,
                "lat": 1.3050,
                "lon": 103.8210,
                "sog_kts": 22.5,
                "cog_deg": 280.0,
                "heading_deg": 280,
                "nav_status": "Engaged in Patrol / Interception",
                "destination": "STRAIT PATROL SECTOR ALPHA",
                "eta": "ON STATION",
                "risk_score": 0.0,
                "risk_tier": "ENFORCEMENT UNIT",
                "anomalies": [],
                "last_ping_offset": 1,
            },
            "419001122": {
                "mmsi": "419001122",
                "imo": "9558833",
                "name": "ICGS VAJRA",
                "type": "Indian Coast Guard OPV",
                "flag": "India",
                "flag_code": "IN",
                "dwt": 2100,
                "length_m": 98,
                "beam_m": 15,
                "draft_m": 3.8,
                "lat": 1.2500,
                "lon": 103.6500,
                "sog_kts": 18.0,
                "cog_deg": 110.0,
                "heading_deg": 112,
                "nav_status": "Naval / EEZ Sovereign Escort",
                "destination": "PORT BLAIR -> MALACCA EEZ",
                "eta": "PATROL CONTINUOUS",
                "risk_score": 0.0,
                "risk_tier": "ENFORCEMENT UNIT",
                "anomalies": [],
                "last_ping_offset": 3,
            }
        }

    def update_kinematics(self) -> None:
        """Advance vessel positions realistically along COG vector based on elapsed time."""
        now = time.time()
        dt = max(0.1, min(10.0, now - self._last_tick))
        self._last_tick = now

        earth_radius = 6371000.0  # meters

        for mmsi, vessel in self._fleet.items():
            sog_mps = vessel["sog_kts"] * 0.514444
            cog_rad = math.radians(vessel["cog_deg"])
            dist = sog_mps * dt

            # Displacement
            d_lat = (dist * math.cos(cog_rad)) / earth_radius
            d_lon = (dist * math.sin(cog_rad)) / (earth_radius * math.cos(math.radians(vessel["lat"])))

            vessel["lat"] = round(vessel["lat"] + math.degrees(d_lat), 6)
            vessel["lon"] = round(vessel["lon"] + math.degrees(d_lon), 6)

            # Minor realistic heading/SOG wobble
            wobble = 0.1 * math.sin(now + float(mmsi[-3:]))
            vessel["sog_kts"] = round(max(0.5, vessel["sog_kts"] + wobble * 0.05), 1)

            # Keep vessels within reasonable bounds, loop if needed
            if vessel["lon"] < 103.4:
                vessel["lon"] = 104.1
            elif vessel["lon"] > 104.2:
                vessel["lon"] = 103.5

    def get_live_fleet(self, min_risk: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return snapshot of all vessels with updated kinematic coordinates."""
        self.update_kinematics()
        vessels = list(self._fleet.values())
        if min_risk is not None:
            vessels = [v for v in vessels if v["risk_score"] >= min_risk]
        return vessels

    def get_vessel_by_mmsi(self, mmsi: str) -> Optional[Dict[str, Any]]:
        self.update_kinematics()
        return self._fleet.get(mmsi)

    def search_fleet(self, query: str) -> List[Dict[str, Any]]:
        self.update_kinematics()
        q = query.strip().lower()
        results = []
        for v in self._fleet.values():
            if (
                q in v["mmsi"]
                or q in v["imo"]
                or q in v["name"].lower()
                or q in v["flag"].lower()
                or q in v["type"].lower()
            ):
                results.append(v)
        return results


# Singleton instance
live_ais_service = LiveAISFleetService()
