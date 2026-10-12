"""
Synthetic AIS Telemetry Generator for Forensic Verification.
Simulates realistic commercial vessel traffic in the Malacca Strait TSS
with controlled MARPOL Annex I violation signatures and compliant controls.
"""

from datetime import datetime, timedelta
import math
from typing import List, Dict, Any


def generate_scenario_telemetry() -> List[Dict[str, Any]]:
    """
    Generates AIS telemetry points for 5 vessels in the Singapore/Malacca Strait:
    1. M/T OCEAN APEX (Crude Tanker): Primary suspect with speed drop to 5.2 kts, loitering, and deliberate AIS blackout.
    2. M/T AURORA VOYAGER (Crude Tanker, Comoros flag): Secondary vessel with minor gap and congestion slowdown.
    3. C/V PACIFIC BRIDGE (Container Ship): High speed TSS transit, fully compliant.
    4. M/V NORDIC GIANT (Bulk Carrier): Steady bulk transit, fully compliant.
    5. MALACCA LEADER (Tug/Supply): Steady slow workboat speed, low risk.
    """
    records: List[Dict[str, Any]] = []

    # Scenario parameters: Slick detected at 01°21'N, 103°27'E at 2026-09-18T02:00:00Z
    t_start = datetime.fromisoformat("2026-09-18T00:00:00")

    # -------------------------------------------------------------
    # Vessel 1: M/T OCEAN APEX (MMSI: 563001230, IMO: 9421831)
    # -------------------------------------------------------------
    # Phase 1: Cruising at 15.4 knots (00:00 to 01:25)
    lat = 1.1500
    lon = 103.1000
    curr_t = t_start
    while curr_t < datetime.fromisoformat("2026-09-18T01:25:00"):
        records.append({
            "timestamp": curr_t,
            "mmsi": "563001230",
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "sog": 15.4,
            "cog": 65.0,
            "heading": 66.0,
            "nav_status": 0,
            "imo": "9421831",
            "vessel_name": "OCEAN APEX",
            "vessel_type": "Crude Oil Tanker"
        })
        lat += 0.0014
        lon += 0.0032
        curr_t += timedelta(minutes=2)

    # Phase 2: AIS BLACKOUT GAP (01:25 to 02:12 -> 47 minutes silence)
    lat = 1.2090
    lon = 103.2680

    # Phase 3: Reappears at 02:12 UTC emerging from the envelope, accelerating from 5.2 kts to 14.8 kts
    curr_t = datetime.fromisoformat("2026-09-18T02:12:00")
    speeds = [5.2, 5.8, 7.2, 9.5, 12.0, 14.8, 15.0]
    for spd in speeds:
        records.append({
            "timestamp": curr_t,
            "mmsi": "563001230",
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "sog": spd,
            "cog": 72.0,
            "heading": 73.0,
            "nav_status": 0,
            "imo": "9421831",
            "vessel_name": "OCEAN APEX",
            "vessel_type": "Crude Oil Tanker"
        })
        lat += 0.0010
        lon += 0.0025
        curr_t += timedelta(minutes=3)

    # Continued cruising past 02:30 to 04:00
    while curr_t <= datetime.fromisoformat("2026-09-18T04:00:00"):
        records.append({
            "timestamp": curr_t,
            "mmsi": "563001230",
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "sog": 15.0,
            "cog": 65.0,
            "heading": 65.0,
            "nav_status": 0,
            "imo": "9421831",
            "vessel_name": "OCEAN APEX",
            "vessel_type": "Crude Oil Tanker"
        })
        lat += 0.0014
        lon += 0.0032
        curr_t += timedelta(minutes=2)

    # -------------------------------------------------------------
    # Vessel 2: M/T AURORA VOYAGER (MMSI: 354112000, IMO: 9287651)
    # -------------------------------------------------------------
    curr_t = t_start
    lat = 1.1000
    lon = 103.0500
    while curr_t <= datetime.fromisoformat("2026-09-18T04:00:00"):
        if datetime.fromisoformat("2026-09-18T02:15:00") <= curr_t <= datetime.fromisoformat("2026-09-18T02:45:00"):
            sog = 9.2
        else:
            sog = 13.5

        if not (datetime.fromisoformat("2026-09-18T02:20:00") < curr_t < datetime.fromisoformat("2026-09-18T02:42:00")):
            records.append({
                "timestamp": curr_t,
                "mmsi": "354112000",
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "sog": sog,
                "cog": 68.0,
                "heading": 69.0,
                "nav_status": 0,
                "imo": "9287651",
                "vessel_name": "AURORA VOYAGER",
                "vessel_type": "Crude Oil Tanker"
            })
        lat += 0.0012
        lon += 0.0028
        curr_t += timedelta(minutes=2)

    # -------------------------------------------------------------
    # Vessel 3: C/V PACIFIC BRIDGE (MMSI: 477210980, IMO: 9684122)
    # -------------------------------------------------------------
    curr_t = t_start
    lat = 1.1800
    lon = 103.0000
    while curr_t <= datetime.fromisoformat("2026-09-18T04:00:00"):
        records.append({
            "timestamp": curr_t,
            "mmsi": "477210980",
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "sog": 19.5,
            "cog": 66.0,
            "heading": 66.0,
            "nav_status": 0,
            "imo": "9684122",
            "vessel_name": "PACIFIC BRIDGE",
            "vessel_type": "Container Ship"
        })
        lat += 0.0018
        lon += 0.0040
        curr_t += timedelta(minutes=2)

    # -------------------------------------------------------------
    # Vessel 4: M/V NORDIC GIANT (MMSI: 636018920, IMO: 9312042)
    # -------------------------------------------------------------
    curr_t = t_start
    lat = 1.1400
    lon = 103.0200
    while curr_t <= datetime.fromisoformat("2026-09-18T04:00:00"):
        records.append({
            "timestamp": curr_t,
            "mmsi": "636018920",
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "sog": 12.8,
            "cog": 65.5,
            "heading": 65.0,
            "nav_status": 0,
            "imo": "9312042",
            "vessel_name": "NORDIC GIANT",
            "vessel_type": "Bulk Carrier"
        })
        lat += 0.0011
        lon += 0.0026
        curr_t += timedelta(minutes=2)

    # -------------------------------------------------------------
    # Vessel 5: MALACCA LEADER (MMSI: 533009840, IMO: 9503342)
    # -------------------------------------------------------------
    curr_t = t_start
    lat = 1.2300
    lon = 103.3200
    while curr_t <= datetime.fromisoformat("2026-09-18T04:00:00"):
        records.append({
            "timestamp": curr_t,
            "mmsi": "533009840",
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "sog": 5.2,
            "cog": 110.0,
            "heading": 112.0,
            "nav_status": 0,
            "imo": "9503342",
            "vessel_name": "MALACCA LEADER",
            "vessel_type": "Tug / Offshore Support"
        })
        lat -= 0.0002
        lon += 0.0006
        curr_t += timedelta(minutes=2)

    return records
