"""
DuckDB Historical AIS Telemetry Store.
Provides spatio-temporal corridor querying, spatial indexing,
and candidate trajectory extraction for forensic attribution.
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import duckdb
import pandas as pd
from engine.models import AISMessage, SpatioTemporalCorridor


class TelemetryStore:
    def __init__(self, db_path: str = ":memory:"):
        """Initialize DuckDB connection and schema."""
        self.con = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        """Create AIS telemetry table and indexes."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ais_telemetry (
                timestamp TIMESTAMP NOT NULL,
                mmsi VARCHAR NOT NULL,
                lat DOUBLE NOT NULL,
                lon DOUBLE NOT NULL,
                sog DOUBLE NOT NULL,
                cog DOUBLE NOT NULL,
                heading DOUBLE,
                nav_status INTEGER,
                imo VARCHAR,
                vessel_name VARCHAR,
                vessel_type VARCHAR
            );
        """)
        # Create indexes for fast spatio-temporal queries
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS idx_ais_time_mmsi ON ais_telemetry(timestamp, mmsi);
        """)
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS idx_ais_spatial ON ais_telemetry(lat, lon);
        """)

    def ingest_records(self, records: List[Dict[str, Any]]):
        """Ingest AIS messages into DuckDB from list of dictionaries or AISMessage objects."""
        if not records:
            return
        df = pd.DataFrame([
            r.model_dump() if hasattr(r, "model_dump") else r
            for r in records
        ])
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        self.con.register("df_import", df)
        self.con.execute("""
            INSERT INTO ais_telemetry
            SELECT 
                timestamp, mmsi, lat, lon, sog, cog, heading, nav_status,
                imo, vessel_name, vessel_type
            FROM df_import
        """)
        self.con.unregister("df_import")

    @staticmethod
    def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in Nautical Miles between two coordinates."""
        r_earth_km = 6371.0088
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (math.sin(delta_phi / 2.0) ** 2 +
             math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        distance_km = r_earth_km * c
        return distance_km / 1.852

    def query_candidate_mmsis(self, corridor: SpatioTemporalCorridor) -> List[str]:
        """
        Identify MMSIs whose tracks intersect the spatio-temporal uncertainty envelope:
        (x_0, y_0, t_0 +/- delta_t, radius_nm).
        Uses spatial bounding box pre-filtering followed by exact spherical distance filtering.
        """
        t_start = corridor.t_origin - timedelta(hours=corridor.delta_t_hours)
        t_end = corridor.t_origin + timedelta(hours=corridor.delta_t_hours)

        # Coordinate bounding box in degrees
        lat_margin = corridor.radius_nm / 60.0
        # Account for latitude convergence
        cos_lat = max(0.1, math.cos(math.radians(corridor.center_lat)))
        lon_margin = corridor.radius_nm / (60.0 * cos_lat)

        min_lat = corridor.center_lat - lat_margin
        max_lat = corridor.center_lat + lat_margin
        min_lon = corridor.center_lon - lon_margin
        max_lon = corridor.center_lon + lon_margin

        # Query candidates within bounding box and time window
        query = """
            SELECT DISTINCT mmsi
            FROM ais_telemetry
            WHERE timestamp BETWEEN ? AND ?
              AND lat BETWEEN ? AND ?
              AND lon BETWEEN ? AND ?
        """
        params = [t_start, t_end, min_lat, max_lat, min_lon, max_lon]
        candidates_bbox = [row[0] for row in self.con.execute(query, params).fetchall()]

        # Precise geodetic distance verification
        confirmed_candidates = []
        for mmsi in candidates_bbox:
            # Query track points within the time window
            pts = self.con.execute("""
                SELECT lat, lon
                FROM ais_telemetry
                WHERE mmsi = ? AND timestamp BETWEEN ? AND ?
            """, [mmsi, t_start, t_end]).fetchall()

            for lat, lon in pts:
                dist = self.haversine_distance_nm(corridor.center_lat, corridor.center_lon, lat, lon)
                if dist <= corridor.radius_nm:
                    confirmed_candidates.append(mmsi)
                    break

        return confirmed_candidates

    def get_vessel_trajectory(
        self,
        mmsi: str,
        t_start: Optional[datetime] = None,
        t_end: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve sorted trajectory points for a given vessel across specified time range.
        Includes SOG, COG, heading, and spatial coordinates.
        """
        if t_start is not None and t_end is not None:
            query = """
                SELECT timestamp, mmsi, lat, lon, sog, cog, heading, nav_status,
                       imo, vessel_name, vessel_type
                FROM ais_telemetry
                WHERE mmsi = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            rows = self.con.execute(query, [mmsi, t_start, t_end]).fetchall()
        else:
            query = """
                SELECT timestamp, mmsi, lat, lon, sog, cog, heading, nav_status,
                       imo, vessel_name, vessel_type
                FROM ais_telemetry
                WHERE mmsi = ?
                ORDER BY timestamp ASC
            """
            rows = self.con.execute(query, [mmsi]).fetchall()

        columns = [
            "timestamp", "mmsi", "lat", "lon", "sog", "cog", "heading", "nav_status",
            "imo", "vessel_name", "vessel_type"
        ]
        return [dict(zip(columns, row)) for row in rows]

    def close(self):
        """Close database connection."""
        self.con.close()
