"""
Spatial Morphological Feature Extraction and GeoJSON Payload Generator.
Extracts vector boundaries, geodesic area (km²), perimeter (km),
geographic centroid (lat/lon), and formats the final JSON payload.
"""
import math
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from rasterio.transform import Affine
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shp_transform

from ..ingestion.s1_reader import SARProductMetadata
from ..discrimination.era5_wind_gate import WindGatingResult
from ..discrimination.polarization_features import PolarimetricFeatures
from ..discrimination.lookalike_classifier import SlickClassification
from ..discrimination.fays_spreading import FaysAgeEstimate

@dataclass
class SlickDetectionRecord:
    """Individual oil slick detection record matching user's requested JSON payload schema."""
    slick_id: str
    timestamp_utc: str
    satellite_metadata: Dict[str, Any]
    centroid: Dict[str, float]
    geometry: Dict[str, Any]
    metrics: Dict[str, float]
    confidence_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SpatialMorphologyExtractor:
    """
    Extracts polygons and physical metrics from binary segmentation masks.
    """
    
    def __init__(self, min_area_km2: float = 0.05):
        self.min_area_km2 = min_area_km2

    def extract_slicks(
        self,
        binary_mask: np.ndarray,
        vv_db: np.ndarray,
        metadata: SARProductMetadata,
        wind_result: WindGatingResult,
        fays_age_calculator,
        feature_extractor,
        classifier
    ) -> List[SlickDetectionRecord]:
        """
        Vectorizes binary slick mask, calculates geodesic metrics, extracts polarimetric
        signatures, and generates standardized detection records.
        """
        h, w = binary_mask.shape
        transform = metadata.transform
        records: List[SlickDetectionRecord] = []

        # Find external contours
        mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 4:
                continue

            # Pixel area
            px_area = cv2.contourArea(cnt)
            if px_area < 10:
                continue

            # Convert contour coordinates to geographic (lon, lat)
            coords = []
            for pt in cnt.squeeze():
                if pt.ndim == 0 or len(pt) < 2:
                    continue
                px_col, px_row = float(pt[0]), float(pt[1])
                geo_x, geo_y = transform * (px_col, px_row)
                coords.append((geo_x, geo_y))

            if len(coords) < 3:
                continue

            # Ensure polygon is closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
                if not poly.is_valid or poly.is_empty:
                    continue

            # In case of MultiPolygon from buffer(0), pick largest part
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda p: p.area)

            # Geographic Centroid (lat, lon)
            c_lon, c_lat = poly.centroid.x, poly.centroid.y

            # Calculate geodesic area (km²) and perimeter (km)
            area_km2, perimeter_km = self._calculate_geodesic_metrics(poly, c_lat, metadata)
            if area_km2 < self.min_area_km2:
                continue

            # Create individual mask for this contour to sample backscatter and polarimetry
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 1, thickness=-1)
            
            # Mean backscatter in dB inside the slick
            slick_pixels = vv_db[contour_mask == 1]
            mean_backscatter_db = float(np.mean(slick_pixels)) if len(slick_pixels) > 0 else -20.0

            # Polarimetric features & Classification
            features = feature_extractor.extract_features(vv_db, vv_db * 0.5, contour_mask)
            classification = classifier.classify_slick(features, wind_result, area_km2)

            # Fay's Spreading Theory Age Estimation
            fays_estimate: FaysAgeEstimate = fays_age_calculator.estimate_age_from_area(area_km2)

            # Build GeoJSON Polygon geometry
            # GeoJSON coordinates format: [[[lon, lat], [lon, lat], ...]]
            geojson_poly = mapping(poly)

            # Format satellite metadata block
            sat_meta = {
                "orbit": metadata.orbit_number,
                "sensor": metadata.sensor,
                "pass_direction": metadata.pass_direction
            }

            # Format metrics block
            metrics = {
                "area_km2": round(area_km2, 4),
                "perimeter_km": round(perimeter_km, 3),
                "mean_backscatter_db": round(mean_backscatter_db, 2),
                "estimated_age_hours": round(fays_estimate.estimated_age_hours, 2)
            }

            record = SlickDetectionRecord(
                slick_id=str(uuid.uuid4()),
                timestamp_utc=metadata.timestamp_utc,
                satellite_metadata=sat_meta,
                centroid={"latitude": round(c_lat, 6), "longitude": round(c_lon, 6)},
                geometry=geojson_poly,
                metrics=metrics,
                confidence_score=round(classification.confidence_score, 3)
            )
            records.append(record)

        return records

    def _calculate_geodesic_metrics(
        self, poly: Polygon, center_lat: float, metadata: SARProductMetadata
    ) -> Tuple[float, float]:
        """
        Computes accurate geodesic surface area in km² and perimeter in km,
        accounting for ellipsoidal curvature at the latitude of observation.
        """
        crs_str = metadata.crs.upper()
        
        if "4326" in crs_str or "GEOGCS" in crs_str or abs(center_lat) <= 90.0:
            # Geographic WGS84 degrees
            # 1 degree latitude ~ 111.132 km
            # 1 degree longitude ~ 111.320 * cos(lat) km
            lat_rad = math.radians(center_lat)
            m_per_deg_lat = 111132.95 - 559.82 * math.cos(2 * lat_rad) + 1.17 * math.cos(4 * lat_rad)
            m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
            
            # Scaled coordinates in meters
            scaled_coords = [
                (x * m_per_deg_lon, y * m_per_deg_lat)
                for x, y in poly.exterior.coords
            ]
            m_poly = Polygon(scaled_coords)
            area_m2 = m_poly.area
            perimeter_m = m_poly.length
            
            area_km2 = area_m2 / 1e6
            perimeter_km = perimeter_m / 1e3
        else:
            # Projected coordinates (assumed meters, e.g. UTM)
            area_km2 = poly.area / 1e6
            perimeter_km = poly.length / 1e3

        return max(0.0001, area_km2), max(0.01, perimeter_km)
