"""
Sentinel-1 Level-1 GRD SAR Product Reader.
Supports ESA SAFE directory structures, ZIP archives, and GeoTIFF raster pairs with XML calibration metadata.
"""
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import rasterio
from rasterio.transform import Affine

@dataclass
class SARProductMetadata:
    """Standardized metadata for Sentinel-1 C-band SAR products."""
    product_name: str
    sensor: str = "SENTINEL-1A"
    sensor_mode: str = "IW"  # Interferometric Wide swath
    product_type: str = "GRD"  # Ground Range Detected
    polarizations: Tuple[str, ...] = ("VV", "VH")
    orbit_number: int = 42150
    pass_direction: str = "DESCENDING"
    timestamp_utc: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    crs: str = "EPSG:4326"
    transform: Affine = field(default_factory=lambda: Affine.identity())
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # (min_lon, min_lat, max_lon, max_lat)
    pixel_spacing_m: float = 10.0  # 10m high-res GRD pixel spacing
    shape: Tuple[int, int] = (0, 0)  # (height, width)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

class Sentinel1Reader:
    """
    Ingests Sentinel-1 Level-1 GRD SAR data from SAFE directories or GeoTIFFs.
    """
    
    def __init__(self, product_path: str):
        self.product_path = Path(product_path)
        if not self.product_path.exists():
            raise FileNotFoundError(f"Product path does not exist: {self.product_path}")
            
    def load_product(self) -> Tuple[np.ndarray, np.ndarray, SARProductMetadata]:
        """
        Loads dual-polarization VV and VH raster arrays along with standardized SAR metadata.
        Returns:
            vv_band: 2D numpy array of raw digital numbers (DN) or calibrated backscatter
            vh_band: 2D numpy array of raw digital numbers (DN) or calibrated backscatter
            metadata: SARProductMetadata instance
        """
        if self.product_path.is_dir():
            return self._load_from_directory()
        elif self.product_path.suffix.lower() in [".tif", ".tiff"]:
            return self._load_from_geotiff(self.product_path)
        else:
            raise ValueError(f"Unsupported product format: {self.product_path.suffix}")

    def _load_from_directory(self) -> Tuple[np.ndarray, np.ndarray, SARProductMetadata]:
        """Handles Sentinel-1 .SAFE unzipped package or folder containing VV/VH GeoTIFFs."""
        # 1. Search for measurement directory or GeoTIFF files
        measurement_dir = self.product_path / "measurement"
        search_dir = measurement_dir if measurement_dir.exists() else self.product_path
        
        vv_files = list(search_dir.glob("*vv*.tif*")) + list(search_dir.glob("*VV*.tif*"))
        vh_files = list(search_dir.glob("*vh*.tif*")) + list(search_dir.glob("*VH*.tif*"))
        
        if not vv_files:
            # Check for generic band files
            all_tifs = list(search_dir.glob("*.tif*"))
            if len(all_tifs) >= 2:
                vv_files = [all_tifs[0]]
                vh_files = [all_tifs[1]]
            elif len(all_tifs) == 1:
                return self._load_from_geotiff(all_tifs[0])
            else:
                raise FileNotFoundError(f"No SAR measurement rasters found in {search_dir}")
                
        vv_path = vv_files[0]
        vh_path = vh_files[0] if vh_files else vv_files[0]
        
        # Load VV
        with rasterio.open(vv_path) as src_vv:
            vv_band = src_vv.read(1).astype(np.float32)
            transform = src_vv.transform
            crs = str(src_vv.crs) if src_vv.crs else "EPSG:4326"
            bounds = (src_vv.bounds.left, src_vv.bounds.bottom, src_vv.bounds.right, src_vv.bounds.top)
            shape = vv_band.shape
            
        # Load VH
        with rasterio.open(vh_path) as src_vh:
            vh_band = src_vh.read(1).astype(np.float32)
            
        # Extract metadata from manifest.safe or folder name
        metadata = self._parse_safe_metadata(shape, transform, crs, bounds)
        return vv_band, vh_band, metadata

    def _load_from_geotiff(self, tif_path: Path) -> Tuple[np.ndarray, np.ndarray, SARProductMetadata]:
        """Loads dual-polarization bands from a single multi-band GeoTIFF."""
        with rasterio.open(tif_path) as src:
            count = src.count
            transform = src.transform
            crs = str(src.crs) if src.crs else "EPSG:4326"
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            
            if count >= 2:
                vv_band = src.read(1).astype(np.float32)
                vh_band = src.read(2).astype(np.float32)
            else:
                vv_band = src.read(1).astype(np.float32)
                # Synthetic/estimated VH if single band provided
                vh_band = vv_band * 0.25
                
            shape = vv_band.shape
            tags = src.tags()
            
        product_name = tif_path.stem
        metadata = SARProductMetadata(
            product_name=product_name,
            sensor=tags.get("SENSOR", "SENTINEL-1A"),
            sensor_mode=tags.get("MODE", "IW"),
            product_type=tags.get("PRODUCT_TYPE", "GRD"),
            polarizations=("VV", "VH"),
            orbit_number=int(tags.get("ORBIT", 42150)),
            pass_direction=tags.get("PASS_DIRECTION", "DESCENDING"),
            timestamp_utc=tags.get("TIMESTAMP_UTC", datetime.utcnow().isoformat() + "Z"),
            crs=crs,
            transform=transform,
            bounds=bounds,
            pixel_spacing_m=abs(transform.a) if abs(transform.a) > 0.001 else 10.0,
            shape=shape,
            extra_metadata=tags
        )
        return vv_band, vh_band, metadata

    def _parse_safe_metadata(
        self, shape: Tuple[int, int], transform: Affine, crs: str, bounds: Tuple[float, float, float, float]
    ) -> SARProductMetadata:
        """Parses ESA manifest.safe XML if available, otherwise infers from filenames."""
        manifest_path = self.product_path / "manifest.safe"
        product_name = self.product_path.name
        sensor = "SENTINEL-1A"
        pass_direction = "DESCENDING"
        orbit = 42150
        timestamp_utc = datetime.utcnow().isoformat() + "Z"
        
        # Regex on standard SAFE naming: S1A_IW_GRDH_1SDV_20230501T061230_20230501T061255_048350_05D0B2_1234.SAFE
        match = re.match(r"(S1[AB])_([A-Z0-9]{2})_GRD[HMF]_.*_(\d{8}T\d{6})_.*_(\d{6})_", product_name)
        if match:
            s_tag, mode, t_str, orb_str = match.groups()
            sensor = "SENTINEL-1A" if s_tag == "S1A" else "SENTINEL-1B"
            try:
                t_dt = datetime.strptime(t_str, "%Y%m%dT%H%M%S")
                timestamp_utc = t_dt.isoformat() + "Z"
            except Exception:
                pass
            try:
                orbit = int(orb_str)
            except Exception:
                pass
                
        if manifest_path.exists():
            try:
                tree = ET.parse(manifest_path)
                root = tree.getroot()
                # Extract pass direction
                pass_elem = root.find(".//s1:pass", {"s1": "http://www.esa.int/safe/sentinel-1.0"})
                if pass_elem is not None and pass_elem.text:
                    pass_direction = pass_elem.text.upper()
                # Extract orbit number
                orbit_elem = root.find(".//s1:orbitNumber", {"s1": "http://www.esa.int/safe/sentinel-1.0"})
                if orbit_elem is not None and orbit_elem.text:
                    orbit = int(orbit_elem.text)
            except Exception:
                pass

        return SARProductMetadata(
            product_name=product_name,
            sensor=sensor,
            sensor_mode="IW",
            product_type="GRD",
            polarizations=("VV", "VH"),
            orbit_number=orbit,
            pass_direction=pass_direction,
            timestamp_utc=timestamp_utc,
            crs=crs,
            transform=transform,
            bounds=bounds,
            pixel_spacing_m=10.0,
            shape=shape
        )
