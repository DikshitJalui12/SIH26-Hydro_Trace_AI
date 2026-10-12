"""
Synthetic Sentinel-1 GRD C-band SAR Scene Generator.
Synthesizes dual-polarization (VV/VH) radar scenes with realistic sea clutter,
multiplicative speckle noise, calibrated geotransforms, and embedded oil slicks / biogenic lookalikes.
"""
from pathlib import Path
from typing import Tuple
import numpy as np
import rasterio
from rasterio.transform import from_origin

def generate_synthetic_sar_scene(
    output_tif: str,
    height: int = 512,
    width: int = 512,
    center_lat: float = 24.5,
    center_lon: float = 54.5,
    pixel_size_deg: float = 0.0001  # ~10 meters per pixel
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a dual-polarization GeoTIFF with:
    Band 1: VV polarization (DN values corresponding to ~ -12 dB ocean, -20 dB slick)
    Band 2: VH polarization (DN values corresponding to ~ -22 dB ocean, -28 dB slick)
    Ground truth mask: binary (1 inside genuine slick, 2 inside biogenic lookalike)
    """
    np.random.seed(42)

    # 1. Base clean ocean backscatter in dB
    # VV ocean mean: -12 dB, VH ocean mean: -22 dB
    ocean_vv_db = -12.0 + np.random.normal(0.0, 0.8, size=(height, width)).astype(np.float32)
    ocean_vh_db = -22.0 + np.random.normal(0.0, 0.8, size=(height, width)).astype(np.float32)

    gt_mask = np.zeros((height, width), dtype=np.uint8)

    # 2. Add genuine mineral oil slick (plume at center)
    # Mineral oil: strong backscatter drop (~ -8 dB drop in VV, -6 dB drop in VH)
    y_grid, x_grid = np.indices((height, width))
    
    # Mineral slick: curved elliptical plume
    cy, cx = height // 2, width // 2
    # Coordinate rotation for realistic slick drift
    theta = np.radians(35.0)
    dx = x_grid - cx
    dy = y_grid - cy
    rx = dx * np.cos(theta) - dy * np.sin(theta)
    ry = dx * np.sin(theta) + dy * np.cos(theta)
    
    # Elliptical core + tail
    slick_dist = (rx / 45.0)**2 + (ry / 15.0)**2
    slick_core = slick_dist <= 1.0
    
    # Mineral oil plume attenuation (sharp drop)
    attenuation_vv = np.zeros((height, width), dtype=np.float32)
    attenuation_vv[slick_core] = -8.5
    
    attenuation_vh = np.zeros((height, width), dtype=np.float32)
    attenuation_vh[slick_core] = -6.0
    
    gt_mask[slick_core] = 1

    # 3. Add lookalike biogenic surfactant film (mild diffuse damping in corner)
    by, bx = height // 4, width // 4
    bio_dist = ((x_grid - bx) / 35.0)**2 + ((y_grid - by) / 30.0)**2
    bio_mask = bio_dist <= 1.0
    # Biogenic film: mild damping (~ 3.2 dB) with diffuse borders
    bio_attenuation = np.exp(-bio_dist[bio_mask]) * -3.5
    attenuation_vv[bio_mask] = bio_attenuation
    attenuation_vh[bio_mask] = bio_attenuation * 0.7
    gt_mask[bio_mask] = 2

    # 4. Total decibel backscatter
    total_vv_db = ocean_vv_db + attenuation_vv
    total_vh_db = ocean_vh_db + attenuation_vh

    # 5. Convert to linear Sigma0
    sigma0_vv = 10.0 ** (total_vv_db / 10.0)
    sigma0_vh = 10.0 ** (total_vh_db / 10.0)

    # 6. Apply SAR speckle noise (Gamma multi-look speckle with ENL ~ 4)
    enl = 4.0
    speckle_vv = np.random.gamma(shape=enl, scale=1.0/enl, size=(height, width)).astype(np.float32)
    speckle_vh = np.random.gamma(shape=enl, scale=1.0/enl, size=(height, width)).astype(np.float32)
    
    speckled_vv = sigma0_vv * speckle_vv
    speckled_vh = sigma0_vh * speckle_vh

    # 7. Convert to Digital Numbers (DN) using typical S1 calibration factor (gain = 300)
    dn_vv = np.sqrt(speckled_vv) * 300.0
    dn_vh = np.sqrt(speckled_vh) * 300.0

    dn_vv = np.clip(dn_vv, 0, 65535).astype(np.uint16)
    dn_vh = np.clip(dn_vh, 0, 65535).astype(np.uint16)

    # 8. Write to georeferenced GeoTIFF
    out_path = Path(output_tif)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    top_lat = center_lat + (height / 2.0) * pixel_size_deg
    left_lon = center_lon - (width / 2.0) * pixel_size_deg
    transform = from_origin(left_lon, top_lat, pixel_size_deg, pixel_size_deg)

    with rasterio.open(
        out_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=2,
        dtype=np.uint16,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(dn_vv, 1)
        dst.write(dn_vh, 2)
        dst.set_band_description(1, "VV_DN")
        dst.set_band_description(2, "VH_DN")
        dst.update_tags(
            SENSOR="SENTINEL-1A",
            MODE="IW",
            PRODUCT_TYPE="GRD",
            ORBIT="42150",
            PASS_DIRECTION="DESCENDING",
            TIMESTAMP_UTC="2026-09-18T05:30:00Z"
        )

    return dn_vv, dn_vh, gt_mask

if __name__ == "__main__":
    generate_synthetic_sar_scene("data/sample_scene/s1_synthetic_grd.tif")
    print("Synthetic Sentinel-1 SAR scene generated successfully.")
