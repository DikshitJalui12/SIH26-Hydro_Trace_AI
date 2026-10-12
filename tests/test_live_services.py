"""
Tests for Live Metocean Service, Live AIS Fleet Radar, and What-If Simulation routes.
"""
import pytest
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)


def test_live_weather_endpoint():
    """Verify live marine weather conditioning data endpoint."""
    response = client.get("/api/v1/weather/live?lat=1.2897&lon=103.8501")
    assert response.status_code == 200
    data = response.json()
    assert "sea_state" in data
    assert "wave_height_m" in data["sea_state"]
    assert "surface_current" in data
    assert "surface_wind" in data
    assert "spill_dispersion_index" in data
    assert data["status"] == "LIVE_SYNCHRONIZED"


def test_satellite_catalog_endpoint():
    """Verify satellite layer catalog endpoint."""
    response = client.get("/api/v1/weather/satellite/catalog")
    assert response.status_code == 200
    catalog = response.json()
    assert len(catalog) >= 4
    categories = [cat["id"] for cat in catalog]
    assert "sentinel1_sar_vv" in categories
    assert "sentinel1_sar_vh" in categories
    assert "sentinel2_optical_rgb" in categories


def test_live_ais_fleet_endpoint():
    """Verify real-time dynamic AIS fleet radar endpoint."""
    response = client.get("/api/v1/ais/live-fleet")
    assert response.status_code == 200
    fleet = response.json()
    assert len(fleet) >= 5
    mmsis = [v["mmsi"] for v in fleet]
    assert "352001928" in mmsis  # PACIFIC VOYAGER

    # Filter suspect
    resp_suspect = client.get("/api/v1/ais/live-fleet?min_risk=0.7")
    assert resp_suspect.status_code == 200
    suspects = resp_suspect.json()
    for s in suspects:
        assert s["risk_score"] >= 0.7


def test_live_ais_single_vessel_telemetry():
    """Verify telemetry and anomaly inspection for specific vessel."""
    response = client.get("/api/v1/ais/vessel/352001928")
    assert response.status_code == 200
    v = response.json()
    assert v["name"] == "PACIFIC VOYAGER"
    assert len(v["anomalies"]) > 0


def test_whatif_simulation_endpoint():
    """Verify interactive what-if hydrodynamic modeling sensitivity endpoint."""
    payload = {
        "slick_id": "SLICK-2026-MALACCA-0881",
        "eddy_diffusivity_kh": 25.0,
        "wind_leeway_coefficient": 0.04,
        "duration_hours": 12.0,
        "particles_count": 10000
    }
    response = client.post("/api/v1/simulation/whatif", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "estimated_origin" in data
    assert "uncertainty_ellipse" in data
    assert data["uncertainty_ellipse"]["area_km2"] > 28.4  # expanded due to higher Kh
    assert "attribution_impact" in data
