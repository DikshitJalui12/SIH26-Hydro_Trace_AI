"""
End-to-end integration tests for ReverseLagrangianDriftEngine.
"""

import json
from pathlib import Path
import pytest
from engine.models import DriftMode, DriftOutputPayload
from engine.drift_engine import ReverseLagrangianDriftEngine
from engine.visualizer import generate_interactive_map_html


@pytest.fixture
def sample_slick_feature():
    sample_path = Path(__file__).parent.parent / "sample_data" / "sample_slick.geojson"
    with open(sample_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_reverse_hindcast_e2e(sample_slick_feature):
    engine = ReverseLagrangianDriftEngine(
        num_particles=2000, # fast test
        step_minutes=15,
        alpha_leeway=0.03,
        eddy_diffusivity_kh=10.0
    )
    
    payload = engine.run(
        slick_geojson=sample_slick_feature,
        detection_time_iso="2026-09-18T06:30:00Z",
        duration_hours=6.0,
        mode=DriftMode.REVERSE_HINDCAST
    )
    
    # 1. Output model validation
    assert isinstance(payload, DriftOutputPayload)
    assert payload.slick_id == "SLICK-SAR-20260918-0042"
    assert payload.mode == DriftMode.REVERSE_HINDCAST
    
    # 2. Check hindcast window
    assert payload.hindcast_window.step_minutes == 15
    assert "2026-09-18T06:30:00" in payload.hindcast_window.end_utc
    assert "2026-09-18T00:30:00" in payload.hindcast_window.start_utc
    
    # 3. Check origin uncertainty cone
    cone = payload.origin_uncertainty_cone
    assert cone.centroid.latitude > 0.0
    assert cone.centroid.longitude > 0.0
    assert cone.semi_major_axis_km > 0.0
    assert cone.semi_minor_axis_km > 0.0
    assert 0.0 <= cone.orientation_degrees < 360.0
    assert cone.origin_polygon["type"] == "Polygon"
    assert len(cone.origin_polygon["coordinates"][0]) >= 4
    
    # 4. Check drift path FeatureCollection
    fc = payload.drift_path_linestring
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) > 0
    # Mean track present
    mean_tracks = [f for f in fc["features"] if f["properties"].get("type") == "ensemble_mean_path"]
    assert len(mean_tracks) == 1
    assert mean_tracks[0]["geometry"]["type"] == "LineString"
    
    # 5. Check weather summary
    weather = payload.weather_summary
    assert weather.avg_current_knots > 0.0
    assert weather.avg_wind_knots > 0.0
    assert 10.0 < weather.sea_surface_temp < 35.0
    
    # 6. Test HTML visualization generation
    html = generate_interactive_map_html(payload, sample_slick_feature)
    assert "<!DOCTYPE html>" in html
    assert "SLICK-SAR-20260918-0042" in html
    assert "leaflet.js" in html


def test_forward_forecast_48h_e2e(sample_slick_feature):
    engine = ReverseLagrangianDriftEngine(
        num_particles=1000,
        step_minutes=30,
        alpha_leeway=0.03,
        eddy_diffusivity_kh=10.0
    )
    
    payload = engine.run(
        slick_geojson=sample_slick_feature,
        detection_time_iso="2026-09-18T06:30:00Z",
        duration_hours=48.0,
        mode=DriftMode.FORWARD_FORECAST
    )
    
    assert payload.mode == DriftMode.FORWARD_FORECAST
    assert "2026-09-18T06:30:00" in payload.hindcast_window.start_utc
    assert "2026-09-20T06:30:00" in payload.hindcast_window.end_utc
    assert payload.origin_uncertainty_cone.semi_major_axis_km > 0.0
