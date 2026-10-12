"""
Interactive geospatial visualizer for Lagrangian drift results using Leaflet.js.
"""

import json
from typing import Dict, Any
from .models import DriftOutputPayload


def generate_interactive_map_html(
    payload: DriftOutputPayload,
    slick_geojson: Dict[str, Any]
) -> str:
    """
    Generate a standalone interactive HTML/Leaflet map visualization.
    """
    slick_geom = slick_geojson.get("geometry", slick_geojson)
    payload_dict = payload.model_dump()
    payload_json = json.dumps(payload_dict)
    slick_json = json.dumps(slick_geom)
    
    centroid_lat = payload.origin_uncertainty_cone.centroid.latitude
    centroid_lon = payload.origin_uncertainty_cone.centroid.longitude
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reverse Lagrangian Drift Engine - {payload.slick_id}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        #map {{ width: 100%; height: 100vh; background: #0a1118; }}
        .hud-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: rgba(13, 20, 30, 0.88);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(0, 220, 255, 0.3);
            border-radius: 8px;
            padding: 16px 20px;
            color: #e0f2fe;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            max-width: 360px;
        }}
        .hud-title {{
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #38bdf8;
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(56, 189, 248, 0.3);
            padding-bottom: 6px;
        }}
        .hud-metric {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            margin: 6px 0;
        }}
        .hud-label {{ color: #94a3b8; }}
        .hud-val {{ font-weight: 600; color: #f8fafc; font-family: monospace; }}
        .legend-box {{
            position: absolute;
            bottom: 25px;
            left: 20px;
            z-index: 1000;
            background: rgba(13, 20, 30, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            padding: 10px 14px;
            color: #e2e8f0;
            font-size: 12px;
        }}
        .legend-item {{ display: flex; align-items: center; margin: 4px 0; gap: 8px; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 3px; display: inline-block; }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="hud-panel">
        <div class="hud-title">Lagrangian Drift Vector Engine</div>
        <div class="hud-metric"><span class="hud-label">Slick ID:</span><span class="hud-val">{payload.slick_id}</span></div>
        <div class="hud-metric"><span class="hud-label">Mode:</span><span class="hud-val">{payload.mode.value}</span></div>
        <div class="hud-metric"><span class="hud-label">Detection (T):</span><span class="hud-val">{payload.hindcast_window.end_utc[:19]}Z</span></div>
        <div class="hud-metric"><span class="hud-label">Origin (t0):</span><span class="hud-val">{payload.hindcast_window.start_utc[:19]}Z</span></div>
        <div class="hud-metric"><span class="hud-label">Particles:</span><span class="hud-val">10,000 Stochastic</span></div>
        <div class="hud-metric"><span class="hud-label">Diffusivity (Kh):</span><span class="hud-val">10.0 m²/s</span></div>
        <div class="hud-metric"><span class="hud-label">Wind Leeway (α):</span><span class="hud-val">3% (deflected)</span></div>
        <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 10px 0;">
        <div class="hud-metric"><span class="hud-label">95% Semi-Major:</span><span class="hud-val">{payload.origin_uncertainty_cone.semi_major_axis_km} km</span></div>
        <div class="hud-metric"><span class="hud-label">95% Semi-Minor:</span><span class="hud-val">{payload.origin_uncertainty_cone.semi_minor_axis_km} km</span></div>
        <div class="hud-metric"><span class="hud-label">Orientation:</span><span class="hud-val">{payload.origin_uncertainty_cone.orientation_degrees}° N</span></div>
        <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 10px 0;">
        <div class="hud-metric"><span class="hud-label">Avg Current:</span><span class="hud-val">{payload.weather_summary.avg_current_knots} kts</span></div>
        <div class="hud-metric"><span class="hud-label">Avg 10m Wind:</span><span class="hud-val">{payload.weather_summary.avg_wind_knots} kts</span></div>
        <div class="hud-metric"><span class="hud-label">Sea Surface Temp:</span><span class="hud-val">{payload.weather_summary.sea_surface_temp} °C</span></div>
    </div>

    <div class="legend-box">
        <div style="font-weight: 700; margin-bottom: 6px; color: #38bdf8;">MAP OVERLAYS</div>
        <div class="legend-item"><span class="legend-color" style="background: #ef4444; border: 1px solid #ffffff;"></span> Satellite Detection Slick (t=T)</div>
        <div class="legend-item"><span class="legend-color" style="background: #eab308; border: 1px solid #ffffff;"></span> 95% Confidence Origin Cone (t=t0)</div>
        <div class="legend-item"><span class="legend-color" style="background: #38bdf8; height: 3px;"></span> Ensemble Mean Drift Track</div>
        <div class="legend-item"><span class="legend-color" style="background: rgba(56, 189, 248, 0.25); height: 1px;"></span> RK4 Stochastic Particle Corridors</div>
    </div>

    <script>
        const payload = {payload_json};
        const slickGeom = {slick_json};

        const map = L.map('map').setView([{centroid_lat}, {centroid_lon}], 11);

        // Dark nautical ocean base map
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; CartoDB &copy; OpenStreetMap contributors',
            maxZoom: 19
        }}).addTo(map);

        // 1. Plot Stochastic Streamlines
        L.geoJSON(payload.drift_path_linestring, {{
            filter: function(feature) {{
                return feature.properties.type === 'stochastic_particle_streamline';
            }},
            style: {{
                color: '#38bdf8',
                weight: 1,
                opacity: 0.2
            }}
        }}).addTo(map);

        // 2. Plot Ensemble Mean Track
        const ensembleLayer = L.geoJSON(payload.drift_path_linestring, {{
            filter: function(feature) {{
                return feature.properties.type === 'ensemble_mean_path';
            }},
            style: {{
                color: '#00e5ff',
                weight: 3.5,
                opacity: 0.95,
                dashArray: '6, 4'
            }}
        }}).addTo(map);

        // 3. Plot 95% Confidence Origin Ellipse
        const ellipseLayer = L.geoJSON(payload.origin_uncertainty_cone.origin_polygon, {{
            style: {{
                color: '#eab308',
                weight: 2.5,
                fillColor: '#facc15',
                fillOpacity: 0.35
            }}
        }}).addTo(map);

        ellipseLayer.bindPopup(`
            <b>95% Spatial Confidence Origin Zone</b><br>
            Centroid: ${{payload.origin_uncertainty_cone.centroid.latitude.toFixed(4)}}°N, ${{payload.origin_uncertainty_cone.centroid.longitude.toFixed(4)}}°E<br>
            Semi-Major: ${{payload.origin_uncertainty_cone.semi_major_axis_km}} km<br>
            Semi-Minor: ${{payload.origin_uncertainty_cone.semi_minor_axis_km}} km<br>
            Orientation: ${{payload.origin_uncertainty_cone.orientation_degrees}}°
        `);

        // Centroid marker
        L.circleMarker([payload.origin_uncertainty_cone.centroid.latitude, payload.origin_uncertainty_cone.centroid.longitude], {{
            radius: 5,
            color: '#ffffff',
            fillColor: '#eab308',
            fillOpacity: 1.0,
            weight: 2
        }}).addTo(map).bindPopup("Estimated Discharge Origin Centroid (t0)");

        // 4. Plot Input Satellite Slick
        const slickLayer = L.geoJSON(slickGeom, {{
            style: {{
                color: '#ef4444',
                weight: 2.5,
                fillColor: '#dc2626',
                fillOpacity: 0.55
            }}
        }}).addTo(map).bindPopup(`<b>Satellite Slick Detection (t=T)</b><br>ID: ${{payload.slick_id}}`);

        // Fit map bounds to encompass both slick and origin ellipse
        const group = L.featureGroup([slickLayer, ellipseLayer, ensembleLayer]);
        map.fitBounds(group.getBounds().pad(0.15));
    </script>
</body>
</html>
"""
    return html_content
