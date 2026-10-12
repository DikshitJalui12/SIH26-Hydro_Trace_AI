"""
Command-line interface for the Reverse Lagrangian Drift Engine.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .models import DriftMode
from .drift_engine import ReverseLagrangianDriftEngine
from .visualizer import generate_interactive_map_html


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reverse Lagrangian Drift Engine for Oil Slick Backtracking & Forecast"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input GeoJSON file containing detected slick polygon"
    )
    parser.add_argument(
        "--detection-time", "-t",
        type=str,
        default=None,
        help="ISO 8601 detection timestamp (e.g. 2026-09-18T06:00:00Z). Inferred if not given."
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=12.0,
        help="Drift simulation duration in hours (default: 12.0)"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["reverse", "forward"],
        default="reverse",
        help="Drift mode: reverse (hindcast origin) or forward (containment forecast)"
    )
    parser.add_argument(
        "--step-minutes",
        type=int,
        default=15,
        help="RK4 integration time-step in minutes (default: 15)"
    )
    parser.add_argument(
        "--particles", "-n",
        type=int,
        default=10000,
        help="Number of Monte Carlo stochastic particles (default: 10000)"
    )
    parser.add_argument(
        "--kh",
        type=float,
        default=10.0,
        help="Horizontal eddy diffusivity Kh in m²/s (default: 10.0)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.03,
        help="Aerodynamic leeway wind drift coefficient (default: 0.03 = 3%)"
    )
    parser.add_argument(
        "--slick-id",
        type=str,
        default=None,
        help="Slick ID matching Agent 1 detection"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="File path to save the structured output JSON payload"
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="File path to save interactive Leaflet HTML map"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input GeoJSON file '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    with open(input_path, "r", encoding="utf-8") as f:
        slick_geojson = json.load(f)
        
    # Determine detection time
    detection_time = args.detection_time
    if not detection_time:
        props = slick_geojson.get("properties", {})
        detection_time = props.get("detection_time", props.get("timestamp"))
        if not detection_time:
            detection_time = datetime.now(timezone.utc).isoformat()
            
    drift_mode = DriftMode.REVERSE_HINDCAST if args.mode == "reverse" else DriftMode.FORWARD_FORECAST
    
    engine = ReverseLagrangianDriftEngine(
        alpha_leeway=args.alpha,
        step_minutes=args.step_minutes,
        num_particles=args.particles,
        eddy_diffusivity_kh=args.kh
    )
    
    payload = engine.run(
        slick_geojson=slick_geojson,
        detection_time_iso=detection_time,
        duration_hours=args.hours,
        mode=drift_mode,
        slick_id=args.slick_id
    )
    
    payload_json = payload.model_dump_json(indent=2)
    
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload_json)
        print(f"[OK] Structured JSON payload written to: {out_path.resolve()}")
    else:
        print(payload_json)
        
    if args.html:
        html_content = generate_interactive_map_html(payload, slick_geojson)
        html_path = Path(args.html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[OK] Interactive Leaflet map saved to: {html_path.resolve()}")


if __name__ == "__main__":
    main()
