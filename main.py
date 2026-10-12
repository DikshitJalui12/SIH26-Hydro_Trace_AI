"""
AIS Attribution Engine - Main CLI & Operational Runner.
Ingests spatio-temporal corridor input from Agent 2, executes multi-factor
attribution analysis, and outputs structured forensic JSON.
"""

import json
import argparse
import sys
from datetime import datetime
from engine.models import SpatioTemporalCorridor
from engine.telemetry_store import TelemetryStore
from engine.pipeline import AISAttributionPipeline
from data.synthetic_ais_generator import generate_scenario_telemetry


def parse_arguments():
    parser = argparse.ArgumentParser(description="AIS Attribution Engine for MARPOL Annex I Forensic Investigation")
    parser.add_argument(
        "--corridor-file",
        type=str,
        default="data/sample_corridor.json",
        help="Path to JSON file containing Agent 2 spatio-temporal corridor parameters"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to save structured output JSON"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional JSON string of custom scoring weights"
    )
    return parser.parse_args()


def run_attribution(corridor_data: dict, weights: dict = None) -> dict:
    """Run attribution pipeline and return structured JSON dictionary."""
    # Parse corridor
    corridor = SpatioTemporalCorridor(
        slick_id=corridor_data["slick_id"],
        center_lat=corridor_data["center_lat"],
        center_lon=corridor_data["center_lon"],
        radius_nm=corridor_data["radius_nm"],
        t_origin=datetime.fromisoformat(corridor_data["t_origin"]),
        delta_t_hours=corridor_data["delta_t_hours"],
        slick_detection_time=datetime.fromisoformat(corridor_data["slick_detection_time"]) if corridor_data.get("slick_detection_time") else None,
        slick_area_sqkm=corridor_data.get("slick_area_sqkm"),
        estimated_volume_m3=corridor_data.get("estimated_volume_m3")
    )

    # Initialize Telemetry Store in DuckDB
    store = TelemetryStore()

    # Ingest historical telemetry
    print(f"[*] Ingesting historical AIS telemetry into DuckDB...", file=sys.stderr)
    records = generate_scenario_telemetry()
    store.ingest_records(records)
    print(f"[+] Successfully loaded {len(records)} AIS telemetry points.", file=sys.stderr)

    # Initialize Pipeline
    pipeline = AISAttributionPipeline(telemetry_store=store)

    # Execute attribution analysis
    print(f"[*] Querying spatio-temporal corridor for slick: {corridor.slick_id}...", file=sys.stderr)
    report = pipeline.process_corridor(corridor=corridor, weights=weights)
    print(f"[+] Analysis complete. Identified {report.candidates_analyzed} candidate vessels intersecting corridor.", file=sys.stderr)

    return report.model_dump()


def main():
    args = parse_arguments()

    with open(args.corridor_file, "r") as f:
        corridor_dict = json.load(f)

    weights_dict = None
    if args.weights:
        weights_dict = json.loads(args.weights)

    report_dict = run_attribution(corridor_dict, weights_dict)

    json_output = json.dumps(report_dict, indent=2, default=str)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(json_output)
        print(f"[+] Output written to {args.output_file}", file=sys.stderr)

    # Print JSON payload to stdout
    print(json_output)


if __name__ == "__main__":
    main()
