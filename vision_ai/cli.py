"""
Command Line Interface for Vision AI Tier Autonomous Oil Spill Detection.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from .config import PipelineConfig, PreprocessingConfig, SegmentationConfig
from .pipeline import VisionAITier

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vision AI Tier: Autonomous SAR Oil Spill Detection Engine"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to Sentinel-1 Level-1 GRD product (SAFE directory or GeoTIFF)"
    )
    parser.add_argument(
        "--output", "-o",
        default="oil_spill_detections.json",
        help="Destination path for output JSON payload"
    )
    parser.add_argument(
        "--weights", "-w",
        default="vision_ai/models/attention_unet_s1.pth",
        help="Path to Attention U-Net model weights file (.pth)"
    )
    parser.add_argument(
        "--filter",
        choices=["refined_lee", "gamma_map"],
        default="refined_lee",
        help="Speckle filter algorithm (default: refined_lee)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Segmentation probability threshold (0.0 to 1.0)"
    )
    parser.add_argument(
        "--wind-u10",
        type=float,
        default=None,
        help="Optional override for ECMWF ERA5 U10 wind component (m/s)"
    )
    parser.add_argument(
        "--wind-v10",
        type=float,
        default=None,
        help="Optional override for ECMWF ERA5 V10 wind component (m/s)"
    )
    parser.add_argument(
        "--allow-wind-fail",
        action="store_true",
        help="Do not discard detections if wind gating fails (useful for debugging)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging"
    )
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    config = PipelineConfig()
    config.preprocessing.speckle_filter_type = args.filter
    config.segmentation.probability_threshold = args.threshold
    config.segmentation.base_filters = 16

    weights_path = args.weights if Path(args.weights).exists() else None
    tier = VisionAITier(config=config, weights_path=weights_path)

    try:
        detections = tier.process_scene(
            product_path=args.input,
            simulated_u10=args.wind_u10,
            simulated_v10=args.wind_v10,
            discard_on_wind_fail=not args.allow_wind_fail
        )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2)

        print(f"\n[Vision AI Tier] Processing successful. {len(detections)} slick(s) detected.")
        print(f"Results written to: {out_path.resolve()}")
        if detections:
            print("\nSample Detection Payload:")
            print(json.dumps(detections[0], indent=2))

    except Exception as e:
        logging.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
