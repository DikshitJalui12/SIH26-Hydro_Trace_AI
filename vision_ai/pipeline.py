"""
Master Vision AI Tier Pipeline for Autonomous Oil Spill Detection.
Integrates SAR ingestion, calibration, speckle filtering, Attention U-Net segmentation,
ERA5 wind gating, lookalike discrimination, and Fay's hydrodynamic age estimation.
"""
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .config import PipelineConfig
from .ingestion.s1_reader import Sentinel1Reader, SARProductMetadata
from .ingestion.preprocessing import RadiometricCalibrator
from .ingestion.speckle_filter import RefinedLeeFilter, GammaMapFilter
from .ingestion.normalization import SARNormalizer
from .models.model_factory import OilSpillModelInference
from .discrimination.era5_wind_gate import ERA5WindGater, WindGatingResult
from .discrimination.polarization_features import PolarimetricFeatureExtractor
from .discrimination.lookalike_classifier import LookalikeClassifier
from .discrimination.fays_spreading import FaysSpreadingEstimator
from .morphology.spatial_extractor import SpatialMorphologyExtractor, SlickDetectionRecord

logger = logging.getLogger("VisionAITier")

class VisionAITier:
    """
    Operational Vision AI Tier for Sentinel-1 C-band SAR oil spill detection.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        weights_path: Optional[str] = None
    ):
        self.config = config or PipelineConfig()
        
        # Subsystem initialization
        self.calibrator = RadiometricCalibrator()
        
        # Speckle filter selection
        if self.config.preprocessing.speckle_filter_type.lower() == "gamma_map":
            self.speckle_filter = GammaMapFilter(
                window_size=self.config.preprocessing.filter_window_size,
                enl=self.config.preprocessing.equivalent_number_of_looks
            )
        else:
            self.speckle_filter = RefinedLeeFilter(
                window_size=self.config.preprocessing.filter_window_size,
                enl=self.config.preprocessing.equivalent_number_of_looks
            )
            
        self.normalizer = SARNormalizer(
            vv_range=self.config.preprocessing.vv_db_range,
            vh_range=self.config.preprocessing.vh_db_range
        )
        
        self.model_runner = OilSpillModelInference(
            weights_path=weights_path,
            in_channels=self.config.segmentation.in_channels,
            base_filters=self.config.segmentation.base_filters,
            device=self.config.device,
            tile_size=self.config.preprocessing.tile_size,
            overlap=self.config.preprocessing.tile_overlap
        )
        
        self.wind_gater = ERA5WindGater(config=self.config.wind_gating)
        self.feature_extractor = PolarimetricFeatureExtractor()
        self.lookalike_classifier = LookalikeClassifier(config=self.config.discrimination)
        self.fays_estimator = FaysSpreadingEstimator(config=self.config.fays)
        self.spatial_extractor = SpatialMorphologyExtractor(
            min_area_km2=self.config.segmentation.min_slick_area_km2
        )

    def process_scene(
        self,
        product_path: Union[str, Path],
        simulated_u10: Optional[float] = None,
        simulated_v10: Optional[float] = None,
        discard_on_wind_fail: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes complete end-to-end oil spill detection pipeline on a Sentinel-1 product.
        Returns:
            List of structured JSON detection dictionaries matching user specification.
        """
        logger.info(f"Ingesting SAR product: {product_path}")
        reader = Sentinel1Reader(str(product_path))
        vv_raw, vh_raw, metadata = reader.load_product()
        
        # 1. Radiometric Calibration to Sigma0 (Linear and Decibels)
        logger.info("Applying radiometric calibration and thermal noise removal...")
        vv_lin, vv_db = self.calibrator.calibrate_band(vv_raw)
        vh_lin, vh_db = self.calibrator.calibrate_band(vh_raw)

        # 2. Speckle Filtering (Refined Lee or Gamma-MAP)
        logger.info(f"Applying speckle filtering ({self.config.preprocessing.speckle_filter_type})...")
        vv_filtered_lin = self.speckle_filter.filter(vv_lin)
        vh_filtered_lin = self.speckle_filter.filter(vh_lin)
        vv_filtered_db = self.calibrator.linear_to_db(vv_filtered_lin)
        vh_filtered_db = self.calibrator.linear_to_db(vh_filtered_lin)

        # 3. Normalization & Dual-Polarization Tensor Assembly
        logger.info("Normalizing dual-polarization bands (VV, VH)...")
        dual_pol_norm = self.normalizer.normalize_dual_pol(vv_filtered_db, vh_filtered_db)

        # 4. Deep Learning Segmentation (Attention U-Net)
        logger.info("Running Attention U-Net semantic segmentation...")
        prob_mask, binary_mask = self.model_runner.predict_scene(
            dual_pol_norm,
            threshold=self.config.segmentation.probability_threshold
        )

        # 5. ECMWF ERA5 Atmospheric Wind Gating
        scene_center_lat = (metadata.bounds[1] + metadata.bounds[3]) / 2.0
        scene_center_lon = (metadata.bounds[0] + metadata.bounds[2]) / 2.0
        
        logger.info(f"Evaluating ERA5 wind vectors at ({scene_center_lat:.4f}, {scene_center_lon:.4f})...")
        wind_result: WindGatingResult = self.wind_gater.evaluate_wind_gate(
            lat=scene_center_lat,
            lon=scene_center_lon,
            timestamp_utc=metadata.timestamp_utc,
            u10=simulated_u10,
            v10=simulated_v10
        )
        logger.info(f"Wind Status: {wind_result.status} (Speed: {wind_result.wind_speed_ms} m/s, Dir: {wind_result.wind_direction_deg}°)")

        if not wind_result.is_valid and discard_on_wind_fail:
            logger.warning(f"Quality Gating: Discarding detections due to wind constraint. Reason: {wind_result.reason}")
            return []

        # 6. Morphological Extraction, Textural & Polarimetric Discrimination, Fay's Age Estimation
        logger.info("Extracting slick morphology, damping metrics, and Fay's hydrodynamic age...")
        slick_records: List[SlickDetectionRecord] = self.spatial_extractor.extract_slicks(
            binary_mask=binary_mask,
            vv_db=vv_filtered_db,
            metadata=metadata,
            wind_result=wind_result,
            fays_age_calculator=self.fays_estimator,
            feature_extractor=self.feature_extractor,
            classifier=self.lookalike_classifier
        )

        # Filter out records classified as lookalikes if confidence is low
        confirmed_records = [
            rec.to_dict() for rec in slick_records
            if rec.confidence_score >= 0.30
        ]

        logger.info(f"Pipeline complete. Detected {len(confirmed_records)} confirmed slick formations.")
        return confirmed_records
