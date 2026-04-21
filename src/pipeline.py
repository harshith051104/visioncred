"""
VisionCred Pipeline
=====================
End-to-end orchestration pipeline:

    images -> vision -> features -> geo -> economic_model -> fraud -> output

Handles single-store and batch processing with structured JSON output.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.vision import VisionAnalyzer
from src.geo_intel import GeoAnalyzer
from src.features import FeatureEngineer
from src.economic_model import EconomicModel
from src.fraud_detection import FraudDetector
from src.confidence import ConfidenceScorer
from src.config import OUTPUT_DIR
from utils.data_loader import DataLoader, StoreData
from utils.logger import get_logger

logger = get_logger(__name__)


class VisionCredPipeline:
    """
    Main pipeline orchestrator for VisionCred credit assessment.
    
    Pipeline Flow:
        1. Load store data (images + metadata)
        2. Run YOLO vision analysis on all images
        3. Generate geo intelligence from GPS coordinates
        4. Engineer consolidated features
        5. Run economic model for cash flow estimates
        6. Detect fraud indicators
        7. Compute confidence score
        8. Produce structured JSON output
    """

    def __init__(self):
        """Initialize all pipeline components."""
        logger.info("=" * 60)
        logger.info("  VisionCred Pipeline - Initializing")
        logger.info("=" * 60)

        self.vision = VisionAnalyzer()
        self.geo = GeoAnalyzer()
        self.features = FeatureEngineer()
        self.economic = EconomicModel()
        self.fraud = FraudDetector()
        self.confidence = ConfidenceScorer()
        self.data_loader = DataLoader()

        logger.info("All pipeline components initialized successfully")

    def process_store(self, store: StoreData) -> Dict:
        """
        Process a single store through the full pipeline.
        
        Args:
            store: StoreData instance with loaded images and metadata.
        
        Returns:
            Complete assessment dictionary in the required JSON format.
        """
        start_time = time.time()
        logger.info(f"\n{'-' * 60}")
        logger.info(f"  Processing: {store.store_id}")
        logger.info(f"{'-' * 60}")

        # ── Stage 1: Vision Analysis ───────────────────────────────────
        logger.info(">> Stage 1: Vision Analysis")
        vision_features = self.vision.analyze_store_images(
            store.all_images
        )

        # ── Stage 2: Geo Intelligence ─────────────────────────────────
        logger.info(">> Stage 2: Geo Intelligence")
        geo_features = self.geo.analyze(
            latitude=store.latitude,
            longitude=store.longitude,
            has_metadata=store.has_metadata,
        )

        # ── Stage 3: Feature Engineering ──────────────────────────────
        logger.info(">> Stage 3: Feature Engineering")
        consolidated_features = self.features.build_features(
            vision_features, geo_features
        )
        logger.info(
            self.features.get_feature_summary(consolidated_features)
        )

        # ── Stage 4: Economic Model ───────────────────────────────────
        logger.info(">> Stage 4: Economic Model")
        economic_output = self.economic.compute(consolidated_features)

        # ── Stage 5: Fraud Detection ──────────────────────────────────
        logger.info(">> Stage 5: Fraud Detection")
        fraud_output = self.fraud.analyze(
            features=consolidated_features,
            image_paths=store.all_images,
            has_metadata=store.has_metadata,
        )

        # ── Stage 6: Confidence Scoring ───────────────────────────────
        logger.info(">> Stage 6: Confidence Scoring")
        confidence_output = self.confidence.compute(
            features=consolidated_features,
            vision_features=vision_features,
        )

        elapsed = time.time() - start_time

        # ── Assemble Final Output ──────────────────────────────────────
        result = {
            "store_id": store.store_id,
            "daily_sales_range": economic_output["daily_sales_range"],
            "monthly_revenue_range": economic_output["monthly_revenue_range"],
            "monthly_income_range": economic_output["monthly_income_range"],
            "confidence_score": confidence_output["confidence_score"],
            "confidence_interpretation": confidence_output["interpretation"],
            "risk_flags": [
                f["type"] for f in fraud_output["risk_flags"]
            ],
            "risk_details": fraud_output["risk_flags"],
            "overall_risk_level": fraud_output["overall_risk_level"],
            "key_drivers": economic_output["key_drivers"],
            "recommendation": fraud_output["recommendation"],
            "features": {
                "shelf_density_index": consolidated_features[
                    "shelf_density_index"
                ],
                "sku_diversity_score": consolidated_features[
                    "sku_diversity_score"
                ],
                "inventory_value_range": [
                    consolidated_features["inventory_value_low"],
                    consolidated_features["inventory_value_high"],
                ],
                "store_size_proxy": consolidated_features[
                    "store_size_proxy"
                ],
                "geo_footfall_score": consolidated_features[
                    "geo_footfall_score"
                ],
                "competition_density": consolidated_features[
                    "competition_density"
                ],
                "location_type": consolidated_features["location_type"],
                "store_viability_index": consolidated_features[
                    "store_viability_index"
                ],
            },
            "geo_info": {
                "location_type": geo_features["location_type"],
                "nearest_metro": geo_features["nearest_metro_center"],
                "distance_km": geo_features["distance_to_nearest_km"],
                "gps_source": geo_features["gps_source"],
                "coordinates": geo_features["coordinates"],
            },
            "confidence_breakdown": confidence_output["components"],
            "model_parameters": economic_output["model_parameters"],
            "formula": economic_output["formula"],
            "processing_time_seconds": round(elapsed, 2),
            "num_images_processed": len(store.all_images),
        }

        logger.info(f"\n[OK] {store.store_id} processed in {elapsed:.2f}s")
        self._print_summary(result)

        return result

    def _print_summary(self, result: Dict):
        """Print a formatted summary of results."""
        lines = [
            "",
            "+==================================================+",
            f"|  CREDIT ASSESSMENT: {result['store_id']:>28s}  |",
            "+==================================================+",
            f"|  Daily Sales:    Rs.{result['daily_sales_range'][0]:>8,.0f} - Rs.{result['daily_sales_range'][1]:>8,.0f}  |",
            f"|  Monthly Rev:    Rs.{result['monthly_revenue_range'][0]:>8,.0f} - Rs.{result['monthly_revenue_range'][1]:>8,.0f}  |",
            f"|  Monthly Income: Rs.{result['monthly_income_range'][0]:>8,.0f} - Rs.{result['monthly_income_range'][1]:>8,.0f}  |",
            f"|  Confidence:     {result['confidence_score']:>8.2f}                      |",
            f"|  Risk Level:     {result['overall_risk_level']:>8s}                      |",
            "+==================================================+",
        ]
        for line in lines:
            logger.info(line)

    def process_batch(
        self,
        store_paths: Optional[List[Path]] = None,
    ) -> List[Dict]:
        """
        Process multiple stores in batch mode.
        
        Args:
            store_paths: Optional list of specific store paths.
                         If None, processes all discovered stores.
        
        Returns:
            List of assessment dictionaries.
        """
        logger.info("\n" + "=" * 60)
        logger.info("  VisionCred - BATCH PROCESSING")
        logger.info("=" * 60)

        if store_paths:
            stores = [
                self.data_loader.load_store(p) for p in store_paths
            ]
        else:
            stores = self.data_loader.load_all_stores()

        if not stores:
            logger.warning("No stores found to process!")
            return []

        logger.info(f"Processing {len(stores)} stores...")

        results = []
        for i, store in enumerate(stores, 1):
            logger.info(f"\n[{i}/{len(stores)}] Processing {store.store_id}")
            try:
                result = self.process_store(store)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed to process {store.store_id}: {e}",
                    exc_info=True,
                )
                results.append({
                    "store_id": store.store_id,
                    "error": str(e),
                    "status": "failed",
                })

        # Save batch results
        self._save_results(results)

        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"  Batch complete: {len(results)} stores processed"
        )
        logger.info(f"{'=' * 60}")

        return results

    def process_single_store_path(self, store_path: Path) -> Dict:
        """
        Process a single store from its directory path.
        
        Args:
            store_path: Path to the store directory.
        
        Returns:
            Assessment dictionary.
        """
        store = self.data_loader.load_store(store_path)
        return self.process_store(store)

    def _save_results(self, results: List[Dict]):
        """Save batch results to JSON files."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save individual store results
        for result in results:
            if "error" not in result:
                store_id = result["store_id"]
                output_path = OUTPUT_DIR / f"{store_id}_assessment.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved: {output_path}")

        # Save combined batch report
        batch_path = OUTPUT_DIR / "batch_results.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved batch report: {batch_path}")

        # Save summary table
        summary_path = OUTPUT_DIR / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("VisionCred - Batch Assessment Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(
                f"{'Store':<12} {'Daily Sales (Rs.)':<22} "
                f"{'Monthly Rev (Rs.)':<24} {'Confidence':<12} {'Risk':<8}\n"
            )
            f.write("-" * 80 + "\n")

            for r in results:
                if "error" not in r:
                    ds = r["daily_sales_range"]
                    mr = r["monthly_revenue_range"]
                    f.write(
                        f"{r['store_id']:<12} "
                        f"{ds[0]:>8,.0f} - {ds[1]:>8,.0f}   "
                        f"{mr[0]:>9,.0f} - {mr[1]:>9,.0f}   "
                        f"{r['confidence_score']:>8.2f}     "
                        f"{r['overall_risk_level']:<8}\n"
                    )
                else:
                    f.write(
                        f"{r['store_id']:<12} {'FAILED':^22} "
                        f"{'—':^24} {'—':^12} {'—':^8}\n"
                    )

        logger.info(f"Saved summary: {summary_path}")
