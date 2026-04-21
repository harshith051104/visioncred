"""
VisionCred Feature Engineering Module
=======================================
Consolidates vision and geo features into a unified feature vector
for the economic model.

All features are normalized to [0, 1] for consistency and explainability.
"""

from typing import Dict
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Merges vision-derived and geo-derived features into a single
    feature dictionary for downstream economic modeling.
    
    Feature Vector:
        - shelf_density_index    (0–1): How stocked the shelves are
        - sku_diversity_score    (0–1): Variety of product categories
        - inventory_value_low    (INR): Lower bound of visible inventory
        - inventory_value_high   (INR): Upper bound of visible inventory
        - store_size_proxy       (0–1): Estimated relative store size
        - geo_footfall_score     (0–1): Expected customer traffic
        - competition_density    (0–1): Nearby competition level
        - location_type          (str): urban/semi_urban/rural
        - location_multiplier    (float): Economic multiplier for location
    """

    def __init__(self):
        logger.info("FeatureEngineer initialized")

    def build_features(
        self, vision_features: Dict, geo_features: Dict
    ) -> Dict:
        """
        Combine vision and geo features into a unified feature dict.
        
        Args:
            vision_features: Output from VisionAnalyzer.analyze_store_images()
            geo_features: Output from GeoAnalyzer.analyze()
        
        Returns:
            Consolidated feature dictionary with all engineered features.
        """
        features = {
            # ── Vision Features ────────────────────────────────────────
            "shelf_density_index": vision_features.get(
                "shelf_density_index", 0.0
            ),
            "sku_diversity_score": vision_features.get(
                "sku_diversity_score", 0.0
            ),
            "inventory_value_low": vision_features.get(
                "inventory_value_low", 0.0
            ),
            "inventory_value_high": vision_features.get(
                "inventory_value_high", 0.0
            ),
            "store_size_proxy": vision_features.get(
                "store_size_proxy", 0.0
            ),
            "total_objects_detected": vision_features.get(
                "total_objects_detected", 0
            ),
            "num_unique_classes": vision_features.get(
                "num_unique_classes", 0
            ),
            "avg_detection_confidence": vision_features.get(
                "avg_detection_confidence", 0.0
            ),
            "num_images_analyzed": vision_features.get(
                "num_images_analyzed", 0
            ),

            # ── Geo Features ──────────────────────────────────────────
            "geo_footfall_score": geo_features.get(
                "footfall_score", 0.3
            ),
            "competition_density": geo_features.get(
                "competition_density", 0.5
            ),
            "location_type": geo_features.get(
                "location_type", "semi_urban"
            ),
            "location_multiplier": geo_features.get(
                "location_multiplier", 1.0
            ),
            "nearest_metro_center": geo_features.get(
                "nearest_metro_center", "unknown"
            ),
            "distance_to_nearest_km": geo_features.get(
                "distance_to_nearest_km", 0.0
            ),
            "gps_source": geo_features.get(
                "gps_source", "default_fallback"
            ),
        }

        # ── Derived Composite Features ─────────────────────────────────
        # Store Viability Index: combined metric of inventory × footfall
        features["store_viability_index"] = round(
            (
                features["shelf_density_index"] * 0.4
                + features["sku_diversity_score"] * 0.3
                + features["geo_footfall_score"] * 0.3
            ),
            4,
        )

        # Market Potential: how much opportunity exists
        features["market_potential"] = round(
            features["geo_footfall_score"]
            * (1 - features["competition_density"] * 0.5)
            * features["location_multiplier"],
            4,
        )

        logger.info(
            f"Features built | "
            f"density={features['shelf_density_index']:.3f}, "
            f"diversity={features['sku_diversity_score']:.3f}, "
            f"footfall={features['geo_footfall_score']:.3f}, "
            f"viability={features['store_viability_index']:.3f}"
        )

        return features

    def get_feature_summary(self, features: Dict) -> str:
        """
        Generate a human-readable feature summary for reporting.
        
        Args:
            features: Consolidated feature dictionary.
        
        Returns:
            Formatted string summary.
        """
        lines = [
            "+-------------------------------------------------+",
            "|            FEATURE ENGINEERING SUMMARY           |",
            "+-------------------------------------------------+",
            f"| Shelf Density Index:    {features['shelf_density_index']:>8.4f}              |",
            f"| SKU Diversity Score:    {features['sku_diversity_score']:>8.4f}              |",
            f"| Store Size Proxy:       {features['store_size_proxy']:>8.4f}              |",
            f"| Inventory Value: Rs.{features['inventory_value_low']:>8.0f} - Rs.{features['inventory_value_high']:>8.0f} |",
            "+-------------------------------------------------+",
            f"| Footfall Score:         {features['geo_footfall_score']:>8.4f}              |",
            f"| Competition Density:    {features['competition_density']:>8.4f}              |",
            f"| Location Type:          {features['location_type']:>12s}          |",
            f"| Location Multiplier:    {features['location_multiplier']:>8.2f}              |",
            "+-------------------------------------------------+",
            f"| Store Viability Index:  {features['store_viability_index']:>8.4f}              |",
            f"| Market Potential:       {features['market_potential']:>8.4f}              |",
            "+-------------------------------------------------+",
        ]
        return "\n".join(lines)
