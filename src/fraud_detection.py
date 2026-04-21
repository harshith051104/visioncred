"""
VisionCred Fraud Detection Module
====================================
Detects anomalies and potential fraud indicators in store data.

Checks:
    1. High inventory + low footfall mismatch
    2. Duplicate/similar images (perceptual hashing)
    3. Unrealistic SKU diversity for store size
    4. Missing or suspicious metadata

Each flag includes:
    - Risk type
    - Severity (low/medium/high)
    - Human-readable explanation
    - Recommendation
"""

from pathlib import Path
from typing import Dict, List, Tuple

from src.config import (
    IMAGE_HASH_THRESHOLD,
    HIGH_INVENTORY_THRESHOLD,
    LOW_FOOTFALL_THRESHOLD,
    SKU_ANOMALY_RATIO,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class FraudDetector:
    """
    Detects anomalous patterns in store data that may indicate
    fraud, data quality issues, or unreliable estimates.
    """

    def __init__(self):
        logger.info("FraudDetector initialized")

    def check_inventory_footfall_mismatch(
        self, features: Dict
    ) -> List[Dict]:
        """
        Flag stores with high inventory but low expected footfall.
        
        Logic:
            If shelf_density_index > 0.7 AND geo_footfall_score < 0.3,
            the store claims lots of stock but is in a low-traffic area.
            This is unusual and needs verification.
        """
        flags = []
        density = features.get("shelf_density_index", 0)
        footfall = features.get("geo_footfall_score", 0.5)

        if (
            density > HIGH_INVENTORY_THRESHOLD
            and footfall < LOW_FOOTFALL_THRESHOLD
        ):
            flags.append({
                "type": "inventory_footfall_mismatch",
                "severity": "high",
                "detail": (
                    f"High inventory (density={density:.2f}) in low-footfall "
                    f"area (footfall={footfall:.2f}). This is unusual — "
                    f"well-stocked stores are typically in higher-traffic areas."
                ),
                "recommendation": (
                    "Verify store location and inventory claims. "
                    "Consider physical verification visit."
                ),
            })

        return flags

    def check_duplicate_images(
        self, image_paths: List[Path]
    ) -> List[Dict]:
        """
        Detect duplicate or near-duplicate images using perceptual hashing.
        
        Logic:
            Compute perceptual hash (pHash) for each image.
            If two images have Hamming distance < threshold,
            flag as potential duplicates.
        """
        flags = []

        try:
            import imagehash
            from PIL import Image

            hashes = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path)
                    h = imagehash.phash(img)
                    hashes.append((img_path.name, h))
                except Exception as e:
                    logger.warning(
                        f"Could not hash {img_path.name}: {e}"
                    )

            # Compare all pairs
            for i in range(len(hashes)):
                for j in range(i + 1, len(hashes)):
                    name_i, hash_i = hashes[i]
                    name_j, hash_j = hashes[j]
                    distance = hash_i - hash_j

                    if distance < IMAGE_HASH_THRESHOLD:
                        flags.append({
                            "type": "duplicate_images",
                            "severity": "medium",
                            "detail": (
                                f"Images '{name_i}' and '{name_j}' are "
                                f"very similar (hash distance={distance}). "
                                f"May be duplicates or taken from same angle."
                            ),
                            "recommendation": (
                                "Request different images from varied angles "
                                "for more accurate analysis."
                            ),
                        })

        except ImportError:
            logger.warning(
                "imagehash not installed — skipping duplicate detection"
            )

        return flags

    def check_sku_anomaly(self, features: Dict) -> List[Dict]:
        """
        Flag unrealistic SKU diversity relative to store size.
        
        Logic:
            If (unique classes / store_size_proxy) > threshold,
            the store appears to have too many product types
            for its apparent size — possibly inflated data.
        """
        flags = []
        sku_count = features.get("num_unique_classes", 0)
        store_size = features.get("store_size_proxy", 0.1)

        if store_size > 0 and sku_count > 0:
            ratio = sku_count / max(store_size, 0.01)
            if ratio > SKU_ANOMALY_RATIO * 100:
                flags.append({
                    "type": "unrealistic_sku_diversity",
                    "severity": "medium",
                    "detail": (
                        f"SKU diversity ({sku_count} categories) seems high "
                        f"relative to store size (proxy={store_size:.3f}). "
                        f"Ratio: {ratio:.1f}."
                    ),
                    "recommendation": (
                        "Verify product variety with physical inspection. "
                        "May indicate data from multiple stores."
                    ),
                })

        return flags

    def check_metadata_quality(
        self, has_metadata: bool, gps_source: str
    ) -> List[Dict]:
        """
        Flag issues with missing or fallback metadata.
        """
        flags = []

        if not has_metadata or gps_source == "default_fallback":
            flags.append({
                "type": "missing_gps_data",
                "severity": "low",
                "detail": (
                    "GPS coordinates are missing or using default values. "
                    "Geo-based features (footfall, location type) may be "
                    "inaccurate."
                ),
                "recommendation": (
                    "Collect actual GPS coordinates from the store location "
                    "for more accurate analysis."
                ),
            })

        return flags

    def check_low_image_quality(self, features: Dict) -> List[Dict]:
        """
        Flag stores with very low detection confidence (blurry/bad images).
        """
        flags = []
        avg_conf = features.get("avg_detection_confidence", 0)
        num_images = features.get("num_images_analyzed", 0)

        if avg_conf > 0 and avg_conf < 0.3:
            flags.append({
                "type": "low_image_quality",
                "severity": "medium",
                "detail": (
                    f"Average detection confidence is low ({avg_conf:.2f}). "
                    f"Images may be blurry, dark, or taken from poor angles."
                ),
                "recommendation": (
                    "Request clearer, well-lit images taken from straight-on "
                    "angles for better analysis."
                ),
            })

        if num_images < 3:
            flags.append({
                "type": "insufficient_images",
                "severity": "low",
                "detail": (
                    f"Only {num_images} images provided. "
                    f"Minimum 3-5 images recommended for reliable analysis."
                ),
                "recommendation": (
                    "Provide shelf, counter, and outside images for "
                    "comprehensive assessment."
                ),
            })

        return flags

    def check_view_coverage(self, image_paths: List[Path]) -> List[Dict]:
        """
        Flag incomplete or biased image coverage across mandatory views.
        """
        flags = []
        shelf_count = 0
        counter_count = 0
        outside_count = 0

        for path in image_paths:
            name = path.stem.lower()
            if name.startswith("shelf"):
                shelf_count += 1
            elif name.startswith("counter"):
                counter_count += 1
            elif name.startswith("outside"):
                outside_count += 1

        missing_views = []
        if shelf_count < 2:
            missing_views.append("shelf")
        if counter_count < 1:
            missing_views.append("counter")
        if outside_count < 1:
            missing_views.append("outside")

        if missing_views:
            flags.append({
                "type": "missing_mandatory_views",
                "severity": "high",
                "detail": (
                    "Mandatory views are missing or underrepresented: "
                    f"{', '.join(missing_views)}. "
                    "This increases risk of selective photography."
                ),
                "recommendation": (
                    "Collect 3-5 images covering shelves, counter, and "
                    "storefront/street view before final underwriting."
                ),
            })

        if len(image_paths) < 5:
            flags.append({
                "type": "limited_view_coverage",
                "severity": "medium",
                "detail": (
                    f"Only {len(image_paths)} images provided; limited coverage "
                    "can hide inventory and throughput patterns."
                ),
                "recommendation": (
                    "Request additional angles and aisle-depth coverage."
                ),
            })

        return flags

    def analyze(
        self,
        features: Dict,
        image_paths: List[Path],
        has_metadata: bool,
    ) -> Dict:
        """
        Run all fraud detection checks.
        
        Args:
            features: Consolidated feature dictionary.
            image_paths: List of all image paths for the store.
            has_metadata: Whether real GPS data was available.
        
        Returns:
            Dict with risk_flags, overall_risk_level, and recommendation.
        """
        logger.info("Running fraud detection checks...")

        all_flags = []

        # Run all checks
        all_flags.extend(
            self.check_inventory_footfall_mismatch(features)
        )
        all_flags.extend(
            self.check_duplicate_images(image_paths)
        )
        all_flags.extend(
            self.check_sku_anomaly(features)
        )
        all_flags.extend(
            self.check_metadata_quality(
                has_metadata,
                features.get("gps_source", ""),
            )
        )
        all_flags.extend(
            self.check_low_image_quality(features)
        )
        all_flags.extend(
            self.check_view_coverage(image_paths)
        )

        # Determine overall risk level
        severities = [f["severity"] for f in all_flags]
        if "high" in severities:
            overall_risk = "HIGH"
            recommendation = "manual_review"
        elif "medium" in severities:
            overall_risk = "MEDIUM"
            recommendation = "needs_verification"
        else:
            overall_risk = "LOW"
            recommendation = "proceed"

        fraud_output = {
            "risk_flags": all_flags,
            "num_flags": len(all_flags),
            "overall_risk_level": overall_risk,
            "recommendation": recommendation,
        }

        logger.info(
            f"Fraud detection complete | "
            f"flags={len(all_flags)}, risk_level={overall_risk}"
        )

        return fraud_output
