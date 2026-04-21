"""
VisionCred Confidence Scoring Module
======================================
Computes an overall confidence score for each store's credit assessment.

The score reflects HOW MUCH we trust our own estimate, based on:
    - Image count: More images → more data → higher confidence
    - Detection quality: Higher avg YOLO confidence → better detections
    - Consistency: Similar detection counts across images → stable signal
    - Metadata availability: Real GPS data → better geo features

Score range: 0.0 (no confidence) to 1.0 (full confidence)
"""

from typing import Dict, List
import numpy as np

from src.config import CONFIDENCE_WEIGHTS, EXPECTED_IMAGE_COUNT
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceScorer:
    """
    Computes a transparent confidence score for credit assessments.
    
    Each component is weighted and combined linearly:
        confidence = Σ(weight_i × component_i)
    
    Components:
        - image_count_score:      min(num_images / expected_count, 1.0)
        - detection_quality_score: avg YOLO detection confidence
        - consistency_score:       inverse of coefficient of variation
        - metadata_score:          1.0 if GPS present, 0.5 if fallback
    """

    def __init__(self):
        self.weights = CONFIDENCE_WEIGHTS
        logger.info(
            f"ConfidenceScorer initialized | weights={self.weights}"
        )

    def _image_count_score(self, num_images: int) -> float:
        """
        Score based on how many images were provided.
        
        5 images (expected) → 1.0
        3 images → 0.6
        1 image → 0.2
        """
        return min(num_images / EXPECTED_IMAGE_COUNT, 1.0)

    def _detection_quality_score(
        self, avg_confidence: float
    ) -> float:
        """
        Score based on average YOLO detection confidence.
        
        High confidence detections (>0.7) → reliable feature extraction.
        Low confidence (<0.3) → unreliable.
        """
        return min(avg_confidence, 1.0)

    def _consistency_score(
        self, per_image_results: List[Dict]
    ) -> float:
        """
        Score based on consistency of detections across images.
        
        If all shelf images detect similar number of objects,
        the signal is stable and trustworthy.
        
        Uses inverse coefficient of variation (CV).
        Lower CV = more consistent = higher score.
        """
        counts = [r.get("objects", 0) for r in per_image_results]

        if not counts or len(counts) < 2:
            return 0.5  # Neutral when insufficient data

        mean = np.mean(counts)
        std = np.std(counts)

        if mean == 0:
            return 0.3  # No detections at all → low confidence

        # Coefficient of variation
        cv = std / mean

        # Invert: low CV → high score
        # CV of 0 → score 1.0; CV of 2+ → score ~0.1
        consistency = max(1.0 - (cv * 0.5), 0.1)

        return round(consistency, 4)

    def _metadata_score(self, gps_source: str) -> float:
        """
        Score based on metadata availability.
        
        Real GPS data → 1.0
        Fallback defaults → 0.4 (still usable but less reliable)
        """
        if gps_source == "metadata":
            return 1.0
        return 0.4

    def compute(
        self,
        features: Dict,
        vision_features: Dict,
    ) -> Dict:
        """
        Compute the overall confidence score.
        
        Args:
            features: Consolidated feature dictionary.
            vision_features: Raw vision analysis output (for per-image data).
        
        Returns:
            Dict with overall score and component breakdown.
        """
        # Compute each component
        img_score = self._image_count_score(
            features.get("num_images_analyzed", 0)
        )
        det_score = self._detection_quality_score(
            features.get("avg_detection_confidence", 0)
        )
        con_score = self._consistency_score(
            vision_features.get("per_image_results", [])
        )
        meta_score = self._metadata_score(
            features.get("gps_source", "default_fallback")
        )

        # Weighted combination
        overall = (
            self.weights["image_count"] * img_score
            + self.weights["detection_quality"] * det_score
            + self.weights["consistency"] * con_score
            + self.weights["metadata_present"] * meta_score
        )

        overall = round(min(overall, 1.0), 4)

        confidence_output = {
            "confidence_score": overall,
            "components": {
                "image_count": {
                    "score": round(img_score, 4),
                    "weight": self.weights["image_count"],
                    "detail": (
                        f"{features.get('num_images_analyzed', 0)} images "
                        f"provided (expected {EXPECTED_IMAGE_COUNT})"
                    ),
                },
                "detection_quality": {
                    "score": round(det_score, 4),
                    "weight": self.weights["detection_quality"],
                    "detail": (
                        f"Average detection confidence: "
                        f"{features.get('avg_detection_confidence', 0):.3f}"
                    ),
                },
                "consistency": {
                    "score": round(con_score, 4),
                    "weight": self.weights["consistency"],
                    "detail": "Cross-image detection consistency",
                },
                "metadata": {
                    "score": round(meta_score, 4),
                    "weight": self.weights["metadata_present"],
                    "detail": (
                        f"GPS source: {features.get('gps_source', 'unknown')}"
                    ),
                },
            },
        }

        # Confidence interpretation
        if overall >= 0.8:
            confidence_output["interpretation"] = "HIGH — Estimate is reliable"
        elif overall >= 0.5:
            confidence_output["interpretation"] = (
                "MODERATE — Estimate is reasonable but verify key assumptions"
            )
        else:
            confidence_output["interpretation"] = (
                "LOW — Estimate has high uncertainty; more data needed"
            )

        logger.info(
            f"Confidence score: {overall:.3f} | "
            f"img={img_score:.2f}, det={det_score:.2f}, "
            f"con={con_score:.2f}, meta={meta_score:.2f}"
        )

        return confidence_output
