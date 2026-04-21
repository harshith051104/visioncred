"""
VisionCred Vision Module
=========================
Uses YOLOv8 (ultralytics) to extract visual features from store images.

Outputs:
    - Total product count across all images
    - Per-image detection details
    - Shelf Density Index (0–1)
    - SKU Diversity Score (0–1)
    - Inventory Value Approximation (INR range)
    - Average detection confidence
    - Store size proxy from outside image
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)


class VisionAnalyzer:
    """
    Analyzes kirana store images using YOLOv8 object detection.
    
    The analyzer extracts three key visual features:
    
    1. **Shelf Density Index**: How packed the shelves are with products.
       Computed as (total detections / max expected detections) clamped to [0, 1].
    
    2. **SKU Diversity Score**: Variety of product categories.
       Computed as (unique COCO classes / max expected categories) clamped to [0, 1].
    
    3. **Inventory Value**: Estimated total visible inventory in INR.
       Uses average item pricing with variance for range estimation.
    """

    def __init__(self):
        """Initialize the YOLO model (lazy-loaded on first use)."""
        self._model = None
        self._model_loaded = False
        logger.info("VisionAnalyzer initialized (model will lazy-load)")

    def _ensure_model(self):
        """Lazy-load YOLOv8 model on first detection call."""
        if not self._model_loaded:
            try:
                from ultralytics import YOLO
                from src.config import YOLO_MODEL_NAME
                
                logger.info(f"Loading YOLO model: {YOLO_MODEL_NAME}")
                self._model = YOLO(YOLO_MODEL_NAME)
                self._model_loaded = True
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                raise

    def detect_objects(self, image_path: Path) -> Dict:
        """
        Run YOLOv8 detection on a single image.
        
        Args:
            image_path: Path to the image file (.webp, .jpg, .png)
        
        Returns:
            Dict containing:
                - detections: list of (class_name, confidence, bbox)
                - total_count: number of objects detected
                - unique_classes: set of unique class names
                - avg_confidence: mean detection confidence
                - image_area: pixel area of the image
        """
        from src.config import (
            YOLO_CONFIDENCE_THRESHOLD,
            YOLO_IOU_THRESHOLD,
            YOLO_IMAGE_SIZE,
        )

        self._ensure_model()

        try:
            # Run inference (force CPU to avoid CUDA compatibility issues)
            results = self._model(
                str(image_path),
                conf=YOLO_CONFIDENCE_THRESHOLD,
                iou=YOLO_IOU_THRESHOLD,
                imgsz=YOLO_IMAGE_SIZE,
                verbose=False,
                device="cpu",
            )

            detections = []
            unique_classes = set()
            confidences = []

            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = result.names[cls_id]
                        conf = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()

                        detections.append({
                            "class": cls_name,
                            "confidence": conf,
                            "bbox": bbox,
                        })
                        unique_classes.add(cls_name)
                        confidences.append(conf)

            # Get image dimensions
            img = Image.open(image_path)
            width, height = img.size
            image_area = width * height

            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            logger.info(
                f"  {image_path.name}: {len(detections)} objects, "
                f"{len(unique_classes)} classes, "
                f"avg_conf={avg_conf:.3f}"
            )

            return {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "detections": detections,
                "total_count": len(detections),
                "unique_classes": unique_classes,
                "avg_confidence": avg_conf,
                "image_area": image_area,
                "image_size": (width, height),
            }

        except Exception as e:
            logger.error(f"Detection failed for {image_path.name}: {e}")
            return {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "detections": [],
                "total_count": 0,
                "unique_classes": set(),
                "avg_confidence": 0.0,
                "image_area": 0,
                "image_size": (0, 0),
            }

    def analyze_store_images(
        self, image_paths: List[Path]
    ) -> Dict:
        """
        Analyze all images for a single store.
        
        Aggregates detection results across all images and computes
        store-level vision features.
        
        Args:
            image_paths: List of image file paths for one store.
        
        Returns:
            Dict with aggregated vision features.
        """
        from src.config import (
            MAX_ITEMS_PER_SHELF,
            MAX_SKU_CATEGORIES,
            AVG_ITEM_PRICE_INR,
            PRICE_VARIANCE_FACTOR,
        )

        all_results = []
        total_objects = 0
        all_unique_classes = set()
        all_confidences = []

        logger.info(f"Analyzing {len(image_paths)} images...")

        for img_path in image_paths:
            result = self.detect_objects(img_path)
            all_results.append(result)
            total_objects += result["total_count"]
            all_unique_classes |= result["unique_classes"]
            if result["avg_confidence"] > 0:
                all_confidences.append(result["avg_confidence"])

        # ── Shelf Density Index ────────────────────────────────────────────
        # Ratio of total detected items to the maximum expected
        # Higher density = more stocked store
        num_shelf_images = max(
            sum(1 for r in all_results if "shelf" in r["image_name"].lower()),
            1,
        )
        max_expected = MAX_ITEMS_PER_SHELF * num_shelf_images
        shelf_density_index = min(total_objects / max_expected, 1.0)

        # ── SKU Diversity Score ────────────────────────────────────────────
        # Ratio of unique product categories to max expected
        sku_diversity_score = min(
            len(all_unique_classes) / MAX_SKU_CATEGORIES, 1.0
        )

        # ── Inventory Value Approximation ──────────────────────────────────
        # Base value = total items × average price
        base_inventory = total_objects * AVG_ITEM_PRICE_INR
        inventory_value_low = base_inventory * (1 - PRICE_VARIANCE_FACTOR)
        inventory_value_high = base_inventory * (1 + PRICE_VARIANCE_FACTOR)

        # ── Store Size Proxy ───────────────────────────────────────────────
        # Estimated from outside image resolution and detection density
        # Normalized score 0–1 based on total image area and detections
        total_area = sum(r["image_area"] for r in all_results) or 1
        density_per_mpixel = (total_objects / (total_area / 1e6))
        store_size_proxy = min(density_per_mpixel / 500, 1.0)

        # ── Average Detection Quality ──────────────────────────────────────
        avg_detection_confidence = (
            float(np.mean(all_confidences)) if all_confidences else 0.0
        )

        vision_features = {
            "total_objects_detected": total_objects,
            "unique_classes": list(all_unique_classes),
            "num_unique_classes": len(all_unique_classes),
            "shelf_density_index": round(shelf_density_index, 4),
            "sku_diversity_score": round(sku_diversity_score, 4),
            "inventory_value_low": round(inventory_value_low, 2),
            "inventory_value_high": round(inventory_value_high, 2),
            "store_size_proxy": round(store_size_proxy, 4),
            "avg_detection_confidence": round(avg_detection_confidence, 4),
            "num_images_analyzed": len(image_paths),
            "per_image_results": [
                {
                    "image": r["image_name"],
                    "objects": r["total_count"],
                    "classes": list(r["unique_classes"]),
                    "avg_confidence": round(r["avg_confidence"], 4),
                }
                for r in all_results
            ],
        }

        logger.info(
            f"Vision analysis complete | "
            f"objects={total_objects}, "
            f"density={shelf_density_index:.3f}, "
            f"diversity={sku_diversity_score:.3f}, "
            f"inventory=Rs.{inventory_value_low:.0f}-{inventory_value_high:.0f}"
        )

        return vision_features
