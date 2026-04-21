"""
VisionCred Configuration Module
================================
Central configuration for all system parameters, thresholds, and constants.
Designed for explainability — every magic number is documented.
"""

import os
from pathlib import Path

# ─── Project Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── YOLO Configuration ───────────────────────────────────────────────────────
YOLO_MODEL_NAME = "yolov8n.pt"  # Nano model for speed; swap to yolov8s.pt for accuracy
YOLO_CONFIDENCE_THRESHOLD = 0.25  # Minimum detection confidence
YOLO_IOU_THRESHOLD = 0.45  # Non-max suppression IoU threshold
YOLO_IMAGE_SIZE = 640  # Input image size for YOLO

# ─── Image Types ───────────────────────────────────────────────────────────────
SHELF_IMAGE_PREFIXES = ["shelf"]
COUNTER_IMAGE_PREFIX = "counter"
OUTSIDE_IMAGE_PREFIX = "outside"
SUPPORTED_EXTENSIONS = [".webp", ".jpg", ".jpeg", ".png"]

# ─── Vision Feature Thresholds ─────────────────────────────────────────────────
# Shelf Density: ratio of detected objects to shelf image area-equivalent
# A well-stocked kirana shelf has ~50-150 visible items per shelf image
MAX_ITEMS_PER_SHELF = 150  # Upper bound for normalization
MIN_ITEMS_PER_SHELF = 5   # Below this → poorly stocked

# SKU Diversity: unique COCO classes detected across all images
# Typical kirana stores carry 20-80 COCO-detectable categories
MAX_SKU_CATEGORIES = 60
MIN_SKU_CATEGORIES = 3

# ─── Inventory Value Estimation ────────────────────────────────────────────────
# Average price per item in INR (based on typical kirana store FMCG pricing)
# This is a weighted average: ₹5 sachets to ₹500 items
AVG_ITEM_PRICE_INR = 45.0
PRICE_VARIANCE_FACTOR = 0.3  # ±30% variance for range estimation

# ─── Geo Intelligence ──────────────────────────────────────────────────────────
# Hyderabad metro area reference coordinates (for distance-based scoring)
METRO_CENTERS = {
    "hyderabad": (17.3850, 78.4867),
    "secunderabad": (17.4399, 78.4983),
    "kukatpally": (17.4947, 78.3996),
    "gachibowli": (17.4401, 78.3489),
    "lb_nagar": (17.3457, 78.5522),
    "miyapur": (17.4969, 78.3548),
    "dilsukhnagar": (17.3688, 78.5247),
    "ameerpet": (17.4375, 78.4483),
}

# Distance thresholds for location classification (in km)
URBAN_RADIUS_KM = 10.0       # Within 10km of any metro center → urban
SEMI_URBAN_RADIUS_KM = 25.0  # 10-25km → semi-urban
# Beyond 25km → rural

# Location multipliers for economic model
LOCATION_MULTIPLIERS = {
    "urban": 1.4,
    "semi_urban": 1.0,
    "rural": 0.7,
}

# Footfall base scores by location type
FOOTFALL_BASE = {
    "urban": 0.8,
    "semi_urban": 0.5,
    "rural": 0.3,
}

# ─── Economic Model Parameters ────────────────────────────────────────────────
# Inventory Turnover Rate: how many times daily inventory turns over
# Kirana throughput is high; visible shelf stock is only a subset of working stock.
TURNOVER_RATE_LOW = 0.20   # Conservative turnover of effective inventory
TURNOVER_RATE_HIGH = 0.45  # Optimistic turnover of effective inventory

# Visible shelves typically represent only part of the total working inventory.
# This uplift converts visible inventory estimate to effective operating stock.
VISIBLE_TO_TOTAL_INVENTORY_LOW = 8.0
VISIBLE_TO_TOTAL_INVENTORY_HIGH = 14.0

# Demand Factor: seasonal/market demand multiplier
DEMAND_FACTOR_LOW = 0.8
DEMAND_FACTOR_HIGH = 1.2

# Profit Margins for kirana stores (net margin after costs)
MARGIN_LOW = 0.10   # 10% margin (minimum typical)
MARGIN_HIGH = 0.20  # 20% margin (well-managed store)

# Days in month for revenue calculation
DAYS_IN_MONTH = 30

# ─── Fraud Detection Thresholds ───────────────────────────────────────────────
# Image similarity threshold for duplicate detection (perceptual hash)
IMAGE_HASH_THRESHOLD = 10  # Hamming distance; <10 means very similar

# Anomaly: high inventory but low footfall
HIGH_INVENTORY_THRESHOLD = 0.7  # Inventory value index > 0.7
LOW_FOOTFALL_THRESHOLD = 0.3    # Footfall score < 0.3

# Unrealistic SKU diversity (too many unique items for store size)
SKU_ANOMALY_RATIO = 3.0  # If SKU diversity / store_size_proxy > 3 → suspicious

# ─── Confidence Score Weights ──────────────────────────────────────────────────
CONFIDENCE_WEIGHTS = {
    "image_count": 0.25,       # More images → higher confidence
    "detection_quality": 0.35,  # Higher avg detection confidence → better
    "consistency": 0.20,        # Consistent detections across images → better
    "metadata_present": 0.20,   # Having GPS data → better
}

# Expected images per store for full confidence
EXPECTED_IMAGE_COUNT = 5

# ─── Logging ───────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"
