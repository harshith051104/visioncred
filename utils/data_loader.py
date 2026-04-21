"""
VisionCred Data Loader
=======================
Handles loading store data: images, metadata, and directory enumeration.
Robust handling of missing files and malformed metadata.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from src.config import (
    DATA_DIR,
    SUPPORTED_EXTENSIONS,
    SHELF_IMAGE_PREFIXES,
    COUNTER_IMAGE_PREFIX,
    OUTSIDE_IMAGE_PREFIX,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class StoreData:
    """Container for all data associated with a single kirana store."""

    def __init__(self, store_id: str, store_path: Path):
        self.store_id = store_id
        self.store_path = store_path
        self.metadata: Dict = {}
        self.shelf_images: List[Path] = []
        self.counter_images: List[Path] = []
        self.outside_images: List[Path] = []
        self.extra_images: List[Path] = []
        self.all_images: List[Path] = []
        self.latitude: Optional[float] = None
        self.longitude: Optional[float] = None
        self.has_metadata: bool = False

    def __repr__(self) -> str:
        return (
            f"StoreData(id={self.store_id}, "
            f"shelves={len(self.shelf_images)}, "
            f"counter={len(self.counter_images)}, "
            f"outside={len(self.outside_images)}, "
            f"extra={len(self.extra_images)}, "
            f"has_gps={self.has_metadata})"
        )


class DataLoader:
    """
    Loads and validates store data from the raw data directory.
    
    Responsibilities:
        - Enumerate store directories
        - Classify images by type (shelf, counter, outside, extra)
        - Parse metadata.json for GPS coordinates
        - Handle missing/malformed data gracefully
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        logger.info(f"DataLoader initialized | data_dir={self.data_dir}")

    def discover_stores(self) -> List[Path]:
        """
        Find all store directories in the data folder.
        
        Returns:
            List of paths to store directories, sorted by name.
        """
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return []

        stores = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith("store_")
        ])
        logger.info(f"Discovered {len(stores)} store directories")
        return stores

    def _classify_image(self, image_path: Path) -> str:
        """
        Classify an image based on its filename.
        
        Returns:
            'shelf', 'counter', 'outside', or 'extra'
        """
        name = image_path.stem.lower()

        for prefix in SHELF_IMAGE_PREFIXES:
            if name.startswith(prefix):
                return "shelf"

        if name.startswith(COUNTER_IMAGE_PREFIX):
            return "counter"

        if name.startswith(OUTSIDE_IMAGE_PREFIX):
            return "outside"

        return "extra"

    def _load_metadata(self, store_path: Path) -> Tuple[Dict, bool]:
        """
        Load and validate metadata.json from a store directory.
        
        Returns:
            Tuple of (metadata_dict, is_valid)
        """
        metadata_path = store_path / "metadata.json"

        if not metadata_path.exists():
            logger.warning(f"No metadata.json found for {store_path.name}")
            return {}, False

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Validate required fields
            if "latitude" in metadata and "longitude" in metadata:
                lat = float(metadata["latitude"])
                lon = float(metadata["longitude"])
                # Basic coordinate validation
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    logger.info(
                        f"Valid metadata for {store_path.name}: "
                        f"lat={lat:.4f}, lon={lon:.4f}"
                    )
                    return metadata, True
                else:
                    logger.warning(
                        f"Invalid coordinates in {store_path.name}: "
                        f"lat={lat}, lon={lon}"
                    )
                    return metadata, False
            else:
                logger.warning(
                    f"Missing lat/lon in metadata for {store_path.name}"
                )
                return metadata, False

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                f"Failed to parse metadata for {store_path.name}: {e}"
            )
            return {}, False

    def load_store(self, store_path: Path) -> StoreData:
        """
        Load all data for a single store.
        
        Args:
            store_path: Path to the store directory.
        
        Returns:
            Populated StoreData instance.
        """
        store_id = store_path.name
        store = StoreData(store_id=store_id, store_path=store_path)

        # Load metadata
        metadata, is_valid = self._load_metadata(store_path)
        store.metadata = metadata
        store.has_metadata = is_valid

        if is_valid:
            store.latitude = float(metadata["latitude"])
            store.longitude = float(metadata["longitude"])
        else:
            # Fallback: Hyderabad city center
            store.latitude = 17.3850
            store.longitude = 78.4867
            logger.info(
                f"Using default coordinates for {store_id}: "
                f"Hyderabad center (17.3850, 78.4867)"
            )

        # Discover and classify images
        for ext in SUPPORTED_EXTENSIONS:
            for img_path in store_path.glob(f"*{ext}"):
                img_type = self._classify_image(img_path)

                if img_type == "shelf":
                    store.shelf_images.append(img_path)
                elif img_type == "counter":
                    store.counter_images.append(img_path)
                elif img_type == "outside":
                    store.outside_images.append(img_path)
                else:
                    store.extra_images.append(img_path)

                store.all_images.append(img_path)

        # Sort for deterministic processing
        store.shelf_images.sort()
        store.counter_images.sort()
        store.outside_images.sort()
        store.extra_images.sort()
        store.all_images.sort()

        logger.info(f"Loaded {store}")
        return store

    def load_all_stores(self) -> List[StoreData]:
        """
        Load data for all discovered stores.
        
        Returns:
            List of StoreData instances.
        """
        store_paths = self.discover_stores()
        stores = []

        for sp in store_paths:
            try:
                store = self.load_store(sp)
                if store.all_images:
                    stores.append(store)
                else:
                    logger.warning(f"Skipping {sp.name}: no images found")
            except Exception as e:
                logger.error(f"Failed to load {sp.name}: {e}")

        logger.info(f"Successfully loaded {len(stores)} stores")
        return stores
