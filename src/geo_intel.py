"""
VisionCred Geo Intelligence Module
====================================
Generates location-based features from GPS coordinates (metadata.json).

Uses deterministic, rule-based logic — NOT black-box ML:
    - Distance from known metro centers → location_type
    - Population density proxy → footfall_score
    - Proximity clustering → competition_density

All scores are on a 0–1 scale for consistency.
"""

import math
from typing import Dict, Optional, Tuple

from src.config import (
    METRO_CENTERS,
    URBAN_RADIUS_KM,
    SEMI_URBAN_RADIUS_KM,
    LOCATION_MULTIPLIERS,
    FOOTFALL_BASE,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class GeoAnalyzer:
    """
    Generates geo-based intelligence features from store GPS coordinates.
    
    Logic:
        1. **Location Type**: Classify as urban/semi-urban/rural based on 
           Haversine distance to nearest known metro center.
        
        2. **Footfall Score**: Higher near metro centers, modulated by 
           proximity to commercial hubs. Range: 0–1.
        
        3. **Competition Density**: Inversely related to distance from 
           city centers (more central = more competition). Range: 0–1.
        
        4. **Location Multiplier**: Economic multiplier based on location 
           type, used in the revenue model.
    """

    def __init__(self):
        self.metro_centers = METRO_CENTERS
        logger.info(
            f"GeoAnalyzer initialized with {len(self.metro_centers)} "
            f"metro reference points"
        )

    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate the great-circle distance between two points
        on Earth using the Haversine formula.
        
        Args:
            lat1, lon1: Coordinates of point 1 (degrees)
            lat2, lon2: Coordinates of point 2 (degrees)
        
        Returns:
            Distance in kilometers.
        """
        R = 6371.0  # Earth's radius in km

        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _find_nearest_metro(
        self, lat: float, lon: float
    ) -> Tuple[str, float]:
        """
        Find the nearest metro center and its distance.
        
        Returns:
            Tuple of (metro_name, distance_km)
        """
        nearest = None
        min_dist = float("inf")

        for name, (m_lat, m_lon) in self.metro_centers.items():
            dist = self.haversine_distance(lat, lon, m_lat, m_lon)
            if dist < min_dist:
                min_dist = dist
                nearest = name

        return nearest, min_dist

    def classify_location(self, lat: float, lon: float) -> str:
        """
        Classify store location as urban, semi_urban, or rural
        based on distance to nearest metro center.
        
        Rules:
            - < 10 km from any metro center → urban
            - 10–25 km → semi_urban
            - > 25 km → rural
        
        Returns:
            'urban', 'semi_urban', or 'rural'
        """
        _, min_dist = self._find_nearest_metro(lat, lon)

        if min_dist <= URBAN_RADIUS_KM:
            return "urban"
        elif min_dist <= SEMI_URBAN_RADIUS_KM:
            return "semi_urban"
        else:
            return "rural"

    def compute_footfall_score(
        self, lat: float, lon: float, location_type: str
    ) -> float:
        """
        Estimate footfall score (0–1) based on proximity to metro centers.
        
        Logic:
            - Start with a base score from location type
            - Apply a proximity bonus: closer to center → higher footfall
            - The bonus decays exponentially with distance
        
        Returns:
            float: Footfall score between 0.0 and 1.0
        """
        base = FOOTFALL_BASE.get(location_type, 0.3)
        _, min_dist = self._find_nearest_metro(lat, lon)

        # Exponential decay: bonus is highest when distance = 0
        # Decay constant tuned so bonus ≈ 0 at 30km
        proximity_bonus = 0.2 * math.exp(-min_dist / 10.0)

        score = min(base + proximity_bonus, 1.0)
        return round(score, 4)

    def compute_competition_density(
        self, lat: float, lon: float
    ) -> float:
        """
        Estimate competition density (0–1).
        
        Logic:
            - Count how many metro centers are within 15km
            - More nearby centers = higher commercial area = more competition
            - Normalize by total centers count
        
        Returns:
            float: Competition density between 0.0 and 1.0
        """
        nearby_count = 0
        weighted_score = 0.0

        for name, (m_lat, m_lon) in self.metro_centers.items():
            dist = self.haversine_distance(lat, lon, m_lat, m_lon)
            if dist <= 15.0:
                nearby_count += 1
                # Closer centers contribute more to competition
                weighted_score += max(0, 1.0 - (dist / 15.0))

        # Normalize by number of reference points
        max_possible = len(self.metro_centers)
        density = weighted_score / max_possible

        return round(min(density, 1.0), 4)

    def analyze(
        self,
        latitude: float,
        longitude: float,
        has_metadata: bool = True,
    ) -> Dict:
        """
        Generate all geo-intelligence features for a store.
        
        Args:
            latitude: Store latitude (from metadata.json or default)
            longitude: Store longitude (from metadata.json or default)
            has_metadata: Whether real GPS data was available
        
        Returns:
            Dict with all geo features and explanations.
        """
        logger.info(
            f"Geo analysis | lat={latitude:.4f}, lon={longitude:.4f}, "
            f"has_gps={'Yes' if has_metadata else 'No (default)'}"
        )

        # Classify location
        location_type = self.classify_location(latitude, longitude)
        nearest_metro, distance_km = self._find_nearest_metro(
            latitude, longitude
        )

        # Compute scores
        footfall_score = self.compute_footfall_score(
            latitude, longitude, location_type
        )
        competition_density = self.compute_competition_density(
            latitude, longitude
        )
        location_multiplier = LOCATION_MULTIPLIERS.get(location_type, 1.0)

        geo_features = {
            "location_type": location_type,
            "nearest_metro_center": nearest_metro,
            "distance_to_nearest_km": round(distance_km, 2),
            "footfall_score": footfall_score,
            "competition_density": competition_density,
            "location_multiplier": location_multiplier,
            "gps_source": "metadata" if has_metadata else "default_fallback",
            "coordinates": {
                "latitude": latitude,
                "longitude": longitude,
            },
        }

        logger.info(
            f"Geo result | type={location_type}, "
            f"nearest={nearest_metro} ({distance_km:.1f}km), "
            f"footfall={footfall_score:.3f}, "
            f"competition={competition_density:.3f}"
        )

        return geo_features
