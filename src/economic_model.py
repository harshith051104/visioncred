"""
VisionCred Economic Model
===========================
Transparent, formula-based estimation of kirana store cash flow.

NO BLACK BOX — every calculation is explainable:

    Daily Sales = (Inventory Value × Turnover Rate)
                  × Location Multiplier
                  × Demand Factor

    Monthly Revenue = Daily Sales × 30
    Monthly Income  = Monthly Revenue × Margin (10–20%)

All outputs are RANGES (min, max) to reflect uncertainty.
"""

from typing import Dict, List, Tuple

from src.config import (
    TURNOVER_RATE_LOW,
    TURNOVER_RATE_HIGH,
    DEMAND_FACTOR_LOW,
    DEMAND_FACTOR_HIGH,
    MARGIN_LOW,
    MARGIN_HIGH,
    DAYS_IN_MONTH,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class EconomicModel:
    """
    Computes cash flow estimates using transparent economic formulas.
    
    The model uses a layered multiplication approach:
    
    Layer 1 — Inventory Base:
        Start with estimated visible inventory value (from vision module).
    
    Layer 2 — Turnover:
        Apply daily turnover rate (5–15% of inventory sells daily).
    
    Layer 3 — Location:
        Multiply by location factor (urban stores sell more).
    
    Layer 4 — Demand:
        Apply demand seasonality factor (0.8–1.2×).
    
    Layer 5 — Margin:
        Apply profit margin (10–20%) for income estimation.
    """

    def __init__(self):
        logger.info("EconomicModel initialized")

    def estimate_daily_sales(self, features: Dict) -> Tuple[float, float]:
        """
        Estimate daily sales range.
        
        Formula:
            Daily Sales = Inventory Value × Turnover Rate 
                         × Location Multiplier × Demand Factor
        
        Args:
            features: Consolidated feature dictionary.
        
        Returns:
            Tuple of (daily_sales_min, daily_sales_max) in INR.
        """
        inv_low = features.get("inventory_value_low", 0)
        inv_high = features.get("inventory_value_high", 0)
        loc_mult = features.get("location_multiplier", 1.0)
        footfall = features.get("geo_footfall_score", 0.5)

        # Adjust demand factor based on footfall
        # Higher footfall → demand closer to high end
        demand_low = DEMAND_FACTOR_LOW + (footfall * 0.1)
        demand_high = DEMAND_FACTOR_HIGH + (footfall * 0.1)

        # Conservative estimate: low inventory × low turnover × low demand
        daily_min = (
            inv_low * TURNOVER_RATE_LOW * loc_mult * demand_low
        )

        # Optimistic estimate: high inventory × high turnover × high demand
        daily_max = (
            inv_high * TURNOVER_RATE_HIGH * loc_mult * demand_high
        )

        return round(daily_min, 2), round(daily_max, 2)

    def estimate_monthly_revenue(
        self, daily_sales_range: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Compute monthly revenue range from daily sales.
        
        Formula:
            Monthly Revenue = Daily Sales × 30 days
        
        Args:
            daily_sales_range: (min, max) daily sales in INR.
        
        Returns:
            Tuple of (monthly_revenue_min, monthly_revenue_max) in INR.
        """
        monthly_min = daily_sales_range[0] * DAYS_IN_MONTH
        monthly_max = daily_sales_range[1] * DAYS_IN_MONTH
        return round(monthly_min, 2), round(monthly_max, 2)

    def estimate_monthly_income(
        self, monthly_revenue_range: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Compute monthly income (profit) range.
        
        Formula:
            Monthly Income = Monthly Revenue × Margin (10–20%)
        
        Args:
            monthly_revenue_range: (min, max) monthly revenue in INR.
        
        Returns:
            Tuple of (monthly_income_min, monthly_income_max) in INR.
        """
        income_min = monthly_revenue_range[0] * MARGIN_LOW
        income_max = monthly_revenue_range[1] * MARGIN_HIGH
        return round(income_min, 2), round(income_max, 2)

    def identify_key_drivers(self, features: Dict) -> List[str]:
        """
        Identify the top factors driving the economic estimate.
        
        Returns a list of human-readable driver explanations,
        sorted by impact.
        """
        drivers = []

        # Inventory level
        inv_avg = (
            features.get("inventory_value_low", 0)
            + features.get("inventory_value_high", 0)
        ) / 2

        if inv_avg > 5000:
            drivers.append(
                f"High inventory level (Rs.{inv_avg:,.0f} estimated visible stock)"
            )
        elif inv_avg > 1000:
            drivers.append(
                f"Moderate inventory level (Rs.{inv_avg:,.0f} estimated visible stock)"
            )
        else:
            drivers.append(
                f"Low inventory level (Rs.{inv_avg:,.0f} estimated visible stock)"
            )

        # Shelf density impact
        density = features.get("shelf_density_index", 0)
        if density > 0.6:
            drivers.append(
                f"Well-stocked shelves (density index: {density:.2f})"
            )
        elif density > 0.3:
            drivers.append(
                f"Moderately stocked shelves (density index: {density:.2f})"
            )
        else:
            drivers.append(
                f"Under-stocked shelves (density index: {density:.2f})"
            )

        # Location impact
        loc_type = features.get("location_type", "semi_urban")
        loc_mult = features.get("location_multiplier", 1.0)
        drivers.append(
            f"Location type: {loc_type} (multiplier: {loc_mult:.1f}x)"
        )

        # Footfall
        footfall = features.get("geo_footfall_score", 0.5)
        if footfall > 0.7:
            drivers.append(f"High footfall area (score: {footfall:.2f})")
        elif footfall > 0.4:
            drivers.append(f"Moderate footfall area (score: {footfall:.2f})")
        else:
            drivers.append(f"Low footfall area (score: {footfall:.2f})")

        # SKU variety
        diversity = features.get("sku_diversity_score", 0)
        if diversity > 0.5:
            drivers.append(
                f"Good product variety ({features.get('num_unique_classes', 0)} categories)"
            )
        else:
            drivers.append(
                f"Limited product variety ({features.get('num_unique_classes', 0)} categories)"
            )

        return drivers

    def compute(self, features: Dict) -> Dict:
        """
        Run the complete economic model.
        
        Args:
            features: Consolidated feature dictionary from FeatureEngineer.
        
        Returns:
            Dict with all economic estimates, ranges, and explanations.
        """
        # Step 1: Daily sales range
        daily_min, daily_max = self.estimate_daily_sales(features)

        # Step 2: Monthly revenue range
        monthly_rev_min, monthly_rev_max = self.estimate_monthly_revenue(
            (daily_min, daily_max)
        )

        # Step 3: Monthly income range
        monthly_inc_min, monthly_inc_max = self.estimate_monthly_income(
            (monthly_rev_min, monthly_rev_max)
        )

        # Step 4: Key drivers
        key_drivers = self.identify_key_drivers(features)

        economic_output = {
            "daily_sales_range": [daily_min, daily_max],
            "monthly_revenue_range": [monthly_rev_min, monthly_rev_max],
            "monthly_income_range": [monthly_inc_min, monthly_inc_max],
            "key_drivers": key_drivers,
            "model_parameters": {
                "turnover_rate_range": [TURNOVER_RATE_LOW, TURNOVER_RATE_HIGH],
                "demand_factor_range": [DEMAND_FACTOR_LOW, DEMAND_FACTOR_HIGH],
                "margin_range": [MARGIN_LOW, MARGIN_HIGH],
                "location_multiplier": features.get("location_multiplier", 1.0),
                "days_in_month": DAYS_IN_MONTH,
            },
            "formula": (
                "Daily Sales = Inventory Value x Turnover Rate "
                "x Location Multiplier x Demand Factor"
            ),
        }

        logger.info(
            f"Economic model output | "
            f"daily=Rs.{daily_min:,.0f}-{daily_max:,.0f}, "
            f"monthly_rev=Rs.{monthly_rev_min:,.0f}-{monthly_rev_max:,.0f}, "
            f"monthly_inc=Rs.{monthly_inc_min:,.0f}-{monthly_inc_max:,.0f}"
        )

        return economic_output
