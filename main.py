"""
VisionCred — Main Entry Point
================================
Run the complete credit assessment pipeline.

Usage:
    # Process all stores:
    python main.py

    # Process a specific store:
    python main.py --store store_1

    # Process multiple stores:
    python main.py --store store_1 store_3 store_5
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import VisionCredPipeline
from src.config import DATA_DIR, OUTPUT_DIR
from utils.logger import get_logger

logger = get_logger("main")


def print_banner():
    """Print the VisionCred startup banner."""
    banner = r"""
    +===========================================================+
    |                                                           |
    |   __     ___     _              ____              _       |
    |   \ \   / (_)___(_) ___  _ __  / ___|_ __ ___  __| |     |
    |    \ \ / /| / __| |/ _ \| '_ \| |   | '__/ _ \/ _` |     |
    |     \ V / | \__ \ | (_) | | | | |___| | |  __/ (_| |     |
    |      \_/  |_|___/_|\___/|_| |_|\____|_|  \___|\__,_|     |
    |                                                           |
    |   AI Credit Engine for Kirana Stores                      |
    |   Vision + Geo + Economics = Explainable Credit Scoring   |
    |                                                           |
    +===========================================================+
    """
    print(banner)


def main():
    """Main entry point for VisionCred pipeline."""
    parser = argparse.ArgumentParser(
        description="VisionCred: AI Credit Engine for Kirana Stores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Process all stores
  python main.py --store store_1    # Process single store
  python main.py --store store_1 store_3  # Process specific stores
        """,
    )
    parser.add_argument(
        "--store",
        nargs="+",
        help="Specific store folder name(s) to process (e.g., store_1 store_2)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"Override data directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Override output directory (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    print_banner()

    # Initialize pipeline
    pipeline = VisionCredPipeline()

    if args.store:
        # Process specific stores
        data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
        store_paths = []

        for store_name in args.store:
            store_path = data_dir / store_name
            if store_path.exists():
                store_paths.append(store_path)
            else:
                logger.error(
                    f"Store directory not found: {store_path}"
                )

        if store_paths:
            results = pipeline.process_batch(store_paths=store_paths)
        else:
            logger.error("No valid store directories found!")
            sys.exit(1)
    else:
        # Process all stores
        results = pipeline.process_batch()

    # Print final summary
    print("\n")
    logger.info("=" * 60)
    logger.info("  PROCESSING COMPLETE")
    logger.info(f"  Stores processed: {len(results)}")
    logger.info(f"  Results saved to: {OUTPUT_DIR}")
    logger.info("=" * 60)

    # Print condensed results to console
    for r in results:
        if "error" not in r:
            print(f"\n[*] {r['store_id']}:")
            print(
                f"   Daily Sales:    Rs.{r['daily_sales_range'][0]:,.0f}"
                f" - Rs.{r['daily_sales_range'][1]:,.0f}"
            )
            print(
                f"   Monthly Rev:    Rs.{r['monthly_revenue_range'][0]:,.0f}"
                f" - Rs.{r['monthly_revenue_range'][1]:,.0f}"
            )
            print(
                f"   Monthly Income: Rs.{r['monthly_income_range'][0]:,.0f}"
                f" - Rs.{r['monthly_income_range'][1]:,.0f}"
            )
            print(f"   Confidence:     {r['confidence_score']:.2f}")
            print(f"   Risk Level:     {r['overall_risk_level']}")


if __name__ == "__main__":
    main()
