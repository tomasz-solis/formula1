"""
Entrypoint for the F1 data pipeline:
  - Sets up FastF1 cache
  - Parses CLI arguments for seasons and GP filter
  - Invokes run_pipeline
"""

import argparse
import fastf1 as ff1
import os
import warnings
from datetime import datetime
from helpers.general_utils import load_or_build_profiles


# Suppress specific LapStartTime dtype warning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*Setting an item of incompatible dtype is deprecated.*"
)

cache_dir = "data/.fastf1_cache"
os.makedirs(cache_dir, exist_ok=True)
ff1.Cache.enable_cache(cache_dir)


def run_pipeline(from_year: int, to_year: int, gp_name: str | None = None):
    """Run the full F1 data‐processing pipeline over a range of seasons.

    Args:
        from_year (int): First season to process (inclusive).
        to_year   (int): Last season to process (inclusive).
        gp_name (str | None): If given, only build profiles for this specific Grand Prix.
    """
    print(f"🏁 Running pipeline from {from_year} to {to_year}")

    # Circuit profile files
    print(f"\n🛣️  Processing circuit profiles...")
    df_circuit, skipped_circuit = load_or_build_profiles(
        file_type="circuit",
        start_year=from_year,
        end_year=to_year,
        gp_name=args.gp_name
    )
    print(f"✅ Circuit profiles shape: {df_circuit.shape}")

    # Driver profile files
    print(f"\n🏎️  Processing driver profiles...")
    df_driver, skipped_driver = load_or_build_profiles(
        file_type="driver",
        start_year=from_year,
        end_year=to_year
    )
    print(f"✅ Driver profiles shape: {df_driver.shape}")

    print("\n🎉 Pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run F1 data pipeline for selected years"
    )
    parser.add_argument(
        "--from",
        dest="from_year",
        type=int,
        default=datetime.utcnow().year,
        help="Start year (default: current year)",
    )
    parser.add_argument(
        "--to",
        dest="to_year",
        type=int,
        default=datetime.utcnow().year,
        help="End year (default: current year)",
    )
    parser.add_argument(
        "--gp",
        dest="gp_name",
        type=str,
        default=None,
        help="Only build profiles for this exact Grand Prix name (optional)",
    )

    args = parser.parse_args()
    run_pipeline(args.from_year, args.to_year, args.gp_name)
