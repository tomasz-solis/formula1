"""
Entrypoint for the F1 data pipeline:
  - Sets up FastF1 cache
  - Parses CLI arguments for seasons and GP filter
  - Invokes run_pipeline to build profiles
"""

import argparse
import fastf1 as ff1
import os
import warnings
from datetime import datetime, timezone
from helpers.general_utils import load_or_build_profiles
from helpers.prediction import export_completed_classifications_csv

# Suppress deprecated dtype warnings when setting LapStartTime
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*Setting an item of incompatible dtype is deprecated.*"
)

# Configure FastF1 cache directory
cache_dir = "data/.fastf1_cache"
os.makedirs(cache_dir, exist_ok=True)
ff1.Cache.enable_cache(cache_dir)


def run_pipeline(from_year: int, to_year: int, gp_name: str | None = None) -> None:
    """
    Run the full F1 data-processing pipeline over a range of seasons.

    Steps:
      1. Build or update circuit profiles
      2. Build or update driver profiles
      3. Build or update driver timing profiles

    Args:
        from_year: First season to process (inclusive).
        to_year: Last season to process (inclusive).
        gp_name: If specified, only build circuit profiles for this Grand Prix.
    """
    print(f"🏁 Running pipeline from {from_year} to {to_year}")

    # 1) Circuit profiles
    print("\n🛣️  Processing circuit profiles...")
    df_circuit, skipped_circuit = load_or_build_profiles(
        file_type="circuit",
        start_year=from_year,
        end_year=to_year,
        gp_name=gp_name
    )
    print(f"✅ Circuit profiles shape: {df_circuit.shape}")

    # 2) Driver profiles
    print("\n🏎️  Processing driver profiles...")
    df_driver, skipped_driver = load_or_build_profiles(
        file_type="driver",
        start_year=from_year,
        end_year=to_year
    )
    print(f"✅ Driver profiles shape: {df_driver.shape}")

    # 3) Driver timing profiles
    print("\n⏱️  Processing driver timing profiles...")
    df_timing, skipped_timing = load_or_build_profiles(
        file_type="driver_timing",
        start_year=from_year,
        end_year=to_year
    )
    print(f"✅ Driver timing profiles shape: {df_timing.shape}")

    # 4) Export available classifications if session results are ready
    print("\n📤 Exporting available classifications...")

    res = export_completed_classifications_csv(
        season=to_year,
        include_sprint=True,   # set False if to skip sprints
        overwrite=False        # set True to regenerate the CSVs
    )

    for sess_type, r in res.items():
        where = f" ({r.written_path})" if r.written_path else ""
        print(f"  {sess_type:18s} → {r.status}{where if where else ''}")


    print("\n🎉 Pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run F1 data pipeline for selected years"
    )
    parser.add_argument(
        "--from", dest="from_year", type=int,
        default=datetime.now(timezone.utc).year,
        help="Start year (inclusive)."
    )
    parser.add_argument(
        "--to", dest="to_year", type=int,
        default=datetime.now(timezone.utc).year,
        help="End year (inclusive)."
    )
    parser.add_argument(
        "--gp", dest="gp_name", type=str, default=None,
        help="Optional Grand Prix name to filter circuit profiles."
    )

    args = parser.parse_args()
    # Invoke pipeline with parsed arguments
    run_pipeline(
        from_year=args.from_year,
        to_year=args.to_year,
        gp_name=args.gp_name
    )