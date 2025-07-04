import argparse
from datetime import datetime
from helpers.general_utils import load_or_build_profiles
import os
import fastf1 as ff1
import warnings

# Suppress specific LapStartTime dtype warning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*Setting an item of incompatible dtype is deprecated.*"
)

cache_dir = "data/.fastf1_cache"
os.makedirs(cache_dir, exist_ok=True)
ff1.Cache.enable_cache(cache_dir)

def run_pipeline(from_year: int, to_year: int):
    print(f"🏁 Running pipeline from {from_year} to {to_year}")

    # Circuit profile files
    print(f"\n🛣️  Processing circuit profiles...")
    df_circuit, skipped_circuit = load_or_build_profiles(
        file_type="circuit",
        start_year=from_year,
        end_year=to_year
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
    parser = argparse.ArgumentParser(description="Run F1 data pipeline for selected years")
    parser.add_argument("--from", dest="from_year", type=int, default=datetime.utcnow().year,
                        help="Start year (default: current year)")
    parser.add_argument("--to", dest="to_year", type=int, default=datetime.utcnow().year,
                        help="End year (default: current year)")

    args = parser.parse_args()
    run_pipeline(args.from_year, args.to_year)
