"""
Entrypoint for the F1 data pipeline:
  - Sets up FastF1 cache
  - Parses CLI arguments for seasons and GP filter
  - Invokes run_pipeline to build profiles and features
"""

import argparse
import logging
import warnings

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
# Suppress noisy warnings (cache is working, just being verbose)
logging.getLogger('requests_cache.session').setLevel(logging.ERROR)
logging.getLogger('ergast_py').setLevel(logging.ERROR)

# Suppress deprecated dtype warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*Setting an item of incompatible dtype is deprecated.*"
)

# ============================================================================
# MAIN IMPORTS
# ============================================================================
import fastf1 as ff1
import os
import pandas as pd
from datetime import datetime, timezone
from helpers.general_utils import load_or_build_profiles
from helpers.prediction import export_completed_classifications_csv_range
from helpers.historical_features import compute_historical_features
from helpers.auto_retrain import auto_retrain_if_needed

# Configure FastF1 cache directory
cache_dir = "data/.fastf1_cache"
os.makedirs(cache_dir, exist_ok=True)
ff1.Cache.enable_cache(cache_dir)


def run_pipeline(
    from_year: int,
    to_year: int,
    gp_name: str | None = None,
    build_features: bool = True
) -> None:
    """
    Run the full F1 data-processing pipeline over a range of seasons.

    Steps:
      1. Build or update circuit profiles
      2. Build or update driver profiles
      3. Build or update driver timing profiles
      4. Export classifications (append-only, all seasons)
      5. Compute historical features for ML (optional)

    Args:
        from_year: First season to process (inclusive).
        to_year: Last season to process (inclusive).
        gp_name: If specified, only build circuit profiles for this Grand Prix.
        build_features: If True, compute historical features after profiles (default: True)
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

    # 4) Export classifications (append-only for ALL seasons)
    print("\n📤 Exporting classifications (append-only, all seasons)...\n")
    res_by_season = export_completed_classifications_csv_range(
        start_year=from_year,
        end_year=to_year,
        include_sprint=True,
        up_to_utc=None,
    )

    for season, res in sorted(res_by_season.items()):
        print(f"\n  📅 Season {season}")
        for sess_type, r in res.items():
            where = f" ({r.written_path})" if r.written_path else ""
            
            # Status icons for readability
            if r.status == "appended":
                status_icon = "✅"
            elif r.status == "created":
                status_icon = "📝"
            elif r.status == "skipped":
                status_icon = "ℹ️ "
            elif r.status == "error":
                status_icon = "❌"
                # Show error reason
                reason = f" - {r.reason}" if r.reason else ""
                where = reason
            else:
                status_icon = "  "
                
            print(f"    {status_icon} {sess_type:18s} → {r.status}{where}")

    # 5) Compute historical features (optional)
    if build_features:
        print("\n🔮 Computing historical features for ML...")
        
        try:
            features_df = compute_historical_features(
                driver_profiles=df_driver,
                circuit_profiles=df_circuit,
                lookback_years=3,
                form_window=5,
                rain_threshold=0.1,
                start_year=from_year,  
                end_year=to_year 
            )
            
            # 🔧 FIX: Convert datetime columns to strings for Parquet compatibility
            print("🔧 Converting datetime columns for Parquet compatibility...")
            
            # Get datetime64 columns
            datetime_cols = features_df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Check object columns for Timestamp objects
            for col in features_df.select_dtypes(include=['object']).columns:
                if len(features_df) > 0:
                    # Check first non-null value
                    non_null_values = features_df[col].dropna()
                    if len(non_null_values) > 0:
                        first_val = non_null_values.iloc[0]
                        if isinstance(first_val, (pd.Timestamp, pd.Timedelta)):
                            datetime_cols.append(col)
            
            # Convert to strings
            for col in datetime_cols:
                features_df[col] = features_df[col].astype(str)
                print(f"   ✅ Converted {col} to string")
            
            # Save to cache
            features_dir = "data/features"
            os.makedirs(features_dir, exist_ok=True)
            
            features_file = os.path.join(
                features_dir,
                "ml_features.parquet"
            )
            
            features_df.to_parquet(
                features_file,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            print(f"✅ Historical features shape: {features_df.shape}")
            print(f"💾 Saved to: {features_file}")
            
            # Print feature summary
            print("\n📊 Feature Summary:")
            historical_cols = [
                'circuit_avg_position', 'circuit_best_position',
                'recent_avg_position', 'form_trend',
                'wet_dry_delta', 'team_circuit_avg_position',
                'team_momentum'
            ]
            
            available_features = [col for col in historical_cols if col in features_df.columns]
            
            print(f"   Total columns: {len(features_df.columns)}")
            print(f"   Historical features: {len(available_features)}")
            print(f"   Sample size: {len(features_df):,} driver-sessions")
            
            # Show missingness
            missing_pct = features_df[available_features].isnull().mean() * 100
            print("\n   Missing data by feature:")
            for feat, pct in missing_pct.items():
                if pct > 0:
                    print(f"     {feat:30s}: {pct:5.1f}%")
            
        except Exception as e:
            print(f"⚠️  Failed to compute historical features: {e}")
            print("   Continuing without features...")
            # Print full traceback for debugging
            import traceback
            traceback.print_exc()

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
    parser.add_argument(
        "--no-features", dest="build_features", action="store_false",
        help="Skip historical feature computation (faster for data-only runs)."
    )

    args = parser.parse_args()
    run_pipeline(
        from_year=args.from_year,
        to_year=args.to_year,
        gp_name=args.gp_name,
        build_features=args.build_features
    )

    # AUTO-RETRAIN: Check for new data and retrain if needed
    print("\n" + "="*80)
    print("🤖 CHECKING FOR AUTO-RETRAINING...")
    print("="*80)
    
    result = auto_retrain_if_needed(
        features_file="data/features/ml_features.parquet"
    )
    
    if result['status'] == 'deployed':
        print("\n✅ NEW MODEL VERSION DEPLOYED!")
        print(f"   Version: v{result['version']}")
        print(f"   Improvements:")
        for model, improvement in result['comparison']['improvements'].items():
            print(f"      {model}: {improvement:+.2%}")
    
    elif result['status'] == 'skipped':
        print(f"\n⏭️  {result['reason']}")
    
    print("="*80)