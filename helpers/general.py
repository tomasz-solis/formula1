import os
import pandas as pd
from datetime import datetime, timedelta
import fastf1 as ff1
from .circuit_utils import update_profiles_file, build_circuit_profile_df


def is_update_needed(cache_path: str, season: int = datetime.now().year) -> bool:
    """
    Decide whether the cache file should be refreshed, based on whether
    we are approaching or in an upcoming race weekend.

    Parameters:
    - cache_path (str): Path to the cached CSV
    - season (int): Year to check the schedule for (defaults to current)

    Returns:
    - bool: True if update is needed
    """
    if not os.path.exists(cache_path):
        return True  # No cache file → needs update

    try:
        schedule = ff1.get_event_schedule(season, backend='ergast')
        now = datetime.utcnow()
        upcoming = schedule[schedule.Session1DateUtc > now]

        if upcoming.empty:
            return False  # All sessions complete — no need to rebuild

        next_race = upcoming.iloc[0]
        session1_utc = next_race['Session1DateUtc']
        weekend_start = session1_utc - timedelta(hours=6)  # FP1 prebuffer

        return now >= weekend_start

    except Exception as e:
        print(f"⚠️ Could not check race schedule: {e}")
        return True  # Conservative fallback


def load_or_build_profiles(cache_path="data/circuit_profiles.csv", start_year=2020, end_year=2025):
    """
    Load cached profiles if still valid; else append new race or rebuild from scratch.

    Parameters:
    - cache_path (str): Path to cached CSV file.
    - start_year (int): First season to include if rebuilding.
    - end_year (int): Last season to include.

    Returns:
    - df: DataFrame of circuit profiles.
    - skipped: Skipped sessions (or None if cache used)
    """
    # If cache missing → rebuild everything
    if not os.path.exists(cache_path):
        print("📂 No cache found. Rebuilding full dataset...")
        df, skipped = build_circuit_profile_df(start_year, end_year)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_csv(cache_path, index=False)
        return df, skipped

    # If new race weekend → update
    if is_update_needed(cache_path, season=end_year):
        print("🔁 Race weekend started — updating recent sessions only...")
        return update_profiles_file(cache_path)

    # Otherwise, use cached data
    print("✅ Using cached circuit profile file.")
    return pd.read_csv(cache_path), None

