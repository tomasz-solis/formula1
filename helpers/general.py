import os
import pandas as pd
from datetime import datetime, timedelta
import fastf1 as ff1

def is_update_needed(cache_path: str, season: int = datetime.now().year) -> bool:
    """
    Decide whether the cache file should be refreshed, based on the next race weekend.
    """
    if not os.path.exists(cache_path):
        return True  # No file → definitely update

    try:
        backend = 'ergast' if session_name not in ['SQ'] else 'f1api'
        schedule = ff1.get_event_schedule(year, backend=backend)
        now = datetime.utcnow()
        upcoming = schedule[schedule.Session1DateUtc > now]

        if upcoming.empty:
            return False  # All races are over

        next_race = upcoming.iloc[0]
        session1_utc = next_race['Session1DateUtc']
        weekend_start = session1_utc - timedelta(hours=6)  # allow buffer before FP1

        # Only reload if we're already in race weekend territory
        return now >= weekend_start

    except Exception as e:
        print(f"⚠️ Could not check race schedule: {e}")
        return True  # Be safe: fallback to updating

def load_or_build_profiles(cache_path="circuit_profiles.csv", start_year=2020, end_year=2025):
    """
    Load cached profiles if they're still valid; otherwise rebuild them.
    """
    if is_update_needed(cache_path, end_year):
        print("🔁 Rebuilding circuit profiles...")
        from helpers import build_circuit_profile_df
        df, skipped = build_circuit_profile_df(start_year, end_year)
        df.to_csv(cache_path, index=False)
        return df, skipped
    else:
        print("✅ Using cached circuit profile file.")
        return pd.read_csv(cache_path), None
