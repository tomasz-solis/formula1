import fastf1 as ff1
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from fastf1.ergast import Ergast
import requests
from functools import lru_cache
import logging
from tqdm import tqdm 
from IPython.display import display
from pathlib import Path
import os


# Set pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Suppress FastF1 INFO and DEBUG messages
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


# --- Session loading and metadata ---

def load_session(year, event_name, session_name):
    """
    Load a session using FastF1. Fallback to OpenF1 if FastF1 fails.

    Parameters:
    - year: int — F1 season year
    - event_name: str — Grand Prix name
    - session_name: str — 'FP1', 'FP2', 'Q', 'Race', etc.

    Returns:
    - dict with keys: source, session, laps, status, reason
    """
    try:
        session = ff1.get_session(year, event_name, session_name)
        session.load(telemetry=True, laps=True)
        if session.laps.empty:
            raise ValueError("FastF1 session loaded but contains no lap data")
        return {
            "source": "fastf1",
            "session": session,
            "laps": session.laps,
            "status": "ok",
            "reason": None
        }
    except Exception as e:
        print(f"⚠️ FastF1 failed for {year} {event_name} {session_name}: {e}")

    # Fallback to OpenF1 API
    try:
        print("🔄 Falling back to OpenF1 API...")
        session_map = {
            "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3",
            "Q": "Qualifying", "Race": "Race", "SQ": "Sprint Qualifying",
            "S": "Sprint", "SS": "Sprint Shootout"
        }
        api_session_name = session_map.get(session_name, session_name)
        url = "https://api.openf1.org/v1/lap_times"
        params = {"year": year, "session": api_session_name}
        response = requests.get(url, params=params)
        response.raise_for_status()
        laps_df = pd.DataFrame(response.json())
        if laps_df.empty:
            raise ValueError("OpenF1 returned an empty dataset")
        return {
            "source": "openf1",
            "session": None,
            "laps": laps_df,
            "status": "fallback",
            "reason": None
        }
    except Exception as e:
        print(f"❌ OpenF1 fallback failed for {year} {event_name} {session_name}: {e}")
        return {
            "source": None,
            "session": None,
            "laps": None,
            "status": "error",
            "reason": str(e)
        }

# WEATHER

def get_weather_info(session, year, event_name, session_name):
    """
    Try to extract weather data from FastF1 session. Fallback to OpenF1 API if needed.

    Parameters:
    - session (FastF1.Session or None): Loaded FastF1 session, or None if fallback needed.
    - year (int): Season year
    - event_name (str): Event name (e.g., 'Bahrain Grand Prix')
    - session_name (str): Session label ('FP1', 'Race', etc.)

    Returns:
    - dict with average air temp, track temp, and boolean rain presence
    """
    # --- Try FastF1 session ---
    if session is not None and hasattr(session, "weather_data") and not session.weather_data.empty:
        weather = session.weather_data
        return {
            "air_temp_avg": weather["AirTemp"].mean(),
            "track_temp_avg": weather["TrackTemp"].mean(),
            "rain_detected": weather["Rainfall"].max() > 0
        }

    # --- Fallback to OpenF1 ---
    try:
        session_map = {
            "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3",
            "Q": "Qualifying", "Race": "Race", "SQ": "Sprint Qualifying",
            "S": "Sprint", "SS": "Sprint Shootout"
        }
        api_session_name = session_map.get(session_name, session_name)

        url = "https://api.openf1.org/v1/weather"
        params = {"year": year, "session": api_session_name}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = pd.DataFrame(response.json())

        if data.empty:
            raise ValueError("No weather data from OpenF1")

        return {
            "air_temp_avg": data["air_temperature"].mean(),
            "track_temp_avg": data["track_temperature"].mean(),
            "rain_detected": (data["rainfall"] > 0).any()
        }

    except Exception as e:
        print(f"⚠️ Weather fallback failed for {year} {event_name} {session_name}: {e}")
        return {
            "air_temp_avg": np.nan,
            "track_temp_avg": np.nan,
            "rain_detected": np.nan
        }
        
# ELEVATION LOOKUP 

@lru_cache(maxsize=None)
def get_elevation(latitude: float, longitude: float,
                  timeout: int = 10) -> float:
    """
    Single-point elevation (Copernicus GLO-90 DEM, 90 m resolution).

    Keeps the original per-point interface but hits Open-Meteo, whose
    public limit is 600 calls/minute instead of 1 call/second.
    """
    url = (
        "https://api.open-meteo.com/v1/elevation"
        f"?latitude={latitude}&longitude={longitude}"
    )

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()                     # network / HTTP errors → exceptions

    payload = r.json()
    if "elevation" not in payload or payload["elevation"] is None:
        raise RuntimeError("Open-Meteo returned no elevation")

    return payload["elevation"][0]           # note: array, not dict

    
# --- CIRCUIT METADATA ---

def get_circuits(season):
    """
    Get main geolocation info of the tracks:
    - latitude
    - longitude
    - official circuit name
    - altitude above sea level

    Parameters:
    - season: int indicating the year/season of Formula 1

    Returns:
    - DataFrame with circuitName, location, latitude, longitude, and altitude
    """
    ergast = Ergast()
    racetracks = ergast.get_circuits(season)

    results = []

    for racetrack in racetracks.circuitName:
        try:
            row = racetracks[racetracks.circuitName == racetrack].iloc[0]
            circuit_name = row['circuitName']
            latitude = row['lat']
            longitude = row['long']
            locality = row['locality']
            country = row['country']

            altitude = get_elevation(latitude, longitude)

            results.append({
                'circuitName': racetrack,
                'location': locality,
                'country': country,
                'lat': latitude,
                'lon': longitude,
                'altitude': altitude
            })

        except Exception as e:
            print(f"⚠️ Failed to get altitude for {racetrack}: {e}")
            continue

    return pd.DataFrame(results)


def get_all_circuits(start_year=2020, end_year=2025):
    all_rows = []
    for year in tqdm(range(start_year, end_year + 1), desc="Processing seasons"):
        df = get_circuits(year)
        all_rows.append(df)
    
    full_df = pd.concat(all_rows, ignore_index=True)
    
    # Keep only the first unique occurrence per circuit
    deduped = full_df.drop_duplicates(subset=['circuitName'], keep='first').reset_index(drop=True)
    
    return deduped

# --- Telemetry extraction ---

def extract_track_metrics(session):
    """
    Extract average speed, top speed, and braking profile from a loaded session.

    Parameters:
    - session (FastF1.Session): Loaded FastF1 session

    Returns:
    - dict: {
        avg_speed, top_speed, braking_events,
        low_pct, med_pct, high_pct
      } or None if extraction failed
    """
    try:
        if session.laps.empty:
            return None

        lap = session.laps.pick_fastest()
        telemetry = lap.get_car_data().add_distance()
        telemetry['delta_speed'] = telemetry['Speed'].diff()
        heavy_brakes = telemetry['delta_speed'] < -30
        braking_events = heavy_brakes.sum()

        return {
            'avg_speed': telemetry['Speed'].mean(),
            'top_speed': telemetry['Speed'].max(),
            'braking_events': braking_events,
            'low_pct': (telemetry.Speed < 120).mean(),
            'med_pct': ((telemetry.Speed >= 120) & (telemetry.Speed < 200)).mean(),
            'high_pct': (telemetry.Speed >= 200).mean()
        }

    except Exception as e:
        print(f"⚠️ Failed to extract metrics: {e}")
        return None


def get_circuit_corner_profile(session, low_thresh=100, med_thresh=170):
    """
    Detect corners and categorize them by entry speed using local speed minima.

    Parameters:
    - session (FastF1.Session): Loaded session
    - low_thresh (int): max speed for slow corners (km/h)
    - med_thresh (int): max speed for medium corners (km/h)

    Returns:
    - dict: {
        slow_corners, medium_corners, fast_corners, chicanes
      } or None if failed
    """
    try:
        lap = session.laps.pick_fastest()
        tel = lap.get_car_data().add_distance()
        tel['prev_speed'] = tel['Speed'].shift(1)
        tel['next_speed'] = tel['Speed'].shift(-1)
        tel['is_corner'] = (tel['Speed'] < tel['prev_speed']) & (tel['Speed'] < tel['next_speed'])
        corners = tel[tel['is_corner']].copy()

        corners['corner_type'] = pd.cut(
            corners['Speed'],
            bins=[0, low_thresh, med_thresh, 400],
            labels=['slow', 'medium', 'fast']
        )
        counts = corners['corner_type'].value_counts().to_dict()

        corners['DistanceFromPrev'] = corners['Distance'].diff().fillna(9999)
        chicanes = (corners['DistanceFromPrev'] < 200).sum()

        return {
            'slow_corners': counts.get('slow', 0),
            'medium_corners': counts.get('medium', 0),
            'fast_corners': counts.get('fast', 0),
            'chicanes': chicanes
        }

    except Exception as e:
        print(f"⚠️ Corner profile failed: {e}")
        return None


def get_drs_info(session, track_length, event=None, session_name=None):
    """
    Estimate DRS info from telemetry (fastest lap).

    Parameters:
    - session (FastF1.Session): Loaded session
    - track_length (float): Estimated lap length (m)
    - event (str): Event name for error reporting (e.g., 'Bahrain Grand Prix')
    - session_name (str): Session label (e.g., 'FP1', 'Q')

    Returns:
    - dict: num_drs_zones, drs_total_len_m, drs_pct_of_lap
    """
    try:
        lap = session.laps.pick_fastest()
        tel = lap.get_car_data().add_distance()

        if 'DRS' not in tel.columns:
            raise ValueError(f"{event} {session_name} - DRS channel not available in telemetry")

        # Identify rows where DRS is active (1- activated, 8 available)
        drs_active = tel[tel['DRS'].isin([1, 8])].copy()
        if drs_active.empty:
            raise ValueError(f"{event} {session_name} - No DRS usage detected in lap")

        # Tag separate DRS zones (gaps > 100m in distance)
        drs_active['gap'] = drs_active['Distance'].diff().fillna(0)
        drs_active['zone_id'] = (drs_active['gap'] > 100).cumsum()

        # Compute total length and count zones
        zone_lengths = drs_active.groupby('zone_id')['Distance'].agg(['min', 'max'])
        zone_lengths['length'] = zone_lengths['max'] - zone_lengths['min']

        num_drs_zones = len(zone_lengths)
        drs_total_len = zone_lengths['length'].sum()
        drs_pct = drs_total_len / track_length if track_length else np.nan

        return {
            'num_drs_zones': num_drs_zones,
            'drs_total_len_m': drs_total_len,
            'drs_pct_of_lap': drs_pct
        }

    except Exception as e:
        print(f"⚠️ {event} {session_name} - Failed to infer DRS zones: {e}")
        return {
            'num_drs_zones': 0,
            'drs_total_len_m': 0,
            'drs_pct_of_lap': np.nan
        }


# --- Main builder ---

def build_profiles_for_season(year, circuit_metadata):
    """
    Build circuit session profiles for a single season (year).

    Parameters:
    - year (int): Season year
    - circuit_metadata (DataFrame): Preloaded metadata (altitude, lat/lon, etc.)

    Returns:
    - df_profiles: DataFrame with session-level metrics
    - df_skipped: DataFrame with skipped sessions and failure reasons
    """
    
    records = []
    skipped = []

    try:
        schedule = ff1.get_event_schedule(year, backend="ergast")
        races = schedule[schedule.Session1DateUtc < datetime.utcnow()]
    except Exception as e:
        print(f"⚠️ Failed to load schedule for {year}: {e}")
        skipped.append({"year": year, "event": None, "session": None, "reason": str(e)})
        return pd.DataFrame(), pd.DataFrame(skipped)

    all_sessions = []
    for _, row in races.iterrows():
        event_name = row.get("EventName")
        location = row.get("Location")
        format_type = row.get("EventFormat")
        sessions = (
            ['FP1', 'FP2', 'FP3', 'Q', 'R']
            if format_type == 'conventional'
            else ['FP1', 'S', 'SS', 'SQ', 'Q', 'R']
        )
        for session_name in sessions:
            all_sessions.append((event_name, location, session_name))

    for event_name, location, session_name in tqdm(all_sessions, desc=f"{year} Sessions", unit="session"):
        try:
            session_data = load_session(year, event_name, session_name)

            if session_data['status'] == 'error':
                raise ValueError(session_data['reason'])

            telemetry_source = session_data['source']
            laps = session_data['laps']
            session = session_data['session']

            if laps is None or laps.empty:
                raise ValueError("Lap data missing")

            if session and telemetry_source == "fastf1":
                try:
                    lap = session.laps.pick_fastest()
                    track_length = lap.get_car_data().add_distance()['Distance'].max()
                except Exception:
                    track_length = np.nan
            else:
                track_length = laps.groupby("driver_number")["lap_distance"].max().max()

            drs_data = get_drs_info(session, track_length, event=event_name, session_name=session_name) if session else {
                'num_drs_zones': np.nan,
                'drs_total_len_m': np.nan,
                'drs_pct_of_lap': np.nan
            }
            track_metrics = extract_track_metrics(session) if session else None
            corner_data = get_circuit_corner_profile(session) if session else None

            altitude = circuit_metadata[circuit_metadata['location'] == location]['altitude'].values
            altitude = altitude[0] if len(altitude) else np.nan

            if not track_metrics:
                raise ValueError("Missing telemetry metrics")

            weather_data = get_weather_info(session, year, event_name, session_name) if session else {
                "air_temp_avg": np.nan,
                "track_temp_avg": np.nan,
                "rain_percentage": np.nan
            }
            
            record = {
                'year': year,
                'event': event_name,
                'location': location,
                'session': session_name,
                'real_altitude': altitude,
                'lap_length': track_length,
                'telemetry_source': telemetry_source,
                **drs_data,
                **track_metrics,
                **corner_data,
                **weather_data
            }

            records.append(record)

        except Exception as e:
            print(f"⚠️ Skipped {year} {event_name} {session_name}: {e}")
            skipped.append({
                "year": year,
                "event": event_name,
                "session": session_name,
                "reason": str(e)
            })

    return pd.DataFrame(records), pd.DataFrame(skipped)


def build_circuit_profile_df(start_year=2020, end_year=2025, only_specific=None):
    """
    Builds a DataFrame with circuit-level metrics for all practice sessions across seasons.

    Parameters:
    - start_year: int — First season to include
    - end_year: int — Last season to include
    - only_specific: set of (event_name, session_name) tuples to include selectively

    Returns:
    - df_profiles: DataFrame with circuit performance and layout characteristics
    - df_skipped: DataFrame logging skipped sessions
    """
    
    all_profiles = []
    all_skipped = []

    circuit_metadata = get_all_circuits(start_year, end_year)

    for year in range(start_year, end_year + 1):
        print(f"\n📅 Processing {year}...")
        df_year, df_skip = build_profiles_for_season(year, circuit_metadata)
        all_profiles.append(df_year)
        all_skipped.append(df_skip)

    df_profiles = pd.concat(all_profiles, ignore_index=True)
    df_skipped = pd.concat(all_skipped, ignore_index=True)

    print(f"\n✅ Done: {len(df_profiles)} sessions parsed, {len(df_skipped)} skipped.")
    return df_profiles, df_skipped
    

def update_profiles_file(cache_path="data/circuit_profiles.csv"):
    """
    Appends only the most recent sessions (not yet in the cached file).

    Parameters:
    - cache_path: str — Path to existing circuit_profiles CSV

    Returns:
    - df: Updated DataFrame with new sessions
    - skipped: DataFrame of skipped or failed sessions
    """
    if not Path(cache_path).exists():
        raise FileNotFoundError(f"❌ Cannot update — cache file not found at: {cache_path}")

    # Load existing cached data and define keys for existing sessions
    existing_df = pd.read_csv(cache_path)
    existing_sessions = set(existing_df.apply(lambda row: (row['year'], row['event'], row['session']), axis=1))

    current_year = datetime.utcnow().year
    schedule = ff1.get_event_schedule(current_year, backend='ergast')
    now = datetime.utcnow()
    past_races = schedule[schedule.Session1DateUtc < now]

    def get_session_list(format_type):
        return ['FP1', 'FP2', 'FP3', 'Q', 'R'] if format_type == 'conventional' else ['FP1','S','SQ','SS','Q','R']

    new_records = []
    skipped = []

    for _, row in past_races.iterrows():
        event = row['EventName']
        location = row['Location']
        year = row['Session1DateUtc'].year  # ✅ FIXED LINE
        format_type = row['EventFormat']

        for session_name in get_session_list(format_type):
            key = (year, event, session_name)
            if key in existing_sessions:
                continue

            try:
                print(f"📥 Appending {year} {event} {session_name}...")
                session_df, failed = build_circuit_profile_df(
                    start_year=year,
                    end_year=year,
                    only_specific={(event, session_name)}  # expects support in your function
                )
                if session_df.empty:
                    raise ValueError("Empty result")
                new_records.append(session_df)
                if failed is not None:
                    skipped.extend(failed.to_dict('records'))

            except Exception as e:
                print(f"⚠️ Failed to load {key}: {e}")
                skipped.append({
                    "year": year, "event": event, "session": session_name,
                    "location": location, "reason": str(e)
                })

    if new_records:
        new_df = pd.concat(new_records, ignore_index=True)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(cache_path, index=False)
        print(f"✅ Appended {len(new_df)} new session(s).")
        return combined_df, pd.DataFrame(skipped) if skipped else None
    else:
        print("ℹ️ No new sessions found to append.")
        return existing_df, None


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