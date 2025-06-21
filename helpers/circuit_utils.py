import fastf1 as ff1
import pandas as pd
from datetime import datetime
import numpy as np
from fastf1.ergast import Ergast
import requests
from functools import lru_cache
import logging
from tqdm.notebook import tqdm
from IPython.display import display


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

        
# --- ELEVATION LOOKUP ---

@lru_cache(maxsize=None)
def get_elevation(latitude, longitude):
    """ 
    Get the elevation of a single point based on coordinates.
    
    Parameters:
    - latitude
    - longitude

    Returns single value - altitude above sea level in meters.
    """
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}"
    
    response = requests.get(url)
    data = response.json()
    elevation = data["results"][0]["elevation"]

    return elevation
    
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
    from tqdm import tqdm

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
                **corner_data
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


def build_circuit_profile_df(start_year=2020, end_year=2025):
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

