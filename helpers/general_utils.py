"""
General utilities for the F1 analytics pipeline.

Includes session loading, data fetching, caching logic, and batch data processing helpers.
"""

# Library imports
import os
import glob
import logging
import functools
import requests
import pandas as pd
import fastf1 as ff1
import warnings

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from contextlib import contextmanager
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------------
# Suppress noisy FutureWarnings from fastf1
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*dtype incompatible with datetime64\\[ns\\].*",
    module="fastf1"
)
# Set logging level to ERROR to minimize output
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# Context managers
# ----------------------------------------------------------------------------

@contextmanager
def _suppress_inner_tqdm():
    """
    Temporarily disable inner tqdm progress bars when nested.

    Yields:
        None: Suppresses inner tqdm __init__ by forcing disable=True.
    """
    try:
        from tqdm import tqdm as tqdm_module
        original = tqdm_module.__init__
        # Monkey-patch to disable nested bars
        tqdm_module.__init__ = lambda self, *a, **kw: original(self, *a, **{**kw, "disable": True})
        yield
    finally:
        # Restore original tqdm constructor
        tqdm_module.__init__ = original

# ----------------------------------------------------------------------------
# Session loading and caching
# ----------------------------------------------------------------------------

def _session_date_col(event_format: str, event_row: pd.Series) -> dict[str, str]:
    """
    For a given event format + event row, return mapping:
    symbolic name → FastF1 schedule date column (e.g., 'Session1DateUtc')

    Example: {"FP1": "Session1DateUtc", "Q": "Session4DateUtc", ...}
    """
    symbolic_to_real: dict[str, str] = {
        "FP1": "Practice 1",
        "FP2": "Practice 2",
        "FP3": "Practice 3",
        "Q":   "Qualifying",
        "SQ":  "Sprint Qualifying",
        "SS":  "Sprint Shootout",
        "S":   "Sprint",
        "R":   "Race",
    }

    mapping: dict[str, str] = {}
    # Iterate through possible session columns
    for i in range(1, 6):  # Session1 to Session5
        label = event_row.get(f"Session{i}", "")
        for sym, real in symbolic_to_real.items():
            if label == real:
                mapping[sym] = f"Session{i}DateUtc"
    return mapping


def _official_schedule(year: int) -> pd.DataFrame:
    """
    Get official F1 schedule using multiple backends with fallbacks.

    Parameters:
        year (int): Championship year.

    Returns:
        pd.DataFrame: F1 event schedule for the given year.
    """
    try:
        # Primary: fastf1 backend
        return ff1.get_event_schedule(year, backend="fastf1")
    except Exception as e:
        print(f"⚠️ fastf1 backend failed: {e}")
        try:
            # Secondary: f1timing backend
            return ff1.get_event_schedule(year, backend="f1timing")
        except Exception as e:
            print(f"⚠️ F1 backend failed: {e}")
            try:
                # Tertiary: ergast backend (less complete)
                return ff1.get_event_schedule(year, backend="ergast")
            except Exception as e:
                print(f"❌ Failed to load event schedule for {year}: {e}")
                return pd.DataFrame()


def get_expected_sessions(year: int) -> Dict[str, List[str]]:
    """
    Determine which session names to expect for a given season.

    Parameters:
        year (int): Championship year.

    Returns:
        Dict mapping 'YYYY_RR' key to list of real session names.
    """
    sched = _official_schedule(year)
    event_sessions: Dict[str, List[str]] = {}

    for _, row in sched.iterrows():
        rnd = int(row['RoundNumber'])
        if rnd == 0:
            # Skip non-championship events (testing)
            continue
        key = f"{year}_{rnd:02d}"
        fmt = row.get('EventFormat', '').lower()
        # Define valid sessions based on event format
        if fmt == 'sprint_shootout':
            valid = ["Practice 1", "Qualifying", "Sprint Shootout", "Sprint", "Race"]
        elif fmt == 'sprint_qualifying':
            valid = ["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"]
        elif fmt == 'sprint':
            valid = ["Practice 1", "Qualifying", "Practice 2", "Sprint", "Race"]
        else:
            valid = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]
        event_sessions[key] = valid
    return event_sessions


def _session_list(event_format: str) -> List[str]:
    """
    Map EventFormat string to list of session codes for data collection.
    """
    fmt = (event_format or "").lower()
    if fmt == "testing":
        return []
    if fmt == "sprint_shootout":
        return ["FP1", "Q", "SS", "S", "R"]
    if fmt == "sprint_qualifying":
        return ["FP1", "SQ", "S", "Q", "R"]
    if fmt == "sprint":
        return ["FP1", "Q", "FP2", "S", "R"]
    # Default conventional format
    return ["FP1", "FP2", "FP3", "Q", "R"]


def _sessions_completed(format_type: str,
                        fp1_utc: datetime,
                        now: datetime) -> List[str]:
    """
    Given FP1 UTC start and current time, determine which sessions have started.
    """
    # Normalize tz-aware to naive UTC
    if fp1_utc.tzinfo:
        fp1_utc = fp1_utc.astimezone(timezone.utc).replace(tzinfo=None)
    if now.tzinfo:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)

    # Offsets (hours) for each session relative to FP1
    mapping = {
        "conventional":      [('FP1',  0), ('FP2',  4), ('FP3', 24), ('Q', 28), ('R', 52)],
        "sprint_qualifying": [('FP1',  0), ('SQ',  4), ('S', 28),  ('Q', 28), ('R', 52)],
        "sprint_shootout":   [('FP1',  0), ('Q',   4), ('SS',28),  ('S', 28), ('R', 52)],
        "sprint":            [('FP1',  0), ('Q',   4), ('FP2',28),  ('S', 28), ('R', 52)],
    }
    key = format_type if format_type in mapping else "conventional"
    # Return labels whose scheduled time ≤ now
    return [label for label, offset in mapping[key] if (fp1_utc + timedelta(hours=offset)) <= now]


def _completed_sessions(schedule: pd.DataFrame,
                        now: datetime) -> List[tuple[int,str,str]]:
    """
    Return list of (year, event_name, session_label) for all finished sessions.
    """
    todo: List[tuple[int,str,str]] = []
    for _, ev in schedule.iterrows():
        fmt = str(ev.EventFormat).lower()
        name = ev.EventName or ""
        fp1_utc = ev.Session1DateUtc
        year_tag = fp1_utc.year
        
        # Skip testing events entirely
        if fmt == "testing" or "test" in name.lower():
            continue
            
        # Add each completed session
        for ses in _sessions_completed(fmt, fp1_utc, now):
            todo.append((year_tag, name, ses))
            
    return todo

# Generic loaders
                    
def load_session(year: int, event_name: str, session_name: str) -> dict:
    """
    Load a session via FastF1 with fallback to OpenF1 API.

    Args:
        year: F1 season year.
        event_name: Grand Prix name.
        session_name: Code ('FP1', 'Q', 'Race', etc.).

    Returns:
        dict with keys: source, session, laps, status, reason.
    """
    # Map short codes to FastF1 full names
    session_map = {
        "FP1": "Practice 1",     
        "FP2": "Practice 2",
        "FP3": "Practice 3",     
        "Q":   "Qualifying",
        "R":   "Race",           
        "SQ":  "Sprint Qualifying",
        "S":   "Sprint",         
        "SS":  "Sprint Shootout"
    }
    ff_session_name = session_map.get(session_name, session_name)

    # Try FastF1 first
    try:
        session = ff1.get_session(year, event_name, ff_session_name)
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

    # Fallback to OpenF1 API for lap times
    try:
        print("🔄 Falling back to OpenF1 API...")
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

def get_weather_info(session, year: int, event_name: str, session_name: str) -> dict:
    """
    Extract weather information (temperature, humidity, pressure) from a session.

    Args:
        session (FastF1.Session or None): Loaded FastF1 session, or None if fallback needed.
        year (int): Season year
        event_name (str): Event name (e.g., 'Bahrain Grand Prix')
        session_name (str): Session label ('FP1', 'Race', etc.)

    Returns:
        dict with average air temp, track temp, and boolean rain presence
    """
    # Use FastF1 session if available
    if session is not None and hasattr(session, "weather_data") and not session.weather_data.empty:
        weather = session.weather_data
        return {
            "air_temp_avg": weather["AirTemp"].mean(),
            "track_temp_avg": weather["TrackTemp"].mean(),
            "rain_detected": weather["Rainfall"].max() > 0
        }

    # Else fallback to OpenF1 weather endpoint
    try:
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

@functools.lru_cache(maxsize=None)
def get_elevation(latitude: float, longitude: float, timeout: int = 10) -> float:
    """
    Query Open-Meteo API for ground elevation at given coordinates.

    Args:
        latitude: GPS latitude.
        longitude: GPS longitude.
        timeout: Request timeout in seconds.

    Returns:
        Elevation in meters.

    Raises:
        RuntimeError: If API returns no elevation.
    """
    url = f"https://api.open-meteo.com/v1/elevation?latitude={latitude}&longitude={longitude}"

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    payload = r.json()
    if "elevation" not in payload or payload["elevation"] is None:
        raise RuntimeError("Open-Meteo returned no elevation")

    return payload["elevation"][0]           # note: array, not dict

# ----------------------------------------------------------------------------
# Profile file management
# ----------------------------------------------------------------------------

# Cache maintenance helpers
def is_update_needed(cache_path: str, season: int = datetime.utcnow().year) -> bool:
    """
    Decide whether the cache CSV needs to be refreshed.

    Logic
        1. If the file does **not** exist  → True
        2. If *today* lies **inside** an ongoing race-weekend
           (Session1DateUtc ≤ now ≤ Session1DateUtc + 4 days) → True
        3. Else, look at the *next* race in the schedule:
           • return **True** once within *6 h before* FP1
           • otherwise **False**
    """
    # missing cache → definitely rebuild 
    if not os.path.exists(cache_path):
        return True

    try:
        sched = _official_schedule(season)
        now   = datetime.utcnow()

        # Check if within a race weekend (FP1 to Race + buffer)
        weekend_length = timedelta(days=4)
        ongoing = sched[
            (sched.Session1DateUtc <= now)
            & (sched.Session1DateUtc + weekend_length >= now)
        ]
        if not ongoing.empty:
            return True

        # Else, next race start -6h threshold
        upcoming = sched[sched.Session1DateUtc > now]
        if upcoming.empty:
            return False

        next_start = upcoming.iloc[0]["Session1DateUtc"]
        prebuffer  = timedelta(hours=6)
        
        return now >= next_start - prebuffer

    except Exception as e:
        print(f"⚠️  Could not check race schedule: {e}")
        return True  # safest fallback
        

def update_profiles_file(
    cache_path: str,
    start_year: int = None,
    end_year:   int = None,
    file_type:  str = "circuit",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Append sessions that (a) have already started and (b) are not yet cached.

    Supports 'circuit', 'driver', 'driver_timing'.

    For file_type=="circuit", builds each missing session in single‐session mode.
    For file_type=="driver", fetches the FastF1 session then runs get_all_driver_features.
    For file_type=="driver_timing", fetches detailed timing for each driver, except out/inlaps.
    """
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found at {cache_path}")

    existing = pd.read_csv(path)
    existing_keys = {(r.year, r.event, r.session) for r in existing.itertuples()}

    now = datetime.utcnow()
    sy = start_year or now.year
    ey = end_year or sy

    new_chunks, skipped = [], []

    for year in range(sy, ey + 1):
        sched = _official_schedule(year)
        completed = sched[sched.Session1DateUtc < now]
        todo = _completed_sessions(completed, now)

        for yr, ev_name, sess_label in todo:
            key = (yr, ev_name, sess_label)
            if key in existing_keys:
                continue

            print(f"📥  appending {key} …")
            
            try:
                
               # Delegate to appropriate builder

                if file_type == "circuit":
                    from .circuit_utils import _build_circuit_profile_df
                    df_ok, df_fail = _build_circuit_profile_df(
                        start_year=yr,
                        end_year=  yr,
                        only_specific={yr: {(ev_name, sess_label)}}
                    )

                elif file_type == "driver":
                    from .driver_utils import _build_driver_profile_df
                    df_ok, df_fail = _build_driver_profile_df(
                        year,
                        only_specific={(ev_name, sess)}
                    )
                    
                elif file_type == "driver_timing":
                    from .driver_utils import _build_detailed_telemetry
    
                    out_dir = os.path.dirname(cache_path)
                    os.makedirs(out_dir, exist_ok=True)
    
                    existing_files = {
                        os.path.basename(p)
                        for p in glob.glob(os.path.join(out_dir, "*.parquet"))
                    }
    
                    chunks = []
                    for yr, ev_name, sess_label in todo:
                        # build the target filename
                        fn = f"{yr}_{ev_name.replace(' ', '_')}_{sess_label}.parquet"
                        if fn in existing_files:
                            # skip already-written weekends
                            continue
    
                        info = load_session(yr, ev_name, sess_label)
                        if info.get("status") != "ok":
                            continue
                        sess_obj = info["session"]
    
                        df_tmp = _build_detailed_telemetry(sess_obj)
                        df_tmp["year"] = yr
                        df_tmp["event"] = ev_name
                        df_tmp["session"] = sess_label
    
                        # write this weekend’s parquet
                        path = os.path.join(out_dir, fn)
                        df_tmp.to_parquet(path,
                                          engine="pyarrow",
                                          compression="snappy",
                                          index=False)
                        chunks.append(df_tmp)
    
                    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                    skipped = pd.DataFrame()

                else:
                    raise ValueError(f"Unsupported file_type: {file_type!r}")

                new_chunks.append(df_ok)
                if df_fail is not None and not df_fail.empty:
                    skipped.extend(df_fail.to_dict("records"))

            except Exception as e:
                print(f"⚠️ failed to append {key} → {e}")
                skipped.append({
                    "year": yr,
                    "event": ev_name,
                    "session": sess_label,
                    "reason":  str(e)
                })

    if new_chunks:
        updated = pd.concat([existing, *new_chunks], ignore_index=True)
        if updated.empty or updated.shape[1] == 0:
            print(f"⚠️ Skipping save: empty or no columns [{path.name}]")
            return existing, pd.DataFrame(skipped)

        updated.to_csv(path, index=False)
        total = sum(len(df) for df in new_chunks)
        print(f"✅ added {total} row(s).")
        return updated, pd.DataFrame(skipped)

    print("ℹ️ No new sessions to append.")
    return existing, pd.DataFrame(skipped)
    
    
def load_or_build_profiles(
    start_year: int,
    end_year: int,
    file_type: str = "circuit",
    gp_name: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads cached profiles if available, otherwise builds from scratch or updates.

    Args
        start_year : int
        end_year   : int
        file_type  : str
            'circuit', 'driver', or 'driver_timing'
        gp_name    : str, optional
            If set and file_type=="circuit", only build circuit profiles for that GP.

    Returns
        df_profiles, 
        df_skipped
    """
    # driver_timing logic
    if file_type == "driver_timing":
        from .driver_utils import _build_driver_timing_profiles
        return _build_driver_timing_profiles(start_year, end_year)
       
    # circuit / driver logic
    end_year = end_year or start_year
    current_year = datetime.utcnow().year

    # Precompute only_specific mapping once, not inside the per-year loop
    only_specific: dict[int, set[tuple[str,str]]] | None = None
    if file_type == "circuit" and gp_name:
        from .circuit_utils import _build_circuit_profile_df
        only_specific = {}
        for yr in range(start_year, end_year + 1):
            sched = ff1.get_event_schedule(yr)
            row = sched[sched["EventName"] == gp_name]
            if row.empty:
                continue
            row = row.iloc[0]
            session_cols = [
                c for c in sched.columns
                if c.startswith("Session") and not c.endswith(("Date","DateUtc"))
            ]
            codes = [row[c] for c in session_cols if pd.notna(row[c])]
            only_specific[yr] = {(gp_name, code) for code in codes}

    all_data    = []
    all_skipped = []

    for year in range(start_year, end_year + 1):
        # driver_timing is already handled above, so we only get circuit/driver here
        cache_path = f"data/{file_type}/{year}_{file_type}_profiles.csv"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # 1) If no cache → build from scratch
        if not os.path.exists(cache_path):
            print(f"📂 No cache for {year}. Rebuilding...")

            if file_type == "circuit" and only_specific:
                from .circuit_utils import _build_circuit_profile_df
                df, skipped = _build_circuit_profile_df(
                    year, year,
                    only_specific={year: only_specific.get(year, set())}
                )

            elif file_type == "circuit":
                from .circuit_utils import _build_circuit_profile_df
                df, skipped = _build_circuit_profile_df(year, year)

            elif file_type == "driver":
                from .driver_utils import _build_driver_profile_df
                df, skipped = _build_driver_profile_df(
                    start_year=year,
                    end_year=year
                )

            else:
                raise ValueError(f"Unsupported file_type: {file_type!r}")

            # only circuit & driver write CSV
            df.to_csv(cache_path, index=False)
            if not skipped.empty or skipped.shape[1] != 0:
                skip_dir = os.path.join("data", "skipped", file_type)
                os.makedirs(skip_dir, exist_ok=True)
                
                skip_path = os.path.join(skip_dir, f"{year}_{file_type}_skipped.csv")
                skipped.to_csv(skip_path, index=False)

        # 2) If it's the current year and needs updating
        elif year == current_year and is_update_needed(cache_path, season=year):
            print(f"🔁 Updating {file_type} profile for {year}...")
            df, skipped = update_profiles_file(cache_path, year, year, file_type)

        # 3) Otherwise just load the cached CSV
        else:
            print(f"✅ Using cached {file_type} profile for {year}")
            df      = pd.read_csv(cache_path)
            skipped = pd.DataFrame()

        all_data.append(df)
        if not skipped.empty:
            all_skipped.append(skipped)

    df_all = pd.concat(all_data, ignore_index=True)    if all_data    else pd.DataFrame()
    skipped_all = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()

    return df_all, skipped_all


def ensure_year_dir(year: int, subdir: str = "data") -> str:
    """
    Create (if needed) and return a directory for a given year under subdir.

    Parameters:
        year (int): Year identifier.
        subdir (str): Parent directory name.

    Returns:
        str: Full path to year-specific directory.
    """
    year_path = os.path.join(subdir, str(year))
    os.makedirs(year_path, exist_ok=True)
    return year_path