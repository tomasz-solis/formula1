"""
General utilities for the F1 analytics pipeline.

Includes session loading, data fetching, caching logic, and batch data processing helpers.
"""

# Library imports

import os
import logging
import functools
import requests
import pandas as pd
import fastf1 as ff1
import warnings

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
from contextlib import contextmanager
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*dtype incompatible with datetime64\\[ns\\].*",
    module="fastf1"
)

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
        tqdm_module.__init__ = lambda self, *a, **kw: original(self, *a, **{**kw, "disable": True})
        yield
    finally:
        tqdm_module.__init__ = original

# ----------------------------------------------------------------------------
# Session loading and caching
# ----------------------------------------------------------------------------

# Session‑mapping helpers
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

    mapping = {}
    for i in range(1, 6):  # Session1 to Session5
        label = event_row.get(f"Session{i}", "")
        for sym, real in symbolic_to_real.items():
            if label == real:
                mapping[sym] = f"Session{i}DateUtc"

    return mapping



def _official_schedule(year: int) -> pd.DataFrame:
    """
    Get official F1 schedule (fastf1->f1timing->ergast backend).

    Parameters:
        year (int): Championship year.

    Returns:
        F1 schedule for a given year
    """
    try:
        return ff1.get_event_schedule(year, backend="fastf1")
    except Exception as e:
        print(f"⚠️ fastf1 backend failed: {e}")
        try:
            return ff1.get_event_schedule(year, backend="f1timing")
        except Exception as e:
            print(f"⚠️ F1 backend failed: {e}")
            try:
                return ff1.get_event_schedule(year, backend="ergast")
            except Exception as e:
                print(f"❌ Failed to load event schedule for {year} with any backend: {e}")
                return pd.DataFrame()


def get_expected_sessions(year: int) -> dict:
    """
    Determine which session names to expect for a given season.

    Parameters:
        year (int): Championship year.

    Returns:
        List[str]: List of session codes (e.g., ['FP1', 'FP2', 'FP3', 'QUALIFYING', 'RACE']).
    """
    import fastf1

    sched = _official_schedule(year)
    event_sessions = {}

    for _, row in sched.iterrows():
        rnd = int(row['RoundNumber'])
        if rnd == 0:
            continue  # skip testing or invalid events

        key = f"{year}_{rnd:02d}"
        event_format = row.get('EventFormat', '').lower()

        if event_format == 'sprint_shootout':
            valid_sessions = ["Practice 1", "Qualifying", "Sprint Shootout", "Sprint", "Race"]
        elif event_format == 'sprint_qualifying':
            valid_sessions = ["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"]
        elif event_format == 'sprint':
            valid_sessions = ["Practice 1", "Qualifying", "Practice 2", "Sprint", "Race"]
        else:
            valid_sessions = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]

        event_sessions[key] = valid_sessions

    return event_sessions


def _session_list(event_format: str) -> list[str]:
    """
    Map EventFormat → list of sessions expected to collect.

    conventional          → FP1 FP2 FP3 Q R
    sprint_shootout       → FP1 Q SS S R
    sprint_qualifying     → FP1 SQ S Q R
    sprint                → FP1 Q FP2 S R
    testing               → []
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

    return ["FP1", "FP2", "FP3", "Q", "R"]


def _sessions_completed(format_type: str,
                        fp1_utc: datetime,
                        now: datetime) -> list[str]:
    """
    Given a single weekend’s FP1 UTC start time (possibly tz-aware),
    return which session labels have already started, working with
    tz-naive UTC datetimes throughout.
    """
    # 1) Normalize fp1_utc to UTC and then drop tzinfo
    if fp1_utc.tzinfo is not None:
        fp1_utc = fp1_utc.astimezone(timezone.utc).replace(tzinfo=None)
    
    # 2) Ensure `now` is naive UTC
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    # if now.tzinfo is already None, assume it's UTC

    mapping = {
        "conventional":      [('FP1',  0), ('FP2',  4), ('FP3', 24), ('Q', 28), ('R', 52)],
        "sprint_qualifying": [('FP1',  0), ('SQ',  4), ('S', 28),  ('Q', 28), ('R', 52)],
        "sprint_shootout":   [('FP1',  0), ('Q',   4), ('SS',28),  ('S', 28), ('R', 52)],
        "sprint":            [('FP1',  0), ('Q',   4), ('FP2',28),  ('S', 28), ('R', 52)],
    }
    key = format_type if format_type in mapping else "conventional"

    return [
        label
        for label, offset in mapping[key]
        if (fp1_utc + timedelta(hours=offset)) <= now
    ]

def _completed_sessions(schedule: pd.DataFrame,
                        now: datetime) -> list[tuple[int,str,str]]:
    """
    Scan the full season schedule and return a list of
    (year, event_name, session_label) for all *finished* sessions
    that you should attempt to append.
    """
    todo: list[tuple[int,str,str]] = []

    for _, ev in schedule.iterrows():
        fmt      = str(ev.EventFormat).lower()
        name     = ev.EventName or ""
        fp1_utc  = ev.Session1DateUtc
        year_tag = fp1_utc.year

        #  skip *any* pre-season or in-season testing events - due to their nature
        if fmt == "testing" or "test" in name.lower():
            continue

        for ses in _sessions_completed(fmt, fp1_utc, now):
            todo.append((year_tag, name, ses))

    return todo

# Generic loaders
                    
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
    # 1) normalize our short codes to FastF1’s full names
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

    # 2) try FastF1
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

    # 3) fallback to OpenF1
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

def get_weather_info(session, year, event_name, session_name):
    """
    Extract weather information (temperature, humidity, pressure) from a session.

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
            "FP1": "Practice 1", 
            "FP2": "Practice 2", 
            "FP3": "Practice 3",
            "Q": "Qualifying", 
            "Race": "Race", 
            "SQ": "Sprint Qualifying",
            "S": "Sprint", 
            "SS": "Sprint Shootout"
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

@functools.lru_cache(maxsize=None)
def get_elevation(latitude: float, longitude: float,
                  timeout: int = 10) -> float:
    """
    Query external API for ground elevation at a GPS coordinate.

    Parameters:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.

    Returns:
        float: Elevation in meters.
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

# ----------------------------------------------------------------------------
# Profile file management
# ----------------------------------------------------------------------------

# Cache maintenance helpers
def is_update_needed(cache_path: str,
                     season: int = datetime.utcnow().year) -> bool:
    """
    Decide whether the cache CSV needs to be refreshed.

    Logic
    -----
    1. If the file does **not** exist  → True
    2. If *today* lies **inside** an ongoing race-weekend
       (Session1DateUtc ≤ now ≤ Session1DateUtc + 4 days) → True
    3. Else, look at the *next* race in the schedule:
       • return **True** once within *6 h before* FP1
       • otherwise **False**
    """
    # 0. missing cache → definitely rebuild 
    if not os.path.exists(cache_path):
        return True

    try:
        sched = _official_schedule(season)
        now   = datetime.utcnow()

        # 1. *inside* a race weekend already?
        weekend_length = timedelta(days=4)          # FP1 .. Race (+buffer)
        ongoing = sched[
            (sched.Session1DateUtc <= now)
            & (sched.Session1DateUtc + weekend_length >= now)
        ]
        if not ongoing.empty:
            return True                             # FP1/FP2/…/Race running

        # 2. otherwise look at the next race
        upcoming = sched[sched.Session1DateUtc > now]
        if upcoming.empty:
            return False                            # season finished

        next_start = upcoming.iloc[0]["Session1DateUtc"]
        prebuffer  = timedelta(hours=6)             # allow pre-FP1 update
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

    For file_type=="circuit", builds each missing session in single‐session mode.
    For file_type=="driver", fetches the FastF1 session then runs get_all_driver_features.
    """
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found at {cache_path}")

    existing      = pd.read_csv(path)
    existing_keys = {(r.year, r.event, r.session) for r in existing.itertuples()}

    now = datetime.utcnow()
    sy  = start_year or now.year
    ey  = end_year   or sy

    new_chunks = []
    skipped    = []

    for year in range(sy, ey + 1):
        sched     = _official_schedule(year)
        completed = sched[sched.Session1DateUtc < now]
        todo      = _completed_sessions(completed, now)

        for yr, ev_name, sess_label in todo:
            key = (yr, ev_name, sess_label)
            if key in existing_keys:
                continue

            print(f"📥  appending {key} …")
            try:
                if file_type == "circuit":
                    from .circuit_utils import _build_circuit_profile_df
                    df_ok, df_fail = _build_circuit_profile_df(
                        start_year=yr,
                        end_year=  yr,
                        only_specific={yr: {(ev_name, sess_label)}}
                    )

                elif file_type == "driver":
                    # 🔧 New: build only this one driver‐profile session
                    from .driver_utils import _build_driver_profile_df
                    df_ok, df_fail = _build_driver_profile_df(
                        year,
                        only_specific={(ev_name, sess)}
                    )

                else:
                    raise ValueError(f"Unsupported file_type: {file_type!r}")

                new_chunks.append(df_ok)
                if df_fail is not None and not df_fail.empty:
                    skipped.extend(df_fail.to_dict("records"))

            except Exception as e:
                print(f"⚠️ failed to append {key} → {e}")
                skipped.append({
                    "year":    yr,
                    "event":   ev_name,
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

    Parameters
    ----------
    start_year : int
    end_year   : int
    file_type  : str
        'circuit' or 'driver'
    gp_name    : str, optional
        If set and file_type=="circuit", only build circuit profiles for that GP.

    Returns
    -------
    df_profiles, df_skipped : DataFrame, DataFrame
    """

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

    all_data = []
    all_skipped = []

    for year in range(start_year, end_year + 1):
        cache_path = f"data/{year}_{file_type}_profiles.csv"

        # 1) If no cache → build from scratch
        if not os.path.exists(cache_path):
            print(f"📂 No cache for {year}. Rebuilding...")

            if file_type == "circuit" and only_specific:
                # only build the sessions in only_specific[year]
                from .circuit_utils import _build_circuit_profile_df
                df, skipped = _build_circuit_profile_df(
                    year, year,
                    only_specific={year: only_specific.get(year, set())}
                )
            elif file_type == "circuit":
                # build every circuit session for this year
                from .circuit_utils import _build_circuit_profile_df
                df, skipped = _build_circuit_profile_df(year, year)
            elif file_type == "driver":
                from .driver_utils import _build_driver_profile_df
                spec = only_specific.get(year) if only_specific else None
                df, skipped = _build_driver_profile_df(
                    start_year=year,
                    end_year=  year,
                    only_specific=spec
                )
            else:
                raise ValueError(f"Unsupported file_type: {file_type!r}")

            df.to_csv(cache_path, index=False)
            if not skipped.empty:
                skipped.to_csv(f"data/{year}_{file_type}_skipped.csv", index=False)

        # 2) If it's the current year and needs updating
        elif year == current_year and is_update_needed(cache_path, season=year):
            print(f"🔁 Updating {file_type} profile for {year}...")
            df, skipped = update_profiles_file(cache_path, year, year, file_type)

        # 3) Otherwise just load the cached CSV
        else:
            print(f"✅ Using cached {file_type} profile for {year}")
            df = pd.read_csv(cache_path)
            skipped = pd.DataFrame()

        all_data.append(df)
        if not skipped.empty:
            all_skipped.append(skipped)

    df_all = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
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