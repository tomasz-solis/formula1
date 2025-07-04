"""
General helpers shared across the analytics stack.
"""
import os, logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import contextlib
import functools
import requests
import pandas as pd
import fastf1 as ff1
from tqdm import tqdm
import warnings


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*dtype incompatible with datetime64\\[ns\\].*",
    module="fastf1"
)

# Suppress FastF1 INFO and DEBUG messages
logging.getLogger("fastf1").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

@contextlib.contextmanager
def _suppress_inner_tqdm():
    """
    Temporarily monkey-patch tqdm.tqdm so all inner bars are disabled.
    """
    import tqdm as _tqdm_mod
    orig = _tqdm_mod.tqdm
    _tqdm_mod.tqdm = functools.partial(orig, disable=True)
    try:
        yield
    finally:
        _tqdm_mod.tqdm = orig


# Session‑mapping helpers
def _session_date_col(sess: str) -> str:
    """Map our session label → the matching Schedule column name."""
    return {
        "FP1": "Session1DateUtc",
        "FP2": "Session2DateUtc",
        "FP3": "Session3DateUtc",
        "SQ":  "Session2DateUtc",   
        "S":   "Session3DateUtc",
        "Q":   "Session4DateUtc",
        "R":   "Session5DateUtc",
    }[sess]


def _session_list(event_format: str) -> list[str]:
    """
    Map EventFormat → list of sessions expected to collect.

    conventional          → FP1 FP2 FP3 Q R
    sprint_*              → FP1 SQ  S   Q R
    testing               → (return []) - Tests are ignored
    """
    fmt = (event_format or "").lower()

    if fmt == "testing":
        return []                      # ← nothing to fetch

    if fmt.startswith("sprint"):
        return ["FP1", "SQ", "S", "Q", "R"]

    # conventional (default)
    return ["FP1", "FP2", "FP3", "Q", "R"]

def _official_schedule(year: int) -> pd.DataFrame:
    """Try FastF1 API first, fall back to F1 API with a warning."""
    try:
        return ff1.get_event_schedule(year, backend="fastf1")
    except Exception as e:        # noqa: BLE001
        print(f"⚠️  F1 API schedule failed → {e}  (falling back to offcial F1 API)")
        return ff1.get_event_schedule(year, backend="f1timing")

def _sessions_completed(format_type: str,
                        fp1_utc: datetime,
                        now: datetime) -> list[str]:
    """
    Given a single weekend’s FP1 UTC start time, return which session
    labels have *already* started (and so should be fetchable).
    """
    # how many hours after FP1 each session typically begins
    mapping = {
        "conventional": [('FP1',  0), ('FP2',  4), ('FP3', 24), ('Q', 28), ('R', 52)],
        "sprint":       [('FP1',  0), ('SQ',  4), ('S', 28),  ('Q', 28), ('R', 52)]
    }
    key = "sprint" if format_type.startswith("sprint") else "conventional"
    return [
        label for label, offset in mapping[key]
        if fp1_utc + timedelta(hours=offset) <= now
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


# ── Generic loaders / external data ───────────────────────────────────────────

                    
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
        "FP1": "Practice 1",     "FP2": "Practice 2",
        "FP3": "Practice 3",     "Q":   "Qualifying",
        "R":   "Race",           "SQ":  "Sprint Qualifying",
        "S":   "Sprint",         "SS":  "Sprint Shootout"
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

@functools.lru_cache(maxsize=None)
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

# ── Cache maintenance helpers ────────────────────────────────────────────────
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

def update_profiles_file(cache_path: str, start_year: int = None,
                         end_year: int = None, file_type: str = "circuit"):
    """
    Append sessions that (a) have already started and (b) are not yet cached.

    For file_type=="circuit", calls build_circuit_profile_df.
    For file_type=="driver", calls get_all_driver_features per session.
    """
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found at {cache_path}")

    existing     = pd.read_csv(path)
    existing_keys = {(r.year, r.event, r.session) for r in existing.itertuples()}

    now = datetime.utcnow()
    sy  = start_year or now.year
    ey  = end_year   or sy

    new_chunks = []
    skipped    = []

    for year in range(sy, ey + 1):
        sched     = _official_schedule(year)
        completed = sched[sched.Session1DateUtc < now]  # sessions whose FP1 has begun

        for _, ev in completed.iterrows():
            ev_name  = ev.EventName
            ev_fmt   = ev.EventFormat
            for sess in _session_list(ev_fmt):
                col     = _session_date_col(sess)
                sess_dt = getattr(ev, col, None)
                if sess_dt is None or sess_dt > now:
                    continue  # not yet started

                key = (year, ev_name, sess)
                if key in existing_keys:
                    continue  # already cached

                print(f"📥  appending {key} …")
                try:
                    if file_type == "circuit":
                        from .circuit_utils import build_circuit_profile_df
                        df_ok, df_fail = build_circuit_profile_df(
                            start_year    = year,
                            end_year      = year,
                            only_specific = {(ev_name, sess)}
                        )

                    elif file_type == "driver":
                        from .driver_utils import get_all_driver_features
                        info = load_session(year, ev_name, sess)
                        if info["status"] != "ok":
                            raise ValueError(info["reason"])
                        df_ok = get_all_driver_features(
                            info["session"], year=year, session_name=sess
                        )
                        df_fail = None
                        if df_ok.empty:
                            raise ValueError("no driver features returned")

                    else:
                        raise ValueError(f"Unsupported file_type: {file_type}")

                    new_chunks.append(df_ok)
                    if df_fail is not None and not df_fail.empty:
                        skipped.extend(df_fail.to_dict("records"))

                except Exception as e:
                    print(f"⚠️ failed to append {key} → {e}")
                    skipped.append({
                        "year":    year,
                        "event":   ev_name,
                        "session": sess,
                        "reason":  str(e)
                    })

    if new_chunks:
        out = pd.concat([existing, *new_chunks], ignore_index=True)
        out.to_csv(path, index=False)
        total = sum(len(df) for df in new_chunks)
        print(f"✅ added {total} row(s).")
        return out, (pd.DataFrame(skipped) if skipped else None)

    print("ℹ️ No new sessions to append.")
    return existing, None

    
def load_or_build_profiles(
    file_type: str = "circuit",
    start_year: int = 2020,
    end_year: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load or build profiles for multiple seasons.
    Only updates current season if needed.
    Returns concatenated DataFrame and skipped sessions (if any).
    """
    from .circuit_utils import _build_circuit_profile_df
    from .driver_utils import get_all_driver_features,_build_driver_profile_df_for_year

    end_year = end_year or start_year
    current_year = datetime.utcnow().year

    all_data = []
    all_skipped = []

    for year in range(start_year, end_year + 1):
        cache_path = f"data/{year}_{file_type}_profiles.csv"

        # 1) If no cache → build
        if not os.path.exists(cache_path):
            print(f"📂 No cache for {year}. Rebuilding...")

            if file_type == "circuit":
                df, skipped = _build_circuit_profile_df(year, year)
            elif file_type == "driver":
                df, skipped = _build_driver_profile_df_for_year(year)
            else:
                raise ValueError(f"Unsupported file_type: {file_type!r}")

            df.to_csv(cache_path, index=False)
            if skipped is not None and not skipped.empty:
                skipped.to_csv(f"data/{year}_{file_type}_skipped.csv", index=False)

        # 2) If current year → maybe update
        elif year == current_year and is_update_needed(cache_path, season=year):
            print(f"🔁 Updating {file_type} profile for {year}...")
            df, skipped = update_profiles_file(cache_path, year, year, file_type)

        else:
            print(f"✅ Using cached {file_type} profile for {year}")
            df = pd.read_csv(cache_path)
            skipped = None

        all_data.append(df)
        if skipped is not None and not skipped.empty:
            all_skipped.append(skipped)

    df_all = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    skipped_all = pd.concat(all_skipped, ignore_index=True) if all_skipped else None

    return df_all, skipped_all


def ensure_year_dir(year: int, subdir: str = "data") -> str:
    """Ensure that the year-specific data directory exists and return the path."""
    year_path = os.path.join(subdir, str(year))
    os.makedirs(year_path, exist_ok=True)
    return year_path