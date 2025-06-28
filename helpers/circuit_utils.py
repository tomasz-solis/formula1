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

# --- basic helpers ---

def _session_date_col(sess: str) -> str:
    """Map our session label → the matching Schedule column name."""
    return {
        "FP1": "Session1DateUtc",
        "FP2": "Session2DateUtc",
        "FP3": "Session3DateUtc",
        "SQ":  "Session2DateUtc",   # Sprint Quali lives in slot #2
        "S":   "Session3DateUtc",   # Sprint lives in slot #3
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

        # ── skip *any* pre-season or in-season testing events ───────────────
        if fmt == "testing" or "test" in name.lower():
            continue

        for ses in _sessions_completed(fmt, fp1_utc, now):
            todo.append((year_tag, name, ses))

    return todo


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

def build_profiles_for_season(
    year: int,
    circuit_metadata: pd.DataFrame,
    *,
    only_specific: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create session-level circuit-performance rows for **one** F1 season.

    Parameters
    ----------
    year : int
        Season to process (e.g. ``2025``).
    circuit_metadata : DataFrame
        Output of ``get_all_circuits`` – contains altitude, lat/lon …
    only_specific : {(event, session), …}, optional
        When given process *only* those (event, session) pairs (used by
        ``update_profiles_file``).  When ``None`` every past session is parsed.

    Returns
    -------
    df_profiles, df_skipped : DataFrame, DataFrame
    """
    records: list[dict] = []
    skipped: list[dict] = []

    # 1 ─ get schedule
    try:
        sched = _official_schedule(year)
        past  = sched[sched.Session1DateUtc < datetime.utcnow()]
    except Exception as e:
        skipped.append(
            {"year": year, "event": None, "session": None, "reason": str(e)}
        )
        return pd.DataFrame(), pd.DataFrame(skipped)

    # 2 ─ iterate events/sessions
    for _, ev in past.iterrows():
        ev_name = ev.EventName
        loc     = ev.Location
        fmt     = str(ev.EventFormat).lower()

        sessions = _session_list(fmt)

        for sess in sessions:
            if only_specific and (ev_name, sess) not in only_specific:
                continue                      # not requested → skip

            try:
                # 2-a  load data
                s_info = load_session(year, ev_name, sess)
                if s_info["status"] == "error":
                    raise ValueError(s_info["reason"])

                tele_src = s_info["source"]      # fastf1 / openf1
                laps     = s_info["laps"]
                session  = s_info["session"]     # FastF1.Session or None

                if laps is None or laps.empty:
                    raise ValueError("Lap data missing")

                # 2-b  basic lap length
                if session is not None:          # FastF1
                    try:
                        fast_lap = session.laps.pick_fastest()
                        lap_len  = (
                            fast_lap.get_car_data()
                            .add_distance()["Distance"].max()
                        )
                    except Exception:
                        lap_len = np.nan
                else:                            # OpenF1 fallback
                    lap_len = (
                        laps.groupby("driver_number")["lap_distance"].max().max()
                    )

                # 2-c  telemetry-derived metrics
                drs  = get_drs_info(session, lap_len, ev_name, sess) \
                       if session is not None else {
                           "num_drs_zones": np.nan,
                           "drs_total_len_m": np.nan,
                           "drs_pct_of_lap": np.nan,
                       }

                tmet = extract_track_metrics(session)           if session else None
                cmet = get_circuit_corner_profile(session)      if session else None
                wmet = get_weather_info(session, year, ev_name, sess)

                if not tmet:
                    raise ValueError("Missing telemetry metrics")

                # 2-d  altitude lookup
                try:
                    alt = (
                        circuit_metadata
                        .loc[circuit_metadata["location"] == loc, "altitude"]
                        .iloc[0]
                    )
                except IndexError:
                    alt = np.nan

                # 2-e  assemble row
                records.append(
                    {
                        "year": year,
                        "event": ev_name,
                        "location": loc,
                        "session": sess,
                        "real_altitude": alt,
                        "lap_length": lap_len,
                        "telemetry_source": tele_src,
                        **drs, **tmet, **cmet, **wmet,
                    }
                )

            except Exception as e:
                skipped.append(
                    {
                        "year": year,
                        "event": ev_name,
                        "session": sess,
                        "reason": str(e),
                    }
                )

    return pd.DataFrame(records), pd.DataFrame(skipped)


def build_circuit_profile_df(
    start_year: int = 2020,
    end_year:   int = 2025,
    *,
    only_specific: set[tuple[str, str]] | None = None,
):
    """
    Build / refresh the circuit-session profile table over multiple seasons.

    Parameters
    ----------
    start_year , end_year : int
        Season range (inclusive).
    only_specific : {(event_name, session_name), …} | None, keyword-only
        Forwarded to ``build_profiles_for_season``; see its doc-string.

    Returns
    -------
    df_profiles , df_skipped
    """
    meta = get_all_circuits(start_year, end_year)

    prof_all, skip_all = [], []
    for yr in range(start_year, end_year + 1):
        print(f"📅 Processing {yr} …")
        df_year, df_skip = build_profiles_for_season(
            yr,
            meta,
            only_specific=only_specific,
        )
        prof_all.append(df_year)
        skip_all.append(df_skip)

    df_profiles = pd.concat(prof_all,  ignore_index=True)
    df_skipped  = pd.concat(skip_all, ignore_index=True)

    print(
        f"✅ Done: {len(df_profiles)} sessions parsed, "
        f"{len(df_skipped)} skipped."
    )
    return df_profiles, df_skipped

    
def update_profiles_file(
    cache_path:  str  = "data/circuit_profiles.csv",
    start_year:  int  = None,
    end_year:    int  = None
):
    """
    Append sessions that (a) have already started and (b) are not yet cached.
    """
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache not found at {cache_path}")

    existing = pd.read_csv(path)
    existing_keys = {
        (r.year, r.event, r.session)
        for r in existing.itertuples()
    }

    now     = datetime.utcnow()
    sy      = start_year or now.year
    ey      = end_year   or sy

    new_chunks = []
    skipped    = []

    for year in range(sy, ey + 1):
        sched     = _official_schedule(year)
        # only events whose FP1 has begun
        completed = sched[sched.Session1DateUtc < now]

        for _, ev in completed.iterrows():
            ev_name  = ev.EventName
            ev_fmt   = ev.EventFormat
            sess_list = _session_list(ev_fmt)

            for sess in sess_list:
                # —— NEW: skip any session whose own timestamp is still in the future
                col = _session_date_col(sess)
                sess_dt = getattr(ev, col, None)
                if sess_dt is None or sess_dt > now:
                    # either missing in the schedule or not yet started
                    continue

                key = (year, ev_name, sess)
                if key in existing_keys:
                    continue

                print(f"📥  appending {key} …")
                try:
                    df_ok, df_fail = build_circuit_profile_df(
                        start_year    = year,
                        end_year      = year,
                        only_specific = {(ev_name, sess)}
                    )
                    if df_ok.empty:
                        raise ValueError("no data returned")
                    new_chunks.append(df_ok)
                    if not df_fail.empty:
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


def load_or_build_profiles(
    cache_path: str = "data/circuit_profiles.csv",
    start_year: int  = 2020,
    end_year:   int  = None
):
    """
    Load cached profiles if still valid; else append new race or rebuild from scratch.

    Parameters
    ----------
    cache_path : str
        Path to cached CSV file.
    start_year : int
        First season to include if (re)building.
    end_year : int | None
        Last season to include; defaults to start_year if not given.
    """
    end_year = end_year or start_year

    # 1) no cache → full rebuild
    if not os.path.exists(cache_path):
        print("📂 No cache found. Rebuilding full dataset...")
        df, skipped = build_circuit_profile_df(start_year, end_year)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_csv(cache_path, index=False)
        return df, skipped

    # 2) cache exists & race weekend started → incremental update
    if is_update_needed(cache_path, season=datetime.utcnow().year):
        print("🔁 Race weekend started — updating recent sessions only...")
        return update_profiles_file(cache_path, start_year, end_year)

    # 3) otherwise just load the file
    print("✅ Using cached circuit profile file.")
    return pd.read_csv(cache_path), None