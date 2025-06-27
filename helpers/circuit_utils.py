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

def build_profiles_for_season(
    year: int,
    circuit_metadata: pd.DataFrame,
    *,
    only_specific: set[tuple[str, str]] | None = None,
):
    """
    Create session-level circuit-performance rows for **one** F1 season.

    Parameters
    ----------
    year : int
        Season to process (e.g. ``2025``).
    circuit_metadata : pandas.DataFrame
        Output of ``get_all_circuits`` – contains altitude and lat/lon per track.
    only_specific : {(event_name, session_name), …} | None, keyword-only
        If given, *process only* those (event, session) pairs.  
        When ``None`` (default) every past session of the year is processed.
        ``update_profiles_file`` sets this filter when it wants to append just
        the newest sessions (FP1, FP2 …) instead of rebuilding everything.

    Returns
    -------
    df_profiles : pandas.DataFrame
        One row per processed session containing telemetry / DRS / weather …
    df_skipped : pandas.DataFrame
        Log of sessions that failed or were unavailable.
    """
    records, skipped = [], []

    # 1. get schedule 
    try:
        sched   = ff1.get_event_schedule(year, backend="ergast")
        past    = sched[sched.Session1DateUtc < datetime.utcnow()]
    except Exception as e:
        skipped.append(
            {"year": year, "event": None, "session": None, "reason": str(e)}
        )
        return pd.DataFrame(), pd.DataFrame(skipped)

    # 2. iterate over events / sessions 
    for _, row in past.iterrows():
        event = row.EventName
        loc   = row.Location
        fmt   = row.EventFormat
        sess_list = (
            ["FP1", "FP2", "FP3", "Q", "R"]
            if fmt == "conventional"
            else ["FP1", "S", "SS", "SQ", "Q", "R"]
        )

        for sess in sess_list:
            #  filter when called from update_profiles_file 
            if only_specific and (event, sess) not in only_specific:
                continue

            try:
                # load session (FastF1 → OpenF1 fallback) 
                s_info = load_session(year, event, sess)
                if s_info["status"] == "error":
                    raise ValueError(s_info["reason"])

                tele_src = s_info["source"]
                laps     = s_info["laps"]
                session  = s_info["session"]

                if laps is None or laps.empty:
                    raise ValueError("Lap data missing")

                # lap length
                if session and tele_src == "fastf1":
                    try:
                        lap_len = (
                            session.laps.pick_fastest()
                            .get_car_data().add_distance()["Distance"].max()
                        )
                    except Exception:
                        lap_len = np.nan
                else:  # OpenF1 fallback
                    lap_len = (
                        laps.groupby("driver_number")["lap_distance"].max().max()
                    )

                # telemetry-derived metrics
                drs = (
                    get_drs_info(session, lap_len, event, sess)
                    if session is not None
                    else {"num_drs_zones": np.nan,
                          "drs_total_len_m": np.nan,
                          "drs_pct_of_lap": np.nan}
                )
                tmet  = extract_track_metrics(session)          if session else None
                cmet  = get_circuit_corner_profile(session)     if session else None
                wmet  = get_weather_info(session, year, event, sess)

                if not tmet:
                    raise ValueError("Missing telemetry metrics")

                # altitude lookup
                try:
                    alt = circuit_metadata.loc[
                        circuit_metadata["location"] == loc, "altitude"
                    ].iloc[0]
                except IndexError:
                    alt = np.nan

                # collect row
                records.append(
                    {
                        "year": year,
                        "event": event,
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
                        "event": event,
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


def _sessions_completed(format_type: str,
                        fp1_utc:      datetime,
                        now:          datetime | None = None) -> list[str]:
    """
    Return the list of session labels that *should already be completed*
    at the moment `now` (UTC).

    We assume fixed offsets from FP1.  These are conservative: if a session
    starts 30-60 min earlier/later at a particular venue it doesn’t matter.

    Offsets (h after FP1)
       conventional weekend            sprint weekend
       ─────────────────────            ──────────────
       FP1   0                          FP1   0
       FP2   4                          SQ    4   (Sprint Quali – Fri)
       FP3  24                          S    28   (Sprint – Sat)
       Q    28                          Q    28   (Quali – Sat)
       R    52                          R    52   (Race – Sun)
    """
    if now is None:
        now = datetime.utcnow()

    elapsed = (now - fp1_utc).total_seconds() / 3600
    schedule = (
        [('FP1', 0), ('FP2', 4), ('FP3', 24), ('Q', 28), ('R', 52)]
        if format_type == 'conventional'
        else [('FP1', 0), ('SQ', 4), ('S', 28), ('Q', 28), ('R', 52)]
    )
    return [label for label, offset in schedule if elapsed >= offset]
    

def update_profiles_file(cache_path: str = "data/circuit_profiles.csv"):
    """
    Append **only the sessions that have actually finished** but are still
    missing from the cached CSV.

    Parameters
    ----------
    cache_path : str
        Path to ``circuit_profiles.csv``

    Returns
    -------
    df  : pandas.DataFrame
        Updated circuit profile table.
    skipped : pandas.DataFrame | None
        Sessions we tried but could not fetch.
    """
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Cache not found →  {cache_path}")

    existing_df       = pd.read_csv(path)
    existing_sessions = {
        (r.year, r.event, r.session) for r in existing_df.itertuples()
    }

    utc_now   = datetime.utcnow()
    season    = utc_now.year
    schedule  = ff1.get_event_schedule(season, backend="ergast")

    # Consider ONLY race weekends whose FP1 has already occurred
    finished_fp1 = schedule[schedule.Session1DateUtc < utc_now]

    new_chunks: list[pd.DataFrame] = []
    skipped:     list[dict]        = []

    for _, ev in finished_fp1.iterrows():
        ev_name  = ev.EventName
        ev_fmt   = ev.EventFormat        # conventional / sprint_qualifying
        fp1_utc  = ev.Session1DateUtc
        year_tag = fp1_utc.year          # should equal `season`

        # sessions that should already have telemetry
        for ses in _sessions_completed(ev_fmt, fp1_utc, utc_now):
            key = (year_tag, ev_name, ses)
            if key in existing_sessions:
                continue       # already cached

            print(f"📥 appending {key} …")
            try:
                df_ok, df_fail = build_circuit_profile_df(
                    start_year=year_tag,
                    end_year=year_tag,
                    only_specific={(ev_name, ses)},
                )
                if df_ok.empty:
                    raise ValueError("no data returned")
                new_chunks.append(df_ok)
                if not df_fail.empty:
                    skipped.extend(df_fail.to_dict("records"))

            except Exception as e:
                print(f"⚠️ {key} failed → {e}")
                skipped.append(dict(year=year_tag, event=ev_name,
                                    session=ses, reason=str(e)))

    # ── write back
    if new_chunks:
        combined = pd.concat([existing_df, *new_chunks], ignore_index=True)
        combined.to_csv(path, index=False)
        print(f"✅ added {sum(len(x) for x in new_chunks)} row(s).")
        return combined, (pd.DataFrame(skipped) if skipped else None)

    print("ℹ️ No new sessions to append.")
    return existing_df, None


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
       • return **True** once we are within *6 h before* FP1
       • otherwise **False**

    This fixes the earlier issue where—once FP1 had already happened—the
    function returned *False* because the *“next”* race was the one **after**
    the weekend in progress.
    """
    # 0. missing cache → definitely rebuild 
    if not os.path.exists(cache_path):
        return True

    try:
        sched = ff1.get_event_schedule(season, backend="ergast")
        now   = datetime.utcnow()

        # 1. are we *inside* a race weekend already?
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