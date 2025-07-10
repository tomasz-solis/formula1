"""
Per‑driver performance metrics & feature extraction.
"""

# Library imports

# monkey‐patch fastf1 to disable driver‐ahead and marker‐distance

import fastf1
import fastf1.core as f1core

# 1) No‐op the driver‐ahead code so lap.get_telemetry() never fails:
f1core.Telemetry.add_driver_ahead       = lambda self, *a, **k: self
f1core.Telemetry.calculate_driver_ahead = lambda self, *a, **k: None

# 2) No‐op the marker‐distance code so get_circuit_info() never fails:
f1core.CircuitInfo.add_marker_distance  = lambda self, *a, **k: None


# end monkey‐patch

import numpy as np
import pandas as pd
import warnings
import os

from datetime import datetime
from typing import Dict, List, Sequence, Tuple, Optional
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*dtype incompatible with datetime64\\[ns\\].*",
    module="fastf1"
)
# ----------------------------------------------------------------------------
# Driver Characteristics per session
# ----------------------------------------------------------------------------
# Throttle & tyre degradation
def get_driver_max_throttle_ratio(session, 
                                  driver, 
                                  max_throttle_threshold: int = 98, 
                                  season: int | None = None,
                                  session_name: str | None = None,):    
    
    """
    Calculate the full throttle ratio for a driver's fastest lap.

    Parameters:
    - session: FastF1 loaded session
    - driver: str, driver code (e.g., 'VER')
    - max_throttle_threshold: float, minimum throttle to be considered full throttle
    - season: optional, int
    - session_name: optional, str (e.g., 'FP1')

    Returns:
    - result: pd.DataFrame with telemetry and weather metrics
    - missing_info: pd.DataFrame fallback if telemetry unavailable
    """
    gp_name = session.event['EventName']
    location = session.event['Location']

    try:
        fastest_driver = session.laps.pick_drivers([driver]).pick_fastest()
        telemetry = fastest_driver.get_telemetry().add_distance()

        telemetry = pd.merge_asof(
            telemetry,
            session.weather_data[['Time', 'Rainfall', 'TrackTemp', 'AirTemp']],
            left_on='SessionTime',
            right_on='Time'
        )

        telemetry['delta_speed'] = telemetry['Speed'].diff()
        heavy_brakes = telemetry['delta_speed'] < -30
        braking_events = heavy_brakes.sum()

        telemetry['nextThrottle'] = telemetry.Throttle.shift(-1)
        telemetry['previousThrottle'] = telemetry.Throttle.shift(1)

        throttle_points = telemetry.loc[
            (telemetry.Throttle >= max_throttle_threshold) &
            (
                (telemetry.nextThrottle < max_throttle_threshold) |
                (telemetry.previousThrottle < max_throttle_threshold) |
                (telemetry.index.isin([telemetry.index[0], telemetry.index[-1]]))
            )
        ].copy()
        throttle_points['FTRelative'] = throttle_points.RelativeDistance.diff().fillna(0)

        max_throttle_ratio = throttle_points.loc[
            (throttle_points.nextThrottle < max_throttle_threshold) |
            (throttle_points.nextThrottle.isna())
        ]['FTRelative'].sum()

        result = pd.DataFrame([{
            'grand_prix': gp_name,
            'location': location,
            'driver': driver,
            'max_throttle_ratio': max_throttle_ratio,
            'compound': fastest_driver['Compound'],
            'tyre_age': fastest_driver['TyreLife'],
            'is_fresh_tyre': fastest_driver['FreshTyre'],
            'avg_rainfall': telemetry['Rainfall'].mean(),
            'avg_track_temp': telemetry['TrackTemp'].mean(),
            'avg_air_temp': telemetry['AirTemp'].mean(),
            'braking_events': braking_events,
            'session_uid': f"{season}_{location}_{session_name}" if season and session_name else None
        }])

        return result, None

    except Exception:
        missing = pd.DataFrame([{
            'grand_prix': gp_name,
            'location': location,
            'driver': driver,
            'session_uid': f"{season}_{location}_{session_name}" if season and session_name else None
        }])
        return None, missing

def _compute_degradation(session, driver):
    """
    Estimate tire degradation from lap time evolution.
    First tries the last stint; if <2 laps there, falls back to all laps.
    Returns None only if the driver has <2 total laps.
    """
    # grab every lap by this driver, drop any missing LapTime
    laps = session.laps.pick_drivers([driver]).copy()
    laps = laps[laps['LapTime'].notna()].sort_values('LapNumber')

    if len(laps) < 2:
        return None

    # try last stint
    last_stint = laps[laps['Stint'] == laps['Stint'].max()]
    data = last_stint if len(last_stint) >= 2 else laps

    X = data['LapNumber'].to_numpy().reshape(-1, 1)
    y = data['LapTime'].dt.total_seconds().to_numpy()

    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]

    # pick a compound label if available (else "unknown")
    compound = (data.iloc[0].get('Compound')
                if 'Compound' in data.columns else "unknown")

    return {
        'driver': driver,
        'compound': compound or "unknown",
        'degradation_slope': slope
    }

def estimate_tire_degradation(session, year, session_name):
    """
    Returns: dict of driver → {'compound':…, 'degradation_slope':…}
    """
    results = {}
    for drv in session.laps["Driver"].unique():
        info = _compute_degradation(session, drv)
        if info is not None:
            results[drv] = info
    return results

# DRS usage
def _compute_drs_for_driver(session,
                          driver: str,
                          return_nan_if_constant: bool = False) -> float:
    """
    Count DRS flap-open activations on the driver's fastest lap.

    Parameters
    ----------
    session : Loaded FastF1 session..
    driver : str
        Three-letter code ('VER', 'HAM', …).
    return_nan_if_constant : bool, default False
        • False – constant DRS channel ⇒ return 0  
        • True  – constant DRS channel ⇒ return np.nan

    Returns
    -------
    int | numpy.nan
        Rising-edge count of the open-flap bit (bit-2).
    """
    if session is None:
        return np.nan # no telemetry at all

    lap = session.laps.pick_drivers([driver]).pick_fastest()
    tel = lap.get_car_data()

    # Channel missing completely?
    if "DRS" not in tel.columns:
        return np.nan

    # Constant channel → choose 0 or nan
    if tel["DRS"].nunique() <= 1:
        return np.nan if return_nan_if_constant else 0

    # Bit-mask: open flap if bit-2 set
    flap_open = (tel["DRS"].astype(int) & 0b0100) > 0
    activations = (flap_open & ~flap_open.shift(fill_value=False)).sum()

    return int(activations)


def count_drs_activations(session, year, session_name):
    """
    Count number of DRS activations per driver and per lap for all drivers in the session.
    Returns a dict: { driver: drs_activation_count }.
    """
    weekend = f"{year} {session.event['EventName']}"

    counts, failures = {}, []
    drivers = session.laps["Driver"].unique()

    for driver in drivers:
        try:
            counts[driver] = _compute_drs_for_driver(session, driver)
        except Exception:
            failures.append(driver)

    return counts

# Braking intensity
def _compute_braking_metric(session, driver, braking_drop_kmh: int = 30):
    """
    Compute braking intensity (max / mean negative g-force) on the driver's
    fastest lap.

    Parameters
    ----------
    session : fastf1.core.Session
    driver  : str   – 'VER', 'HAM', …
    braking_drop_kmh : int
        Speed drop (km/h) between two telemetry samples that qualifies as a
        braking event (default 30 km/h).

    Returns
    -------
    dict  or  None (if telemetry missing)
    """
    lap = session.laps.pick_drivers([driver]).pick_fastest()
    tel = lap.get_car_data().add_distance()

    # Δv and Δt
    tel["delta_speed"] = tel["Speed"].diff()
    tel["delta_time"]  = tel["Time"].diff().dt.total_seconds()

    # instant deceleration (m/s²) → g
    tel["decel"] = -tel["delta_speed"] / 3.6 / tel["delta_time"]
    tel.loc[tel["delta_speed"].abs() < braking_drop_kmh, "decel"] = np.nan

    return {
        "driver":       driver,
        "brake_max_g":  tel["decel"].max(skipna=True)  / 9.81,
        "brake_avg_g":  tel["decel"].mean(skipna=True) / 9.81,
    }


def braking_intensity(session, year, session_name, drop_kmh: int = 30):
    """
    Identify the breaking intensity per driver.
    Returns a dict: { driver: scalar or {metric_name: value, …} }.
    """
    weekend = f"{year} {session.event['EventName']}"

    intensities, failures = {}, []
    drivers = session.laps["Driver"].unique()

    for driver in drivers:
        try:
            bi = _compute_braking_metric(session, driver, drop_kmh)
            intensities[driver] = bi
        except Exception:
            failures.append(driver)

    return intensities


# ----------------------------------------------------------------------------
# Main general driver feature wrapper
# ----------------------------------------------------------------------------
    
def get_all_driver_features(
    session,
    year: int | None         = None,
    session_name: str | None = None,
    *,
    throttle_ratio_min: float = 0.40,
    throttle_ratio_max: float = 0.85,
    braking_drop_kmh: int     = 30,
) -> pd.DataFrame:
    """
    One row per driver with:
      • max_throttle_ratio, tyre_age, is_fresh_tyre
      • avg_rainfall, avg_track_temp, avg_air_temp
      • braking_events, brake_max_g, brake_avg_g
      • drs_activations
      • degradation_slope, compound
    plus year/event/session/location columns.
    """
    if session is None or session.laps.empty:
        return pd.DataFrame()

    # 1) compute the three driver→{…} maps
    degr_map = estimate_tire_degradation(session, year, session_name)
    brake_map = braking_intensity(session, year, session_name, drop_kmh=braking_drop_kmh)
    drs_map = count_drs_activations(session, year, session_name)

    # 2) now assemble each driver’s row
    records = []
    for drv in session.laps["Driver"].unique():
        base_df, missing = get_driver_max_throttle_ratio(
            session, drv,
            season=year,
            session_name=session_name,
        )
        if base_df is None or base_df.empty:
            continue
        row = base_df.iloc[0].to_dict()

        # 3) merge in tire-deg slope + compound
        if drv in degr_map:
            # degr_map[drv] = {'driver':drv, 'compound':…, 'degradation_slope':…}
            info = degr_map[drv]
            row["compound"] = info.get("compound") if info else row.get("compound")
            row["degradation_slope"] = info.get("degradation_slope") if info else np.nan

        # 4) merge in brake stats
        if drv in brake_map:
            # brake_map[drv] = {'driver':drv, 'brake_max_g':…, 'brake_avg_g':…}
            info = brake_map[drv]
            row["brake_max_g"] = info.get("brake_max_g")
            row["brake_avg_g"] = info.get("brake_avg_g")
            row["braking_events"] = info.get("braking_events", row.get("braking_events"))

        # 5) drs activations
        row["drs_activations"] = drs_map.get(drv, 0)

        # 6) tagging
        row["year"] = year
        row["session"] = session_name
        row["event"] = session.event["EventName"]
        row["location"] = session.event["Location"]

        records.append(row)

    df = pd.DataFrame(records)

    # 7) throttle‐ratio outlier filter
    if not df.empty:
        mask = (
            (df["max_throttle_ratio"] < throttle_ratio_min) |
            (df["max_throttle_ratio"] > throttle_ratio_max)
        )
        df = df.loc[~mask]

    return df.reset_index(drop=True)

# ----------------------------------------------------------------------------
# Main wrapper
# ----------------------------------------------------------------------------

def _build_driver_profile_df(
    start_year: int,
    end_year:   int,
    *,
    only_specific: dict[int, set[tuple[str, str]]] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build driver profiles for a range of seasons.

    Parameters
    ----------
    start_year : int
        First year to include.
    end_year : int
        Last year to include (inclusive).
    only_specific : dict of {year: set of (event,session)}, optional
        If provided, only those exact sessions will be processed.

    Returns
    -------
    df_profiles, df_skipped : (DataFrame, DataFrame)
    """
    from .general_utils import _official_schedule, _completed_sessions, load_session

    all_profiles = []
    all_skipped  = []

    now = datetime.utcnow()
    for year in range(start_year, end_year + 1):
        # 1) Grab only the sessions that have finished (skipping testing)
        sched = _official_schedule(year)
        done  = _completed_sessions(sched, now)

        # 2) Optionally prune to only_specific[year]
        if only_specific and year in only_specific:
            want = only_specific[year]
            done = [d for d in done if (d[1], d[2]) in want]

        # 3) Show the progress bar over those completed sessions
        print(f"{year} sessions:") 
        for yr, ev_name, sess_label in tqdm(
            done,
            total=len(done),
            desc=f"{year} sessions",
            colour="magenta",
        ):
            try:
                info = load_session(yr, ev_name, sess_label)
                if info["status"] != "ok":
                    raise ValueError(info["reason"])

                df = get_all_driver_features(
                    info["session"],
                    year=yr,
                    session_name=sess_label
                )
                if df is None or df.empty:
                    raise ValueError("no driver features returned")

                # tag with date for traceability
                df["session_date"] = info["session"].date
                all_profiles.append(df)

            except Exception as e:
                all_skipped.append({
                    "year": yr,
                    "event": ev_name,
                    "session": sess_label,
                    "reason": str(e)
                })

    # 4) Concat results
    df_profiles = (
        pd.concat(all_profiles, ignore_index=True)
        if all_profiles else pd.DataFrame()
    )
    df_skipped  = (
        pd.DataFrame(all_skipped)
        if all_skipped else pd.DataFrame()
    )
    return df_profiles, df_skipped


# ----------------------------------------------------------------------------
# Driver detailed timing in each session
# ----------------------------------------------------------------------------

def get_corner_area(session, max_attempts: int = 5) -> dict[int, float]:
    """
    Compute apex distances for every corner on the circuit.

    Finds a single valid lap with positional data, merges its X/Y track
    coordinates with the lap’s distance timeline, and then uses a KD-tree
    to snap each corner (from circuit info) to its nearest telemetry point.

    Parameters:
        session: A loaded FastF1 Session object.
        max_attempts: How many fastest laps to try before giving up.

    Returns:
        A dict mapping corner_index → apex_distance_m along the lap.

    Raises:
        RuntimeError: If no lap with valid position data is found.
    """
    fast_laps = session.laps.pick_quicklaps().sort_values("LapTime")
    valid_lap = None

    for i, lap in enumerate(fast_laps.itertuples()):
        if i >= max_attempts:
            break
        try:
            _ = session.pos_data[lap.DriverNumber]
            valid_lap = session.laps.loc[lap.Index]
            break
        except Exception:
            continue

    if valid_lap is None:
        raise RuntimeError("No lap with valid position data found.")

    # Merge position (X,Y) with distance timeline
    pos = valid_lap.get_pos_data().copy()
    car = valid_lap.get_car_data().add_distance().copy()
    pos["t"] = pos["Time"].dt.total_seconds()
    car["t"] = car["Time"].dt.total_seconds()

    merged = pd.merge_asof(
        pos[["t", "X", "Y"]].sort_values("t"),
        car[["t", "Distance"]].sort_values("t"),
        on="t",
        direction="nearest"
    ).dropna(subset=["X", "Y", "Distance"])

    # KD‐tree corners → nearest telemetry sample → Distance
    tree    = cKDTree(merged[["X", "Y"]].values)
    corners = (
        session
        .get_circuit_info()
        .corners
        .dropna(subset=["X", "Y"])
        .reset_index()
    )
    coords  = corners[["X", "Y"]].values
    _, idxs = tree.query(coords, k=1)

    apex_distances = merged.iloc[idxs]["Distance"].to_numpy()
    corner_indices = corners["index"].to_numpy()
    return dict(zip(corner_indices, apex_distances))


def get_detailed_lap_telemetry(
    lap,
    corner_dists: dict[int, float],
    corner_window: float = 100.0
) -> pd.DataFrame:
    """
    Build per-sample telemetry from scratch, bypassing FastF1's internals.

    Tags each sample with:
      - Distance, RelativeDistance
      - Sector (1,2,3)
      - Corner (0 if none, else corner index)
      - DriverNumber, LapNumber, Year, EventName, SessionName, Location

    Parameters:
        lap: A FastF1 Lap object
        corner_dists: Mapping corner_index → apex_distance_m
        corner_window: ±m around each apex to mark as “corner”

    Returns:
        DataFrame with:
          DriverNumber, LapNumber, Time, Speed, RPM, nGear,
          Throttle, Brake, DRS, Distance, RelativeDistance,
          Sector, Corner, Year, EventName, SessionName, Location
    """
    # 1) pull raw pos & car (index holds timestamp or timedelta)
    pos = lap.get_pos_data().copy()
    car = lap.get_car_data().copy()

    # 2) drop any existing time cols
    for df in (pos, car):
        for col in ("Time", "SessionTime"):
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # 3) reset index → bring timestamp/timedelta into “Time”
    # index.name may be None, so default to "index"
    idx = pos.index.name or "index"
    pos = pos.reset_index().rename(columns={idx: "Time"})
    idx = car.index.name or "index"
    car = car.reset_index().rename(columns={idx: "Time"})

    # 4) bail if no data or no “Time” column
    if pos.empty or car.empty or "Time" not in pos or "Time" not in car:
        return pd.DataFrame()

    # 5) coerce to datetime or timedelta
    # only need to compute t = seconds since start
    pos["Time"] = pd.to_datetime(pos["Time"])
    car["Time"] = pd.to_datetime(car["Time"])

    # 6) compute elapsed‐seconds since lap start into “t”
    start = pos["Time"].iloc[0]
    pos["t"] = (pos["Time"] - start).dt.total_seconds()
    car["t"] = (car["Time"] - start).dt.total_seconds()

    # 7) merge on t, drop rows without Speed
    merged = pd.merge_asof(
        pos.sort_values("t"),
        car[["t","Speed","RPM","nGear","Throttle","Brake","DRS"]].sort_values("t"),
        on="t", direction="nearest"
    ).dropna(subset=["Speed"])

    # 8) integrate Speed (km/h) → Distance (m) and compute RelativeDistance
    dt = merged["t"].diff().fillna(0)
    merged["Distance"] = (merged["Speed"] * (1000.0/3600.0) * dt).cumsum()
    total_dist = merged["Distance"].iat[-1] if not merged.empty else 0
    merged["RelativeDistance"] = merged["Distance"] / total_dist

    # 9) sector tagging using t
    s1_sec = lap["Sector1Time"].total_seconds()
    s2_sec = (lap["Sector1Time"] + lap["Sector2Time"]).total_seconds()

    merged["Sector"] = 3
    merged.loc[merged["t"] <= s1_sec, "Sector"] = 1
    merged.loc[(merged["t"] > s1_sec) & (merged["t"] <= s2_sec), "Sector"] = 2

    # 10) corner tagging via Distance
    merged["Corner"] = 0
    for cn, apex in corner_dists.items():
        mask = merged["Distance"].between(apex - corner_window, apex + corner_window)
        merged.loc[mask, "Corner"] = int(cn)

    # 11) identifiers & session metadata
    merged["DriverNumber"] = lap.DriverNumber
    merged["LapNumber"] = lap["LapNumber"]
    merged["Year"] = lap.session.date.year
    merged["EventName"] = lap.session.event.get("EventName")
    merged["SessionName"] = lap.session.name
    merged["Location"] = lap.session.event.get("Location")

    # 12) select & return
    return merged[[
        "DriverNumber","LapNumber","Time","Speed","RPM","nGear",
        "Throttle","Brake","DRS","Distance","RelativeDistance",
        "Sector","Corner","Year","EventName","SessionName","Location"
    ]]



def _build_detailed_telemetry(session) -> pd.DataFrame:
    """
    Build a per-sample telemetry DataFrame for every lap in the session,
    tagged with sector and corner, and include session metadata.

    Parameters:
        session: A loaded FastF1 Session.

    Returns:
        A DataFrame containing every telemetry point for every lap,
        with columns:
          Year, EventName, SessionName, Location,
          DriverNumber, LapNumber, Time, RPM, nGear, Throttle, Brake,
          DRS, Distance, RelativeDistance, Sector, Corner
    """
    # 1) extract corner→distance once
    corner_dists = get_corner_area(session)
    # 2) tag every lap’s telemetry - excluding Inlap/Outlap
    laps_df = session.laps.pick_wo_box()
    # only keep laps whose DriverNumber is in session.car_data keys
    valid_drivers = set(session.car_data.keys())
    laps_df = laps_df[laps_df["DriverNumber"].isin(valid_drivers)]
    all_frames = []
    n_laps = len(laps_df)
    for _, lap in tqdm(
        laps_df.iterlaps(),
        desc=f"{session.date.year} {session.event['EventName']} {session.name}",
        total=n_laps,
        colour="green"
    ):
        try:
            df_lap = get_detailed_lap_telemetry(lap, corner_dists)
            if df_lap.empty:
                continue
            all_frames.append(df_lap)
        except Exception as e:
             print(f"   ⚠️ skipped lap {lap.name} ({lap.DriverNumber}) → {type(e).__name__}: {e!r}")
             continue
            
    if not all_frames:
        return pd.DataFrame()
        
    # 3) concatenate
    return pd.concat(all_frames, ignore_index=True)
