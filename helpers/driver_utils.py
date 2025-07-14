"""
Per‑driver performance characteristics metrics & feature extraction.
"""

# Library imports

import fastf1
import fastf1.core as f1core
import numpy as np
import pandas as pd
import warnings
import os
import glob

from datetime import datetime
from typing import Dict, List, Sequence, Tuple, Optional
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.spatial import cKDTree
from .general_utils import _official_schedule, _completed_sessions, load_session

# -------------------------------------------------------------------
# Monkey-patch fastf1 to avoid failures in driver-ahead and marker-distance
# -------------------------------------------------------------------

# 1) No‐op the driver‐ahead code so lap.get_telemetry() never fails:
f1core.Telemetry.add_driver_ahead       = lambda self, *a, **k: self
f1core.Telemetry.calculate_driver_ahead = lambda self, *a, **k: None

# 2) No‐op the marker‐distance code so get_circuit_info() never fails:
f1core.CircuitInfo.add_marker_distance  = lambda self, *a, **k: None

# ----------------------------------------------------------------------------
# Suppress specific FutureWarnings from fastf1
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

def get_driver_max_throttle_ratio(
    session,
    driver: str,
    max_throttle_threshold: int = 98,
    season: Optional[int] = None,
    session_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Calculate the full throttle ratio for a driver's fastest lap.

    Args:
        session: Loaded FastF1 session containing laps and telemetry.
        driver: Three-letter driver code (e.g., 'VER', 'HAM').
        max_throttle_threshold: Throttle value threshold to count as full throttle.
        season: If provided, used to build a unique session identifier.
        session_name: If provided, used to build a unique session identifier.

    Returns:
        result: pd.DataFrame with telemetry and weather metrics for full throttle segments.
        missing_info: Fallback DataFrame if telemetry is unavailable or errors occur.

    Raises:
        None: Exceptions are caught and returned as missing_info.
    """
    
    gp_name = session.event['EventName'] # Grand Prix name
    location = session.event['Location'] # Circuit location

    try:
        fastest_driver = session.laps.pick_drivers([driver]).pick_fastest()
        telemetry = fastest_driver.get_car_data().add_distance() # ensure Distance column
    
        total_dist = telemetry["Distance"].max()
        if total_dist > 0:
            telemetry["RelativeDistance"] = telemetry["Distance"] / total_dist
        else:
            telemetry["RelativeDistance"] = 0

        # Merge weather data by timestamp
        telemetry = pd.merge_asof(
            telemetry,
            session.weather_data[['Time', 'Rainfall', 'TrackTemp', 'AirTemp']],
            on='Time'
        )
        
        # Identify heavy braking events (>30 km/h drop)
        telemetry['delta_speed'] = telemetry['Speed'].diff()
        heavy_brakes = telemetry['delta_speed'] < -30
        braking_events = heavy_brakes.sum()

        # Identify throttle transition points
        telemetry['nextThrottle'] = telemetry.Throttle.shift(-1)
        telemetry['previousThrottle'] = telemetry.Throttle.shift(1)

        # Points where throttle crosses the full-throttle threshold
        throttle_points = telemetry.loc[
            (telemetry.Throttle >= max_throttle_threshold) &
            (
                (telemetry.nextThrottle < max_throttle_threshold) |
                (telemetry.previousThrottle < max_throttle_threshold) |
                (telemetry.index.isin([telemetry.index[0], telemetry.index[-1]]))
            )
        ].copy()

        # Compute proportion of lap at full throttle
        throttle_points['FTRelative'] = throttle_points.RelativeDistance.diff().fillna(0)

        max_throttle_ratio = throttle_points.loc[
            (throttle_points.nextThrottle < max_throttle_threshold) |
            (throttle_points.nextThrottle.isna())
        ]['FTRelative'].sum()

        # Compile results into a DataFrame row
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

    except Exception as e:
        # Fallback: return minimal info if telemetry extraction fails
        missing = pd.DataFrame([{
            'grand_prix': gp_name,
            'location': location,
            'driver': driver,
            'session_uid': f"{season}_{location}_{session_name}" if season and session_name else None
        }])
        
        return None, missing

def _compute_degradation(session, driver: str) -> Optional[Dict[str, float]]:
    """
    Estimate tire degradation slope from lap time evolution.

    Picks the most recent stint; if it has fewer than 2 laps, uses all laps.

    Args:
        session: Loaded FastF1 session.
        driver: Three-letter driver code.

    Returns:
        A dict with keys 'driver', 'compound', and 'degradation_slope',
        or None if fewer than 2 valid laps are present.
    """
    # Select all valid laps for driver and sort by lap number
    laps = session.laps.pick_drivers([driver]).copy()
    laps = laps[laps['LapTime'].notna()].sort_values('LapNumber')

    if len(laps) < 2:
        return None  # Not enough data to compute a slope

    # Choose last stint if it has at least 2 laps, else use all laps
    last_stint = laps[laps['Stint'] == laps['Stint'].max()]
    data = last_stint if len(last_stint) >= 2 else laps

    # Prepare data for linear regression: lap number vs lap time (seconds)
    X = data['LapNumber'].to_numpy().reshape(-1, 1)
    y = data['LapTime'].dt.total_seconds().to_numpy()
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]  # degradation rate in seconds per lap

    # Retrieve compound label if available, default to "unknown"
    compound = data.iloc[0].get('Compound', "unknown") or "unknown"

    return {
        'driver': driver,
        'compound': compound,
        'degradation_slope': slope
    }

def estimate_tire_degradation(
    session,
    year: int,
    session_name: str
) -> Dict[str, Dict[str, float]]:
    """
    Compute tire degradation info for all drivers in a session.

    Args:
        session: Loaded FastF1 session.
        year: Season year.
        session_name: Session label (e.g., 'FP1').

    Returns:
        Mapping of driver code to degradation info dict.
    """
    results: Dict[str, Dict[str, float]] = {}
    for drv in session.laps['Driver'].unique():
        info = _compute_degradation(session, drv)
        if info is not None:
            results[drv] = info
    return results

def _compute_drs_for_driver(
    session,
    driver: str,
    return_nan_if_constant: bool = False
) -> float:
    """
    Count DRS flap-open activations on the driver's fastest lap.

    Args:
        session: Loaded FastF1 session.
        driver: Three-letter code (e.g., 'VER', 'HAM').
        return_nan_if_constant: If True, return NaN when DRS channel is constant.

    Returns:
        int or np.nan: Number of rising-edge activations of the DRS open-flap bit.
    """
    if session is None:
        return np.nan  # no telemetry available

    lap = session.laps.pick_drivers([driver]).pick_fastest()
    tel = lap.get_car_data()

    # If DRS channel missing, cannot compute activations
    if "DRS" not in tel.columns:
        return np.nan

    # Handle constant signal channels
    if tel["DRS"].nunique() <= 1:
        return np.nan if return_nan_if_constant else 0

    # Bit-mask check for flap-open (bit-2)
    flap_open = (tel["DRS"].astype(int) & 0b0100) > 0
    activations = (flap_open & ~flap_open.shift(fill_value=False)).sum()

    return int(activations)


def count_drs_activations(session, year: int, session_name: str) -> Dict[str, float]:
    """
    Count DRS activations per driver for all drivers in the session.

    Args:
        session: Loaded FastF1 session.
        year: Season year.
        session_name: Session label.

    Returns:
        Dict mapping driver code to activation count.
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

def _compute_braking_metric(
    session,
    driver: str,
    braking_drop_kmh: int = 30
) -> Optional[Dict[str, float]]:
    """
    Compute braking intensity (max and mean deceleration in g) on a driver's fastest lap.

    Args:
        session: Loaded FastF1 session.
        driver: Three-letter code.
        braking_drop_kmh: Threshold drop in km/h to qualify as braking event.

    Returns:
        Dict with driver, max g, and avg g, or None if telemetry missing.
    """
    
    lap = session.laps.pick_drivers([driver]).pick_fastest()
    tel = lap.get_car_data().add_distance()

    # Calculate speed and time differences
    tel["delta_speed"] = tel["Speed"].diff()
    tel["delta_time"] = tel["Time"].diff().dt.total_seconds()

    # Compute deceleration (m/s²) and convert to g's
    tel["decel"] = -tel["delta_speed"] / 3.6 / tel["delta_time"]
    tel.loc[tel["delta_speed"].abs() < braking_drop_kmh, "decel"] = np.nan

    return {
        "driver": driver,
        "brake_max_g": tel["decel"].max(skipna=True) / 9.81,
        "brake_avg_g": tel["decel"].mean(skipna=True) / 9.81,
    }


def braking_intensity(
    session,
    year: int,
    session_name: str,
    drop_kmh: int = 30
) -> Dict[str, Dict[str, float]]:
    """
    Identify braking intensity metrics for all drivers in a session.

    Args:
        session: Loaded FastF1 session.
        year: Season year.
        session_name: Session label.
        drop_kmh: Speed drop threshold for braking event.

    Returns:
        Dict mapping driver to braking metrics dict.
    """
    intensities: Dict[str, Dict[str, float]] = {}
    for driver in session.laps["Driver"].unique():
        try:
            intensities[driver] = _compute_braking_metric(session, driver, drop_kmh)
        except Exception:
            intensities[driver] = {}
    return intensities

# ----------------------------------------------------------------------------
# Main general driver feature wrapper
# ----------------------------------------------------------------------------
    
def get_all_driver_features(
    session,
    year: Optional[int] = None,
    session_name: Optional[str] = None,
    *,
    throttle_ratio_min: float = 0.40,
    throttle_ratio_max: float = 0.85,
    braking_drop_kmh: int = 30,
) -> pd.DataFrame:
    """
    Compile all driver feature metrics for a session into a DataFrame.

    Metrics include throttle ratio, tyre metrics, weather averages,
    braking events/g-forces, DRS activations, and degradation slope.

    Args:
        session: Loaded FastF1 session.
        year: Season year (optional).
        session_name: Session label (optional).
        throttle_ratio_min: Lower bound to filter unrealistic values.
        throttle_ratio_max: Upper bound to filter unrealistic values.
        braking_drop_kmh: Threshold for braking calculation.

    Returns:
        DataFrame with one row per driver and feature columns.
    """
    
    if session is None or session.laps.empty:
        return pd.DataFrame()

    # Step 1: compute maps of metrics per driver
    degr_map = estimate_tire_degradation(session, year, session_name)
    brake_map = braking_intensity(session, year, session_name, drop_kmh=braking_drop_kmh)
    drs_map = count_drs_activations(session, year, session_name)

    records: List[Dict] = []
    for drv in session.laps["Driver"].unique():
        base_df, missing = get_driver_max_throttle_ratio(
            session, drv, season=year, session_name=session_name
        )
        if base_df is None or base_df.empty:
            continue  # skip drivers with no telemetry data
        row = base_df.iloc[0].to_dict()

        # Merge tire degradation info
        if drv in degr_map:
            info = degr_map[drv]
            row.update({
                "compound": info.get("compound", row.get("compound")),
                "degradation_slope": info.get("degradation_slope", np.nan)
            })

        # Merge braking stats
        if drv in brake_map:
            info = brake_map[drv]
            row.update({
                "brake_max_g": info.get("brake_max_g"),
                "brake_avg_g": info.get("brake_avg_g"),
                "braking_events": info.get("braking_events", row.get("braking_events"))
            })

        # Add DRS activations
        row["drs_activations"] = drs_map.get(drv, 0)

        # Add metadata tags
        row.update({
            "year": year,
            "session": session_name,
            "event": session.event.get("EventName"),
            "location": session.event.get("Location")
        })

        records.append(row)

    df = pd.DataFrame(records)

    # Step 2: filter out throttle-ratio outliers
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
    end_year: int,
    *,
    only_specific: Optional[Dict[int, set[Tuple[str, str]]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build driver profiles for a range of seasons by loading sessions and extracting features.

    Args:
        start_year: First season year to include.
        end_year: Last season year (inclusive).
        only_specific: Optional map of year to a set of (event,session) to restrict processing.

    Returns:
        Tuple of:
          - df_profiles: concatenated feature DataFrame for all processed sessions.
          - df_skipped: DataFrame of sessions that failed with reasons.
    """

    all_profiles, all_skipped = [], []
    now = datetime.utcnow()

    for year in range(start_year, end_year + 1):
        sched = _official_schedule(year)
        done  = _completed_sessions(sched, now)

        if only_specific and year in only_specific:
            want = only_specific[year]
            done = [d for d in done if (d[1], d[2]) in want]

        print(f"{year} sessions:") 
        for yr, ev_name, sess_label in tqdm(
            done,
            total=len(done),
            desc=f"{year} sessions",
            colour="magenta",
        ):
            try:
                info = load_session(yr, ev_name, sess_label)
                if info.get("status") != "ok":
                    raise ValueError(info.get("reason", "unknown"))

                df = get_all_driver_features(
                    info["session"], 
                    year=yr, 
                    session_name=sess_label
                )
                
                if df.empty:
                    raise ValueError("no driver features returned")

                df["session_date"] = info["session"].date  # for traceability
                all_profiles.append(df)

            except Exception as e:
                all_skipped.append({
                    "year": yr,
                    "event": ev_name,
                    "session": sess_label,
                    "reason": str(e)
                })

    df_profiles = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()
    df_skipped  = pd.DataFrame(all_skipped) if all_skipped else pd.DataFrame()

    return df_profiles, df_skipped

# ----------------------------------------------------------------------------
# Detailed telemetry
# ----------------------------------------------------------------------------

def get_corner_area(
    session,
    max_attempts: int = 5
) -> Dict[int, float]:
    """
    Compute apex distances for every true circuit corner using positional telemetry.

    Args:
        session: Loaded FastF1 session.
        max_attempts: Number of quick laps to probe for valid position data.

    Returns:
        Mapping of corner index to distance along the lap in meters.

    Raises:
        RuntimeError: If no valid lap with position data within max_attempts.
    """
    
    fast_laps = session.laps.pick_quicklaps().sort_values("LapTime")
    valid_lap = None
    for i, lap in enumerate(fast_laps.itertuples()):
        if i >= max_attempts:
            break # found a usable lap
        try:
            _ = session.pos_data[lap.DriverNumber]
            valid_lap = session.laps.loc[lap.Index]
            break
        except Exception:
            continue
    if valid_lap is None:
        raise RuntimeError("No lap with valid position data found.")

    # Merge positional X/Y with Distance from car data
    pos = valid_lap.get_pos_data().copy()
    car = valid_lap.get_car_data().add_distance().copy()
    pos['t'] = pos['Time'].dt.total_seconds()
    car['t'] = car['Time'].dt.total_seconds()
    merged = (pd.merge_asof(
        pos[['t','X','Y']].sort_values('t'),
        car[['t','Distance']].sort_values('t'),
        on='t', direction='nearest'
    )
    .dropna(subset=['X','Y','Distance']))

    corners = session.get_circuit_info().corners.dropna(subset=['X','Y']).reset_index()
    if corners.empty:
        return {}
    # Use spatial tree to snap each corner to nearest telemetry point
    tree = cKDTree(merged[['X','Y']].values)
    _, idxs = tree.query(corners[['X','Y']].values, k=1)

    # Determine actual column for corner index
    if 'CornerNumber' in corners.columns:
        idx_col = 'CornerNumber'
    elif 'CornerIdx' in corners.columns:
        idx_col = 'CornerIdx'
    else:
        idx_col = 'index'
        
    corners = corners.reset_index(drop=True)
    apex_distances = merged.iloc[idxs]["Distance"].to_numpy()
    corner_numbers = np.arange(1, len(apex_distances) + 1)
    
    return dict(zip(corner_numbers, apex_distances))

def get_detailed_lap_telemetry(
    lap,
    corner_dists: Dict[int, float],
    corner_window: float = 100.0,
    debug: bool = True
) -> pd.DataFrame:
    """
    Build per-sample telemetry for a single lap, tagging sectors and corners,
    and injecting lap-summary and session metadata.

    Args:
        lap: A fastf1.Lap object with pos and car data access.
        corner_dists: Mapping of corner index to apex distance.
        corner_window: ± meters around apex to mark as corner region.
        debug: If True, prints tagging debug information.

    Returns:
        Telemetry DataFrame with enriched columns, or empty if failure.
    """
    try:
        # 1) retrieve raw position and car telemetry
        pos = lap.get_pos_data().copy()
        car = lap.get_car_data().add_distance().copy()
        if pos.empty or car.empty:
            return pd.DataFrame()

        # 2) normalize time index into a column for merge
        for df in (pos, car):
            df["LapTimeIndex"] = df.index
            df.reset_index(drop=True, inplace=True)

        # 3) ensure consistent Time column and detect duplicates
        for name, df in (("pos", pos), ("car", car)):
            if "Time" not in df.columns:
                df.rename(columns={"LapTimeIndex": "Time"}, inplace=True)
            else:
                df.drop(columns=["LapTimeIndex"], inplace=True)
                if debug:
                    dup = df.columns[df.columns.duplicated()].tolist()
                    if dup:
                        print(f"⚠️ {name} has duplicate columns: {dup}")

        # 4) convert Time to seconds since lap start
        for df in (pos, car):
            if not np.issubdtype(df["Time"].dtype, np.timedelta64):
                df["Time"] = pd.to_timedelta(df["Time"])
        pos["t"] = (pos["Time"] - pos["Time"].iloc[0]).dt.total_seconds()
        car["t"] = (car["Time"] - car["Time"].iloc[0]).dt.total_seconds()

        # 5) merge on nearest timestamp and drop incomplete rows
        merged = pd.merge_asof(
            pos.sort_values("t"),
            car[["t","Speed","RPM","nGear","Throttle","Brake","DRS","Distance"]]
               .sort_values("t"),
            on="t", direction="nearest"
        ).dropna(subset=["Speed","Distance"])
        if merged.empty:
            return pd.DataFrame()

        # 6) compute relative distance fraction of lap
        total = merged["Distance"].iat[-1]
        merged["RelativeDistance"] = merged["Distance"] / total if total>0 else 0

        # 7) tag each sample with sector based on lap sector times
        merged["Sector"] = 3
        try:
            s1 = lap.Sector1Time
            s2 = lap.Sector2Time
            if debug:
                print(f"↪️ Lap {lap.LapNumber}: Sector1Time={s1}, Sector2Time={s2}")
            if pd.isna(s1) or pd.isna(s2):
                raise ValueError("one or both sector times are NaT")
            s1s = s1.total_seconds()
            s2s = (s1 + s2).total_seconds()
            merged.loc[merged["t"] <= s1s,                          "Sector"] = 1
            merged.loc[(merged["t"] > s1s) & (merged["t"] <= s2s),  "Sector"] = 2
        except Exception as e:
            if debug:
                print(f"⚠️ Sector tagging FAILED on lap {lap.LapNumber}: {e}")

        # 8) mark corner area windows around each apex
        merged["CornerArea"] = 0
        for cid, apex in corner_dists.items():
            mask = merged["Distance"].between(apex - corner_window,
                                              apex + corner_window)
            merged.loc[mask, "CornerArea"] = int(cid)

        # 9) compute distance to next/previous corners
        sorted_apex = sorted(corner_dists.items(), key=lambda x: x[1])
        apex_ids = np.array([cid for cid,_ in sorted_apex])
        apex_ds = np.array([   d for _,d in sorted_apex])
        dists = merged["Distance"].to_numpy()
        nxt = np.searchsorted(apex_ds, dists, side="right")
        prv = nxt - 1

        merged["DistanceToNextCorner"] = np.nan
        merged["DistanceFromPreviousCorner"] = np.nan
        merged["NextCorner"] = np.nan
        merged["PreviousCorner"] = np.nan

        valid_n = nxt < len(apex_ds)
        valid_p = prv >= 0

        merged.loc[valid_n, "DistanceToNextCorner"] = apex_ds[nxt[valid_n]] - dists[valid_n]
        merged.loc[valid_p, "DistanceFromPreviousCorner"] = dists[valid_p] - apex_ds[prv[valid_p]]
        merged.loc[valid_n, "NextCorner"] = [int(apex_ids[i]) for i in nxt[valid_n]]
        merged.loc[valid_p, "PreviousCorner"] = [int(apex_ids[i]) for i in prv[valid_p]]

        # 10) inject lap summary metrics and metadata fields
        for fld in ("Stint","LapNumber","FreshTyre",
                    "Sector1Time","Sector2Time","Sector3Time",
                    "SpeedI1","SpeedI2","SpeedFL","SpeedST"):
            merged[fld] = getattr(lap, fld, pd.NaT if "Time" in fld else np.nan)

        # 11) driver & lap metadata
        merged['DriverNumber'] = lap.DriverNumber
        merged['Driver'] = lap.Driver
        merged['Team'] = lap.Team
        merged['Year'] = lap.session.date.year
        merged['EventName'] = lap.session.event.EventName
        merged['SessionName'] = lap.session.name
        merged['Location'] = lap.session.event.Location

        # 12) reorder exactly
        cols = [
            "DriverNumber","Driver","Team","Stint","LapNumber","Time",
            "Speed","RPM","nGear","Throttle","Brake","DRS",
            "Distance","RelativeDistance","Sector",
            "Sector1Time","Sector2Time","Sector3Time",
            "CornerArea","DistanceToNextCorner","DistanceFromPreviousCorner",
            "NextCorner","PreviousCorner",
            "SpeedI1","SpeedI2","SpeedFL","SpeedST",
            "FreshTyre",
            "Year","EventName","SessionName","Location"
        ]
        
        return merged[cols]

    except Exception as e:
        if debug:
            print(f"⚠️ get_detailed_lap_telemetry failed on lap {lap.LapNumber}: {e}")
        return pd.DataFrame()


def _build_detailed_telemetry(
    session,
    debug: bool = False
) -> pd.DataFrame:
    """
    Compose a full-session detailed telemetry DataFrame by processing all valid laps.

    Filters out pit laps and laps lacking complete sector or car data.

    Args:
        session: Loaded FastF1 session.
        debug: If True, passes debug flag to lap-level function.

    Returns:
        Concatenated DataFrame of all laps' detailed telemetry.
    """
    # 1) compute apex distances just once
    corner_dists = get_corner_area(session)

    # 2) pull all non-pit laps
    laps = session.laps.pick_wo_box()

    # 3) keep only drivers with car_data available
    valid = set(session.car_data.keys())
    laps = laps[laps["DriverNumber"].isin(valid)].copy()

    # 4) require complete sector times
    laps = laps[
        laps["Sector1Time"].notna() &
        laps["Sector2Time"].notna() &
        laps["Sector3Time"].notna()
    ].copy()

    all_data, skipped = [], []

    # 5) iterate
    desc = f"{session.date.year} {session.event.EventName} {session.name}"
    for _, lap in tqdm(laps.iterlaps(), desc=desc, total=len(laps), colour="green"):
        df = get_detailed_lap_telemetry(lap, corner_dists, debug=debug)
        if df.empty:
            skipped.append(lap.LapNumber)
        else:
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def _build_driver_timing_profiles(
    start_year: int,
    end_year: int,
    out_dir: str = "data/driver_timing"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build or load lap-by-lap parquet timing profiles for a range of seasons.

    Scans `out_dir` for existing files, builds missing ones, then loads all.

    Args:
        start_year: First year to include.
        end_year: Last year to include.
        out_dir: Directory to store/read per-session parquet files.

    Returns:
        df_all: DataFrame of every lap’s detailed telemetry.
        df_skipped: DataFrame of sessions that failed during build.
    """
    
    os.makedirs(out_dir, exist_ok=True)

    # Identify already written sessions to skip rebuilds
    existing: set[Tuple[int, str, str]] = set()
    for fn in glob.glob(os.path.join(out_dir, '*.parquet')):
        parts = os.path.splitext(os.path.basename(fn))[0].split('_')
        year = int(parts[0])
        sess = parts[-1]
        event = ' '.join(parts[3:-1])
        existing.add((year, event, sess))

    to_build, skipped = [], []

    # Determine which completed sessions are still missing files
    for year in range(start_year, end_year + 1):
        sched = _official_schedule(year)
        done = sched[sched.Session1DateUtc < datetime.utcnow()]
        for key in _completed_sessions(done, datetime.utcnow()):
            if key not in existing:
                to_build.append(key)

    # Build missing sessions and write parquet files
    for yr, ev_name, sess_label in to_build:
        info = load_session(yr, ev_name, sess_label)
        if info.get('status') != 'ok':
            skipped.append({'year': yr, 'event': ev_name, 'session': sess_label,
                            'reason': info.get('reason', 'load failed')})
            continue
        try:
            df_tmp = _build_detailed_telemetry(info['session'])
            if df_tmp.empty:
                raise ValueError('no valid laps')
        except Exception as e:
            skipped.append({'year': yr, 'event': ev_name, 'session': sess_label,
                            'reason': str(e)})
            continue

        # Tag and write out parquet file for traceability
        sched = _official_schedule(yr)
        sched_row = sched[sched['EventName'] == ev_name]
        round_no = int(sched_row.iloc[0]['RoundNumber']) if not sched_row.empty else 0
        fmt = (sched_row.iloc[0].get('EventFormat') or 'conventional').lower()
        orders = {
            'conventional': ['FP1','FP2','FP3','Q','R'],
            'sprint_shootout': ['FP1','Q','SS','S','R'],
            'sprint_qualifying': ['FP1','SQ','S','Q','R'],
            'sprint': ['FP1','Q','FP2','S','R']
        }
        session_no = orders.get(fmt, orders['conventional']).index(sess_label) + 1 if sess_label in orders.get(fmt, []) else 0
        ev_clean = ev_name.replace(' ', '_')
        fn = f"{yr}_{round_no}_{session_no}_{ev_clean}_{sess_label}.parquet"
        path = os.path.join(out_dir, fn)

        df_tmp['year'] = yr
        df_tmp['event'] = ev_name
        df_tmp['session'] = sess_label
        df_tmp.to_parquet(path, engine='pyarrow', compression='snappy', index=False)

    # Read and concatenate all existing parquets
    files = glob.glob(os.path.join(out_dir, '*.parquet'))
    if not files:
        return pd.DataFrame(), pd.DataFrame(skipped)

    df_all = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df_all, pd.DataFrame(skipped)
