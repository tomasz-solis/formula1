"""
Per‑driver performance metrics & feature extraction.
"""
from __future__ import annotations
import numpy as np, pandas as pd, fastf1 as ff1
from datetime import datetime
from typing import Dict, List
from sklearn.linear_model import LinearRegression

from .general_utils import load_session, _session_list, _official_schedule

# ── Throttle & tyre degradation ──────────────────────────────────────────────
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

# ── DRS usage ----------------------------------------------------------------
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
        return np.nan                                    # no telemetry at all

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

# ── Braking intensity --------------------------------------------------------
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


# ── Full session driver features --------------------------------------------
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
    drs_map   = count_drs_activations(session, year, session_name)

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
            row["compound"]          = info.get("compound") if info else row.get("compound")
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
        row["year"]     = year
        row["session"]  = session_name
        row["event"]    = session.event["EventName"]
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
