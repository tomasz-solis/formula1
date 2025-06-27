from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


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

        ratio = throttle_points.loc[
            (throttle_points.nextThrottle < max_throttle_threshold) |
            (throttle_points.nextThrottle.isna())
        ]['FTRelative'].sum()

        result = pd.DataFrame([{
            'grand_prix': gp_name,
            'location': location,
            'driver': driver,
            'ratio': ratio,
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


def estimate_tire_degradation(session, driver):
    """
    Estimate tire degradation from lap time evolution in latest stint.

    Returns:
    - dict with degradation slope
    """
    try:
        laps = session.laps.pick_drivers([driver])
        stints = laps[laps['Stint'] == laps['Stint'].max()]

        if len(stints) < 2:
            return None

        stints = stints.sort_values('LapNumber')
        X = stints['LapNumber'].values.reshape(-1, 1)
        y = stints['LapTime'].dt.total_seconds().values

        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]

        return {
            'driver': driver,
            'compound': stints.iloc[0]['Compound'],
            'degradation_slope': slope
        }
    except Exception as e:
        print(f"⚠️ Tire degradation calc failed for {driver}: {e}")
        return None


def count_drs_activations(session,
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

    try:
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

    except Exception as e:
        print(f"⚠️ DRS counting failed for {driver}: {e}")
        return np.nan


def braking_intensity(session, driver, braking_drop_kmh: int = 30):
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
    try:
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
    except Exception as e:
        print(f"⚠️ Braking-intensity failed for {driver}: {e}")
        return None


def get_all_driver_features(
    session,
    year: int | None          = None,
    session_name: str | None  = None,
    *,
    throttle_ratio_min: float = 0.40,
    throttle_ratio_max: float = 0.85,
    braking_drop_kmh:  int    = 30,
) -> pd.DataFrame:
    """
    Build a feature row **per driver** with

    • full-throttle ratio        • tyre-degradation slope  
    • DRS activation count       • braking max / mean decel (-g)  
    plus weather & tyre info provided by ``get_driver_max_throttle_ratio``.

    Parameters
    ----------
    session : fastf1.core.Session
        *Loaded* FastF1 session (``session.load(...)`` already called).
    year, session_name : str | int, optional
        Tags copied into the output, useful for merges later on.
    throttle_ratio_min / max : float
        Outlier filter for unrealistic full-throttle ratios.
    braking_drop_kmh : int
        Δ km/h that qualifies as a “heavy-brake” sample.

    Returns
    -------
    pandas.DataFrame
        One row per driver with all numeric features.
    """
    if session.laps.empty:
        print(f"⚠️ No laps in session {session.event['EventName']}")
        return pd.DataFrame()

    event_name  = session.event["EventName"]
    location    = session.event["Location"]     
    driver_ids  = session.laps["Driver"].unique()
    rows: List[Dict] = []

    for drv in driver_ids:
        try:
            # ── base telemetry + tyre + weather
            base_df, _ = get_driver_max_throttle_ratio(
                session, drv,
                season=year,
                session_name=session_name,
            )
            if base_df is None:
                continue          # telemetry missing for this driver

            base = base_df.iloc[0].to_dict()

            # ── enrichment features
            degr  = estimate_tire_degradation(session, drv)
            brake = braking_intensity(session, drv, braking_drop_kmh=braking_drop_kmh)
            drs   = count_drs_activations(session, drv)

            if degr:
                base.update(degr)
            if brake:
                base.update(brake)

            base["drs_activations"] = drs
            base["year"]            = year
            base["session"]         = session_name
            base["event"]           = event_name
            base["location"]        = location

            rows.append(base)

        except Exception as e:
            print(f"⚠️ Skipped {drv} in {session_name}: {e}")

    df = pd.DataFrame(rows)

    # ── outlier filter for throttle ratio
    if not df.empty:
        mask_out = (df["ratio"] < throttle_ratio_min) | (df["ratio"] > throttle_ratio_max)
        df = df.loc[~mask_out]

    return df.reset_index(drop=True)


def merge_driver_features_with_session(driver_data, circuit_data):
    """
    Merge driver-level features with session-level circuit characteristics.

    Returns:
    - DataFrame with enriched training samples per driver-session
    """
    return pd.merge(
        driver_data,
        circuit_data,
        how='left',
        on=['year', 'event', 'session']
    )
