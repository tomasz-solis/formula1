"""
Utilities focused on circuit geometry & track‑level analytics.
"""
import numpy as np, pandas as pd
import fastf1 as ff1
from fastf1.ergast import Ergast
from typing import List
from datetime import datetime
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from .general_utils import (_official_schedule, _session_list,
                            _session_date_col, load_session, get_weather_info)

# ── Circuit lookup helpers ────────────────────────────────────────────────────
def get_elevation(latitude: float, 
                  longitude: float,
                  timeout: int = 10):
    from .general_utils import get_elevation
    return get_elevation(latitude, longitude, timeout)

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
    for year in range(start_year, end_year + 1):
        df = get_circuits(year)
        all_rows.append(df)
    
    full_df = pd.concat(all_rows, ignore_index=True)
    
    # Keep only the first unique occurrence per circuit
    deduped = full_df.drop_duplicates(subset=['circuitName'], keep='first').reset_index(drop=True)
    
    return deduped


#  Track feature extraction ─────────────────────────────────────────────────
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

# Corners

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
        session.load()
        lap = session.laps.pick_fastest()
        pos = lap.get_pos_data().copy()
        car = lap.get_car_data().add_distance().copy()
        corners = session.get_circuit_info().corners.copy()

        if pos.empty or car.empty or corners.empty:
            raise ValueError("Missing required telemetry or circuit data.")

        # Convert Time to seconds
        pos['Time_s'] = pos['Time'].dt.total_seconds()
        car['Time_s'] = car['Time'].dt.total_seconds()

        # Nearest merge using merge_asof
        merged = pd.merge_asof(
            pos.sort_values("Time_s"),
            car[["Time_s", "Speed", "Distance"]].sort_values("Time_s"),
            on="Time_s",
            direction="nearest"
        )

        # KD-tree to map corners to nearest telemetry point
        tree = cKDTree(merged[['X', 'Y']].dropna().values)
        corner_coords = corners[['X', 'Y']].dropna().values
        distances, indices = tree.query(corner_coords, k=1)

        matched = merged.iloc[indices].reset_index(drop=True)
        matched = matched.rename(columns={"Distance": "DriverDistance"})  # avoid conflict
        corners = corners.reset_index(drop=True)
        corners = pd.concat([corners, matched[['Speed', 'DriverDistance']]], axis=1)

        # Classify by speed
        corners['corner_type'] = pd.cut(
            corners['Speed'],
            bins=[0, low_thresh, med_thresh, 1000],
            labels=['slow', 'medium', 'fast'],
            include_lowest=True
        )
        counts = corners['corner_type'].value_counts().to_dict()

        # Chicane detection (distance between consecutive corners)
        corners = corners.sort_values(by='DriverDistance')
        corners['DistanceFromPrev'] = corners['DriverDistance'].diff().fillna(9999)
        chicanes = (corners['DistanceFromPrev'] < 200).sum()

        return {
            'slow_corners': counts.get('slow', 0),
            'medium_corners': counts.get('medium', 0),
            'fast_corners': counts.get('fast', 0),
            'chicanes': chicanes#,
            #'corner_details': corners.reset_index(drop=True)
        }
    
    except Exception as e:
        print(f"⚠️ Failed to compute corner profile: {e}")
        return None

# DRS

        
#  Higher‑level profiling pipelines ─────────────────────────────────────────
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

                # 2-c  telemetry-derived metrics - DRS zone to be added later

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
                        **tmet, **cmet, **wmet,
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


def _build_circuit_profile_df(
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
    # 1) metadata
    meta = get_all_circuits(start_year, end_year)

    # 2) assemble every (year, event_name, session_name) to process
    tasks: list[tuple[int, str, str]] = []
    for yr in range(start_year, end_year + 1):
        sched = _official_schedule(yr)
        past  = sched[sched.Session1DateUtc < datetime.utcnow()]
        for _, ev in past.iterrows():
            ev_name = ev.EventName
            fmt     = str(ev.EventFormat).lower()
            for sess in _session_list(fmt):
                col     = _session_date_col(sess)
                sess_dt = getattr(ev, col, None)
                if sess_dt and sess_dt <= datetime.utcnow():
                    # honor only_specific if set
                    if only_specific is None or (ev_name, sess) in only_specific:
                        tasks.append((yr, ev_name, sess))

    prof_all, skip_all = [], []

    # 3) single tqdm over *sessions*
    for yr, ev_name, sess in tqdm(
        tasks,
        desc="Sessions", # or desc="Race weekends",
        unit="session", # or unit="weekend",
        ncols=80
    ):
        # delegate to your per‐season pipeline, but restrict to exactly this one session
        df_sess, df_skip = build_profiles_for_season(
            year=yr,
            circuit_metadata=meta,
            only_specific={(ev_name, sess)}
        )
        prof_all.append(df_sess)
        skip_all.append(df_skip)

    # 4) stitch results back together
    df_profiles = pd.concat(prof_all, ignore_index=True) if prof_all else pd.DataFrame()
    df_skipped  = pd.concat(skip_all, ignore_index=True) if skip_all else pd.DataFrame()

    print(f"✅ Done: {len(df_profiles)} sessions parsed, {len(df_skipped)} skipped.")
    return df_profiles, df_skipped

# ── Clustering & viz helpers ─────────────────────────────────────────────────
def fit_track_clusters(
    df_profiles: pd.DataFrame,
    group_cols: list[str] = ['event','year'],
    feat_cols: list[str] = None,
    scaler=None,
    clusterer=None,
    do_pca: bool = False,
    n_components: int = 2
) -> tuple[pd.DataFrame, Pipeline]:
    """
    Fit clustering (and optional PCA) per track.
    Returns the per-track cluster assignments and the fitted pipeline.

    Parameters:
    - df_profiles: session-level data
    - group_cols: columns to define each “track” group
    - feat_cols: numeric columns to use for clustering
    - scaler: any scaler (Default: StandardScaler)
    - clusterer: any clustering estimator (Default: KMeans(n_clusters=5))
    - do_pca: whether to include PCA in the pipeline
    Returns:
      • track_profile: one row per group with PC dims + cluster labels  
      • pipe: the fitted sklearn Pipeline (so you can call pipe.predict on new data)
    """
    # Determine features
    feat_cols = feat_cols or df_profiles.select_dtypes(include='number').columns.tolist()
    # Aggregate per track
    track_features = (
        df_profiles
        .groupby(group_cols)[feat_cols]
        .mean()
        .reset_index()
    )
    X = track_features[feat_cols]

    # Build pipeline
    steps = [
        ('imputer', SimpleImputer()),
        ('scaler', scaler or StandardScaler())
    ]
    if do_pca:
        steps.append(('pca', PCA(n_components=n_components, random_state=42)))
    steps.append(('cluster', clusterer or KMeans(n_clusters=5, random_state=42)))
    pipe = Pipeline(steps)

    # Fit clusters
    labels = pipe.fit_predict(X)
    track_profile = track_features.copy()
    track_profile['cluster'] = labels.astype(str)

    # PCA coords if requested
    if do_pca:
        X_imp = pipe.named_steps['imputer'].transform(X)
        X_scl = pipe.named_steps['scaler'].transform(X_imp)
        pcs = pipe.named_steps['pca'].transform(X_scl)
        track_profile[['PC1', 'PC2']] = pcs

    return track_profile, pipe

    
def plot_cluster_radar(df_profiles: pd.DataFrame, categories: list[str], cluster_col: str = 'cluster', normalize: bool = True):
    """
    Build and display a Plotly radar chart where each cluster's mean feature values
    are plotted around a circle. Optionally normalizes each feature to [0,1].

    Parameters:
    -------------
    df_profiles : pd.DataFrame
        DataFrame containing individual samples with cluster assignments.
    categories : list[str]
        List of column names to include in the radar chart.
    cluster_col : str
        Name of the column in df_profiles holding cluster labels.
    normalize : bool
        If True, scales each feature across clusters to the [0,1] range.
    """
    # aggregate
    cluster_norm = (
        df_profiles
        .groupby(cluster_col)[categories]
        .mean()
        .apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
        .reset_index()
    ) if normalize else (
        df_profiles
        .groupby(cluster_col)[categories]
        .mean()
        .reset_index()
    )

    fig = go.Figure()
    for _, row in cluster_norm.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[categories].tolist(),
            theta=categories,
            fill='toself',
            name=f'Cluster {row[cluster_col]}'
        ))
    # set range based on normalize flag
    rrange = [0,1] if normalize else None
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=rrange)),
        title="Driving Style Radar per Cluster",
        showlegend=True
    )
    fig.show()