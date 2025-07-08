"""
Circuit utilities for geometry and track-level analytics.

Includes functions to fetch circuit list, extract metrics, build profiles, cluster analysis, and plot radar charts.
"""

# Library imports

import logging
import os
import numpy as np
import pandas as pd
import fastf1 as ff1
import plotly.graph_objects as go
import plotly.express as px

from fastf1.ergast import Ergast
from typing import List
from datetime import datetime
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional

log = logging.getLogger(__name__)

#  Circuit lookup helpers
def get_elevation(latitude: float, 
                  longitude: float,
                  timeout: int = 10):
    """
    Wrapper around general_utils.get_elevation to fetch elevation data.

    Parameters:
        latitude (float): GPS latitude.
        longitude (float): GPS longitude.
        timeout (int): Request timeout in seconds.

    Returns:
        float: Elevation in meters.
    """
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
    """
    Aggregate circuit data across multiple seasons.

    Parameters:
        years (List[int]): List of championship years.

    Returns:
        pandas.DataFrame: Combined circuits for all years.
    """
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

def get_valid_lap_with_pos(session, max_attempts=5):
    """
    Try to get a valid lap with position data.
    
    Parameters:
        session: FastF1 session object
        max_attempts: number of fastest laps to try before giving up
        
    Returns:
        lap (Lap): A single Lap object with valid position data, or None
    """
    fast_laps = session.laps.pick_quicklaps().sort_values("LapTime")

    for i, lap in enumerate(fast_laps.itertuples()):
        if i >= max_attempts:
            break

        drv = lap.DriverNumber
        try:
            _ = session.pos_data[drv]
            return session.laps.loc[lap.Index]
        except KeyError:
            continue
        except Exception as e:
            log.warning(f"Skipping lap for driver {drv} → {e}")
            continue

    event = session.event.get("EventName", "Unknown Event")
    name  = getattr(session, "name", "Unknown Session")
    log.warning(f"⚠️ No valid lap with position data in {event} {name}")
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
        session.load()
        lap = get_valid_lap_with_pos(session)
        if lap is None:
            event = session.event.get("EventName", "Unknown")
            name = getattr(session, "name", "Unknown")
            log.warning(f"⚠️ Skipping {event} {name}: No valid lap with pos data")
            raise RuntimeError("No valid lap with position data")
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
        event = session.event.get("EventName", "Unknown")
        name = getattr(session, "name", "Unknown")
        raise ValueError(f"Failed to compute corner profile: {event} {name} – {e}")


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
    #print(f"[TRACE] Filtering to only_specific = {only_specific}")

    records: list[dict] = []
    skipped: list[dict] = []

    # 1 ─ get schedule
    try:
        from .general_utils import _official_schedule
        sched = _official_schedule(year)
        past  = sched[sched.Session1DateUtc < datetime.utcnow()]
    except Exception as e:
        skipped.append(
            {"year": year, "event": None, "session": None, "reason": str(e)}
        )
        return pd.DataFrame(), pd.DataFrame(skipped)

    # 2 ─ iterate events/sessions
    from .general_utils import _session_list, _session_date_col, load_session, get_weather_info
    for _, ev in tqdm(past.iterrows(), total=len(past), desc=f"{year} events", leave=True,colour="blue"):
        ev_name = ev["EventName"]
        raw_fmt = ev["EventFormat"]
        location = ev["Location"] 
        fmt = str(raw_fmt.item() if isinstance(raw_fmt, pd.Series) else raw_fmt)
        fmt = fmt.lower() if pd.notnull(fmt) else "conventional"
        date_map = _session_date_col(fmt, ev)

        sessions = _session_list(fmt)

        for sess in tqdm(sessions, desc=f"{year} {ev_name}", leave=False, colour="black"):
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

                try:
                    tmet = extract_track_metrics(session) if session else None
                except Exception as e:
                    raise ValueError(f"Failed to extract telemetry metrics: {e}")
                
                try:
                    cmet = get_circuit_corner_profile(session) if session else None
                except Exception as e:
                    raise ValueError(f"{e}")  # already descriptive
                
                try:
                    wmet = get_weather_info(session, year, ev_name, sess)
                except Exception as e:
                    raise ValueError(f"Failed to fetch weather data: {e}")

                if not tmet:
                    raise ValueError("Missing telemetry metrics")

                # 2-d  altitude lookup
                try:
                    alt = (
                        circuit_metadata
                        .loc[circuit_metadata["location"] == location, "altitude"]
                        .iloc[0]
                    )
                except IndexError:
                    alt = np.nan

                # 2-e  assemble row
                records.append(
                    {
                        "year": year,
                        "event": ev_name,
                        "location": location,
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
                        "reason": f"{type(e).__name__}: {e}",
                    }
                )

    return pd.DataFrame(records), pd.DataFrame(skipped)
    

def _build_circuit_profile_df(
    start_year: int,
    end_year:   int,
    *,
    only_specific: dict[int, set[tuple[str, str]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build circuit profiles for a range of seasons.

    Parameters
    ----------
    start_year : int
        First year to include (e.g. 2023).
    end_year : int
        Last year to include (inclusive).
    only_specific : dict of {year: set of (event, session)}, optional
        If provided, only those exact sessions will be profiled.

    Returns
    -------
    df_profiles, df_skipped : (DataFrame, DataFrame)
    """
    all_profiles = []
    all_skipped  = []

    for year in range(start_year, end_year + 1):
        tqdm.write(f"\n📅 Building profiles for season {year}...")

        # 1) load your track metadata
        circuit_metadata = get_all_circuits(year)

        # 2) delegate entirely to build_profiles_for_season,
        #    handing it only_specific[year] (or None) so that it
        #    itself skips any session not in that set.
        if only_specific and year in only_specific:
            df, skipped = build_profiles_for_season(
                year,
                circuit_metadata,
                only_specific=only_specific[year]
            )
        else:
            df, skipped = build_profiles_for_season(year, circuit_metadata)

        all_profiles.append(df)
        all_skipped.append(skipped)

    # 3) concatenate
    df_profiles = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()
    df_skipped  = pd.concat(all_skipped,  ignore_index=True) if all_skipped  else pd.DataFrame()

    # 4) log any skips
    if not df_skipped.empty:
        tqdm.write("\n⚠️ Skipped sessions:")
        for _, row in df_skipped.iterrows():
            tqdm.write(f"⚠️  - {row['year']} {row['event']} {row['session']} – {row['reason']}")

    return df_profiles, df_skipped
    

# Clustering & viz helpers
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
    Cluster track metrics into driving-style groups.
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
    return fig


def create_pca_for_n_clusters(
    circuits: pd.DataFrame,
    clusters: int,
    feat_cols: List[str],
) -> Tuple[px.scatter, None]:
    """Run PCA + clustering and produce PCA scatter and radar charts.

    Args:
        circuits (pd.DataFrame):
            DataFrame containing track metrics, with at least the columns in `feat_cols`,
            plus 'track_id', 'event' and 'year'.
        clusters (int): Number of KMeans clusters to fit.
        feat_cols (List[str]): List of numeric feature columns to include in PCA.

    Returns:
        Tuple[plotly.graph_objects.Figure, None]:
            - scatter_plot: PCA scatter (PC1 vs. PC2) colored by cluster.
            - radar_plot: Radar chart comparing clusters on their top-8 spread features.
    """
    # 1. Fit PCA + clustering pipeline
    track_profile, pipeline = fit_track_clusters(
        circuits,
        group_cols=['track_id'],
        feat_cols=feat_cols,
        do_pca=True,
        clusterer=KMeans(n_clusters=clusters, random_state=42),
    )

    # 2. Map cluster assignments back to original circuits df
    key = track_profile.set_index('track_id')['cluster']
    circuits['cluster'] = (
        circuits['event'].astype(str) + '_' + circuits['year'].astype(str)
    ).map(key)

    # 3. Prepare ordered cluster labels
    cluster_vals = sorted(track_profile['cluster'].unique(), key=int)

    # 4. PCA scatter plot
    scatter_plot = px.scatter(
        track_profile,
        x='PC1',
        y='PC2',
        color='cluster',
        hover_data=['track_id'],
        title=f'PCA view for k={clusters}',
        category_orders={'cluster': cluster_vals},
    )

    # 5. Identify top-8 most varying features for radar chart
    stats = track_profile.groupby('cluster')[feat_cols].mean()
    spreads = (stats.max() - stats.min()).sort_values(ascending=False)
    top_features = spreads.head(8).index.tolist()
    
    # 6. Radar chart
    radar_plot = plot_cluster_radar(
        track_profile,
        categories=top_features,
        cluster_col='cluster',
        normalize=True,
    )

    return scatter_plot, radar_plot