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
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


#  Circuit lookup helpers
def get_elevation(latitude: float,
                  longitude: float,
                  timeout: int = 10) -> float:
    """
    Fetch elevation data via general_utils.get_elevation wrapper.

    Args:
        latitude: GPS latitude in degrees.
        longitude: GPS longitude in degrees.
        timeout: Request timeout in seconds.

    Returns:
        Elevation in meters.
    """
    from .general_utils import get_elevation as _get_elev
    return _get_elev(latitude, longitude, timeout)


def get_circuits(season: int) -> pd.DataFrame:
    """
    Retrieve basic circuit metadata including location and altitude.

    Args:
        season: Year of the F1 season.

    Returns:
        DataFrame with columns: circuitName, location, country, lat, lon, altitude.
    """
    ergast = Ergast()
    racetracks = ergast.get_circuits(season)
    results = []

    for name in racetracks.circuitName:
        try:
            row = racetracks[racetracks.circuitName == name].iloc[0]
            # Extract geolocation info
            lat = row['lat']; lon = row['long']
            locality = row['locality']; country = row['country']
            # Lookup altitude via external API
            altitude = get_elevation(lat, lon)
            results.append({
                'circuitName': name,
                'location': locality,
                'country': country,
                'lat': lat,
                'lon': lon,
                'altitude': altitude
            })
        except Exception as e:
            log.warning(f"Failed to get altitude for {name}: {e}")
            continue  # skip problematic circuits

    return pd.DataFrame(results)


def get_all_circuits(start_year: int = 2020,
                     end_year: int = 2025) -> pd.DataFrame:
    """
    Aggregate unique circuits across a range of seasons.

    Args:
        start_year: First season year (inclusive).
        end_year: Last season year (inclusive).

    Returns:
        DataFrame of unique circuits with metadata.
    """
    dfs = []
    for year in range(start_year, end_year + 1):
        df = get_circuits(year)
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    # Deduplicate by circuitName, keep first occurrence
    return full.drop_duplicates(subset=['circuitName'], keep='first').reset_index(drop=True)


#  Track feature extraction 
def extract_track_metrics(session) -> Optional[Dict[str, float]]:
    """
    Extract average speed, top speed, and braking profile from a session.

    Args:
        session: Loaded FastF1 session.

    Returns:
        Dict with keys avg_speed, top_speed, braking_events,
        low_pct, med_pct, high_pct, or None if failed.
    """
    try:
        if session.laps.empty:
            return None
        lap = session.laps.pick_fastest()
        tel = lap.get_car_data().add_distance()
        # Compute speed deltas to identify heavy braking
        tel['delta_speed'] = tel['Speed'].diff()
        heavy_brakes = tel['delta_speed'] < -30
        return {
            'avg_speed': tel['Speed'].mean(),
            'top_speed': tel['Speed'].max(),
            'braking_events': int(heavy_brakes.sum()),
            'low_pct': float((tel['Speed'] < 120).mean()),
            'med_pct': float(((tel['Speed'] >= 120) & (tel['Speed'] < 200)).mean()),
            'high_pct': float((tel['Speed'] >= 200).mean())
        }
    except Exception as e:
        log.warning(f"⚠️ Failed to extract track metrics: {e}")
        return None


def get_valid_lap_with_pos(session, max_attempts: int = 5):
    """
    Find a lap with valid positional data (X/Y).

    Args:
        session: FastF1 session object.
        max_attempts: Number of quick laps to try.

    Returns:
        A Lap object with position data, or None if not found.
    """
    fast_laps = session.laps.pick_quicklaps().sort_values('LapTime')
    for i, lap in enumerate(fast_laps.itertuples()):
        if i >= max_attempts:
            break
        drv_num = lap.DriverNumber
        try:
            _ = session.pos_data[drv_num]
            return session.laps.loc[lap.Index]
        except KeyError:
            continue  # no pos data for this driver
        except Exception as e:
            log.warning(f"Skipping lap for driver {drv_num}: {e}")
    log.warning("⚠️ No valid lap with position data found.")
    return None


def get_circuit_corner_profile(
    session,
    low_thresh: int = 100,
    med_thresh: int = 170
) -> Dict[str, int]:
    """
    Detect corners and categorize by entry speed; estimate chicanes.

    Args:
        session: Loaded FastF1 session.
        low_thresh: Max speed for slow corners (km/h).
        med_thresh: Max speed for medium corners (km/h).

    Returns:
        Dict with slow_corners, medium_corners, fast_corners, chicanes.

    Raises:
        ValueError: If extraction fails.
    """
    
    try:
        session.load(telemetry=True)
        lap = get_valid_lap_with_pos(session)
        if lap is None:
            event = session.event.get("EventName", "Unknown")
            name = getattr(session, "name", "Unknown")
            log.warning(f"⚠️ Skipping {event} {name}: No valid lap with pos data")
            raise RuntimeError("⚠️ No valid lap with position data")
        pos = lap.get_pos_data().copy()
        car = lap.get_car_data().add_distance().copy()
        corners = session.get_circuit_info().corners.copy()

        if pos.empty or car.empty or corners.empty:
            raise ValueError("⚠️ Missing required telemetry or circuit data.")

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
        chicanes = (corners['DistanceFromPrev'] < 100).sum()

        return {
            'slow_corners': counts.get('slow', 0),
            'medium_corners': counts.get('medium', 0),
            'fast_corners': counts.get('fast', 0),
            'chicanes': chicanes
        }
    
        
    except Exception as e:
        event = session.event.get("EventName", "Unknown")
        name = getattr(session, "name", "Unknown")
        raise ValueError(f"⚠️ Failed to compute corner profile: {event} {name} – {e}")

        
#  Higher‑level profiling pipelines
def build_profiles_for_season(
    year: int,
    circuit_metadata: pd.DataFrame,
    *,
    only_specific: Optional[set[Tuple[str,str]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build circuit-performance profiles for one F1 season.

    Args:
        year: Season year.
        circuit_metadata: DataFrame from get_all_circuits.
        only_specific: Optional set of (event, session) to process.

    Returns:
        Tuple of (profiles_df, skipped_df).
    """
    records: list[dict] = []
    skipped: list[dict] = []

    try:
        from .general_utils import _official_schedule, _session_list, _session_date_col, load_session, get_weather_info
        sched = _official_schedule(year)
        past  = sched[sched.Session1DateUtc < datetime.utcnow()]
    except Exception as e:
        skipped.append(
            {"year": year, "event": None, "session": None, "reason": str(e)}
        )
        return pd.DataFrame(), pd.DataFrame(skipped)
        
    # Iterate events and sessions
    for _, ev in tqdm(past.iterrows(), total=len(past), desc=f"{year} events", leave=True,colour="blue"):
        ev_name = ev["EventName"]
        raw_fmt = ev["EventFormat"]
        location = ev["Location"] 
        fmt = str(raw_fmt.item() if isinstance(raw_fmt, pd.Series) else raw_fmt)
        fmt = fmt.lower() if pd.notnull(fmt) else "conventional"
        date_map = _session_date_col(fmt, ev)
        sessions = _session_list(fmt)

        for sess in tqdm(sessions, desc=f"{year} {ev_name}", leave=True, colour="black"):
            if only_specific and (ev_name, sess) not in only_specific:
                continue
            try:
                s_info = load_session(year, ev_name, sess)
                if s_info["status"] == "error":
                    raise ValueError(s_info["reason"])

                tele_src = s_info["source"] # fastf1 / openf1
                laps = s_info["laps"]
                session = s_info["session"]  # FastF1.Session or None

                if laps is None or laps.empty:
                    raise ValueError("Lap data missing")

                # Estimate lap length
                if session is not None: # FastF1
                    try:
                        dist = fast1_sess.laps.pick_fastest().get_car_data().add_distance()['Distance'].max()
                    except Exception:
                        lap_len = np.nan
                else: # OpenF1 fallback
                    lap_len = (
                        laps.groupby("driver_number")["lap_distance"].max().max()
                    )

                try:
                    tmet = extract_track_metrics(session) if session else None
                except Exception as e:
                    raise ValueError(f"⚠️ Failed to extract telemetry metrics: {e}")
                
                try:
                    cmet = get_circuit_corner_profile(session) if session else None
                except Exception as e:
                    raise ValueError(f"⚠️ {e}")  # already descriptive
                
                try:
                    wmet = get_weather_info(session, year, ev_name, sess)
                except Exception as e:
                    raise ValueError(f"⚠️ Failed to fetch weather data: {e}")

                if not tmet:
                    raise ValueError("⚠️ Missing telemetry metrics")

                try:
                    alt = (
                        circuit_metadata
                        .loc[circuit_metadata["location"] == location, "altitude"]
                        .iloc[0]
                    )
                except IndexError:
                    alt = np.nan

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
    end_year: int,
    *,
    only_specific: Optional[Dict[int, set[Tuple[str,str]]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build circuit profiles over multiple seasons.

    Args:
        start_year: First year (inclusive).
        end_year: Last year (inclusive).
        only_specific: Optional mapping of year to sessions to include.

    Returns:
        Tuple of (all_profiles_df, all_skipped_df).
    """
    all_profiles, all_skipped = [], []

    for year in range(start_year, end_year + 1):
        tqdm.write(f"\n📅 Building profiles for season {year}...")
        circuit_metadata = get_all_circuits(year)

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
    profiles = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()
    skipped = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()

    # 4) log any skips
    if not skipped.empty:
        tqdm.write("\n⚠️ Skipped sessions:")
        for _, row in skipped.iterrows():
            tqdm.write(f"⚠️  - {row['year']} {row['event']} {row['session']} – {row['reason']}")

    return profiles, skipped
    

def fit_track_clusters(
    df_profiles: pd.DataFrame,
    group_cols: List[str] = ['event','year'],
    feat_cols: Optional[List[str]] = None,
    scaler=None,
    clusterer=None,
    do_pca: bool = False,
    n_components: int = 2
) -> Tuple[pd.DataFrame, Pipeline]:
    """
    Cluster tracks based on performance metrics and optionally project via PCA.

    Args:
        df_profiles: Session-level feature DataFrame.
        group_cols: Columns defining each track group.
        feat_cols: Numeric features for clustering (defaults to all numeric).
        scaler: Preprocessing scaler (default StandardScaler).
        clusterer: Clustering estimator (default KMeans(n_clusters=5)).
        do_pca: Whether to include PCA step.
        n_components: Number of PCA components if used.

    Returns:
        track_profile: DataFrame with cluster labels (and PC coords if PCA).
        pipeline: Fitted sklearn Pipeline.
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

    
def plot_cluster_radar(
    df_profiles: pd.DataFrame,
    categories: List[str],
    cluster_col: str = 'cluster',
    normalize: bool = True
) -> go.Figure:
    """
    Create a radar chart comparing clusters on selected features.

    Args:
        df_profiles: DataFrame with cluster labels.
        categories: Features to plot.
        cluster_col: Column name for clusters.
        normalize: Whether to scale features to [0,1].

    Returns:
        Plotly Figure object.
    """
    
    # Aggregate mean feature per cluster
    agg = df_profiles.groupby(cluster_col)[categories].mean()
    if normalize:
        agg = agg.apply(lambda col: (col - col.min())/(col.max()-col.min()), axis=0)
    agg = agg.reset_index()
    fig = go.Figure()
    for _, row in agg.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[categories].tolist(), theta=categories,
            fill='toself', name=f"Cluster {row[cluster_col]}"
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1] if normalize else None)),
        title="Driving Style Radar per Cluster"
    )
    return fig


def create_pca_for_n_clusters(
    circuits: pd.DataFrame,
    clusters: int,
    feat_cols: List[str]
) -> Tuple[px.scatter, go.Figure]:
    """
    Perform PCA and KMeans clustering, returning scatter and radar plots.

    Args:
        circuits: DataFrame with track metrics and identifiers.
        clusters: Number of clusters for KMeans.
        feat_cols: Feature columns for analysis.

    Returns:
        Tuple of (scatter_plot, radar_plot).
    """
    # Fit on per-track metrics
    track_profile, pipeline = fit_track_clusters(
        circuits,
        group_cols=['track_id'],
        feat_cols=feat_cols,
        do_pca=True,
        clusterer=KMeans(n_clusters=clusters, random_state=42),
    )

    # Map cluster back to original circuits
    key = track_profile.set_index('track_id')['cluster']
    circuits['cluster'] = (
        circuits['event'].astype(str) + '_' + circuits['year'].astype(str)
    ).map(key)

    # Prepare ordered cluster labels
    cluster_vals = sorted(track_profile['cluster'].unique(), key=int)

    # Scatter via PCA dims
    scatter_plot = px.scatter(
        track_profile,
        x='PC1',
        y='PC2',
        color='cluster',
        hover_data=['track_id'],
        title=f'PCA view for k={clusters}',
        category_orders={'cluster': cluster_vals},
    )

    # Identify top varying features for radar
    stats = track_profile.groupby('cluster')[feat_cols].mean()
    spreads = (stats.max() - stats.min()).sort_values(ascending=False)
    top_features = spreads.head(8).index.tolist()
    
    # Radar chart
    radar_plot = plot_cluster_radar(
        track_profile,
        categories=top_features,
        cluster_col='cluster',
        normalize=True,
    )

    return scatter_plot, radar_plot