"""
Historical feature engineering for F1 ML models.

Provides functions to compute rolling performance metrics, momentum indicators,
and context-aware features based on driver/team/circuit history.

Features include:
- Driver performance at circuit (previous years)
- Recent form and momentum (last N races)
- Weather-adjusted performance (wet vs dry)
- Team compatibility scores
- Circuit affinity metrics

Example:
    >>> from helpers.historical_features import compute_historical_features
    >>> features = compute_historical_features(driver_profiles, circuit_profiles)
    >>> print(features[['driver', 'circuit_avg_position', 'recent_form']].head())

Author: Tomasz Solis
Date: November 2025
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime


# =============================================================================
# DRIVER HISTORICAL PERFORMANCE
# =============================================================================

def compute_circuit_history(
    driver_profiles: pd.DataFrame,
    lookback_years: int = 3,
    min_sessions: int = 1
) -> pd.DataFrame:
    """
    Compute driver performance history at each circuit.
    
    For each driver-circuit-year combination, calculates historical performance
    at that circuit from previous years. Metrics include average finishing
    position, best result, consistency, and sample size.
    
    Algorithm:
        1. For each driver-circuit-year combo
        2. Look back N years at same circuit
        3. Compute avg/best/std of qualifying position
        4. Count number of historical sessions
        5. Return as new features
    
    Args:
        driver_profiles: DataFrame with columns:
            - year, event, session, driver, qualifying_position (or race_position)
        lookback_years: How many years back to consider (default: 3)
        min_sessions: Minimum sessions required to compute stats (default: 1)
    
    Returns:
        DataFrame with historical features:
        - driver, event, year: Identifiers
        - circuit_avg_position: Mean position at this circuit (previous years)
        - circuit_best_position: Best position at this circuit
        - circuit_position_std: Consistency at this circuit (lower = more consistent)
        - circuit_sessions_count: Number of previous sessions at this circuit
        
    Example:
        >>> history = compute_circuit_history(driver_profiles, lookback_years=3)
        >>> print(history[history['driver'] == 'VER'].head())
             driver           event  year  circuit_avg_position  circuit_sessions_count
        0      VER  Bahrain GP      2024                  2.3                          3
        1      VER  Monaco GP       2024                  1.0                          3
        
    Note:
        - NaN values mean no historical data at this circuit
        - Only includes sessions where driver participated
        - Qualifying and Race sessions weighted equally
    """
    # Prepare data - need position column
    if 'qualifying_position' in driver_profiles.columns:
        position_col = 'qualifying_position'
    elif 'race_position' in driver_profiles.columns:
        position_col = 'race_position'
    else:
        raise ValueError("DataFrame must have 'qualifying_position' or 'race_position'")
    
    # Create copy for manipulation
    df = driver_profiles[['year', 'event', 'driver', position_col]].copy()
    df = df.dropna(subset=[position_col])
    
    results = []
    
    # Group by driver-circuit
    for (driver, event), group in df.groupby(['driver', 'event']):
        group = group.sort_values('year')
        
        for year in group['year'].unique():
            # Get historical data (years before current year)
            historical = group[
                (group['year'] < year) & 
                (group['year'] >= year - lookback_years)
            ]
            
            if len(historical) >= min_sessions:
                results.append({
                    'driver': driver,
                    'event': event,
                    'year': year,
                    'circuit_avg_position': historical[position_col].mean(),
                    'circuit_best_position': historical[position_col].min(),
                    'circuit_position_std': historical[position_col].std(),
                    'circuit_sessions_count': len(historical)
                })
            else:
                # Not enough historical data
                results.append({
                    'driver': driver,
                    'event': event,
                    'year': year,
                    'circuit_avg_position': np.nan,
                    'circuit_best_position': np.nan,
                    'circuit_position_std': np.nan,
                    'circuit_sessions_count': 0
                })
    
    return pd.DataFrame(results)


def compute_recent_form(
    driver_profiles: pd.DataFrame,
    window_size: int = 5,
    include_current: bool = False
) -> pd.DataFrame:
    """
    Compute driver's recent form based on last N races.
    
    Calculates rolling average of finishing positions to capture momentum
    and current performance trajectory. Can optionally include or exclude
    the current race.
    
    Args:
        driver_profiles: DataFrame with year, event, driver, session_date, position
        window_size: Number of previous races to average (default: 5)
        include_current: Whether to include current race in average (default: False)
    
    Returns:
        DataFrame with form metrics:
        - driver, event, year: Identifiers
        - recent_avg_position: Rolling average of last N positions
        - recent_best_position: Best position in last N races
        - form_trend: Linear trend coefficient (negative = improving)
        - races_in_window: Actual number of races in rolling window
        
    Example:
        >>> form = compute_recent_form(driver_profiles, window_size=5)
        >>> print(form[form['driver'] == 'VER'].head())
             driver           event  year  recent_avg_position  form_trend
        0      VER  Monaco GP       2024                 2.4        -0.3  # Improving!
        
    Note:
        - Sorted by session_date to ensure chronological order
        - form_trend < 0 means improving, > 0 means declining
        - NaN for drivers with fewer than window_size races
    """
    # Need session_date for chronological ordering
    if 'session_date' not in driver_profiles.columns:
        raise ValueError("DataFrame must have 'session_date' column")
    
    # Determine position column
    if 'qualifying_position' in driver_profiles.columns:
        position_col = 'qualifying_position'
    elif 'race_position' in driver_profiles.columns:
        position_col = 'race_position'
    else:
        raise ValueError("DataFrame must have position column")
    
    df = driver_profiles[['year', 'event', 'driver', 'session_date', position_col]].copy()
    df = df.dropna(subset=[position_col])
    df['session_date'] = pd.to_datetime(df['session_date'])
    df = df.sort_values(['driver', 'session_date'])
    
    results = []
    
    for driver, group in df.groupby('driver'):
        group = group.reset_index(drop=True)
        
        for idx, row in group.iterrows():
            if include_current:
                # Include current race in window
                window = group.loc[max(0, idx - window_size + 1):idx]
            else:
                # Exclude current race
                if idx == 0:
                    continue  # No history yet
                window = group.loc[max(0, idx - window_size):idx - 1]
            
            if len(window) > 0:
                positions = window[position_col].values
                
                # Compute trend (negative = improving)
                if len(positions) >= 2:
                    x = np.arange(len(positions))
                    trend = np.polyfit(x, positions, 1)[0]
                else:
                    trend = 0.0
                
                results.append({
                    'driver': row['driver'],
                    'event': row['event'],
                    'year': row['year'],
                    'recent_avg_position': positions.mean(),
                    'recent_best_position': positions.min(),
                    'recent_worst_position': positions.max(),
                    'form_trend': trend,
                    'races_in_window': len(positions)
                })
    
    return pd.DataFrame(results)


def compute_weather_performance(
    driver_profiles: pd.DataFrame,
    rain_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Compute driver performance in wet vs dry conditions.
    
    Splits historical performance by weather conditions to identify drivers
    who excel in the rain vs dry. Uses rainfall data to classify sessions.
    
    Args:
        driver_profiles: DataFrame with driver, avg_rainfall, position columns
        rain_threshold: Rainfall (mm/h) above which session is "wet" (default: 0.1)
    
    Returns:
        DataFrame with weather-split performance:
        - driver, year: Identifiers
        - dry_avg_position: Average position in dry conditions
        - wet_avg_position: Average position in wet conditions
        - wet_dry_delta: Difference (wet - dry, negative = better in wet)
        - dry_sessions: Number of dry sessions
        - wet_sessions: Number of wet sessions
        
    Example:
        >>> weather_perf = compute_weather_performance(driver_profiles)
        >>> print(weather_perf[weather_perf['wet_dry_delta'] < -2])  # Better in wet
             driver  year  dry_avg_position  wet_avg_position  wet_dry_delta
        0      HAM  2024               4.2               1.5           -2.7  # Rain master!
        
    Note:
        - Requires 'avg_rainfall' column in driver_profiles
        - NaN for drivers with no wet or dry sessions
        - wet_dry_delta < 0 means better in rain
    """
    if 'avg_rainfall' not in driver_profiles.columns:
        raise ValueError("DataFrame must have 'avg_rainfall' column")
    
    # Determine position column
    if 'qualifying_position' in driver_profiles.columns:
        position_col = 'qualifying_position'
    elif 'race_position' in driver_profiles.columns:
        position_col = 'race_position'
    else:
        raise ValueError("DataFrame must have position column")
    
    df = driver_profiles[['year', 'driver', 'avg_rainfall', position_col]].copy()
    df = df.dropna(subset=[position_col, 'avg_rainfall'])
    
    # Classify as wet or dry
    df['is_wet'] = df['avg_rainfall'] > rain_threshold
    
    results = []
    
    for (driver, year), group in df.groupby(['driver', 'year']):
        dry_sessions = group[~group['is_wet']]
        wet_sessions = group[group['is_wet']]
        
        dry_avg = dry_sessions[position_col].mean() if len(dry_sessions) > 0 else np.nan
        wet_avg = wet_sessions[position_col].mean() if len(wet_sessions) > 0 else np.nan
        
        results.append({
            'driver': driver,
            'year': year,
            'dry_avg_position': dry_avg,
            'wet_avg_position': wet_avg,
            'wet_dry_delta': wet_avg - dry_avg if not (pd.isna(wet_avg) or pd.isna(dry_avg)) else np.nan,
            'dry_sessions': len(dry_sessions),
            'wet_sessions': len(wet_sessions)
        })
    
    return pd.DataFrame(results)


# =============================================================================
# TEAM HISTORICAL PERFORMANCE
# =============================================================================

def compute_team_circuit_performance(
    driver_profiles: pd.DataFrame,
    lookback_years: int = 3
) -> pd.DataFrame:
    """
    Compute team performance history at each circuit.
    
    Aggregates driver results by team to measure team-level competitiveness
    at specific circuits. Useful for identifying team strengths/weaknesses
    by track type.
    
    Args:
        driver_profiles: DataFrame with year, event, team, driver, position
        lookback_years: Years of history to consider (default: 3)
    
    Returns:
        DataFrame with team circuit performance:
        - team, event, year: Identifiers
        - team_circuit_avg_position: Team's average position at circuit
        - team_circuit_best_position: Team's best position at circuit
        - team_circuit_drivers: Number of drivers in sample
        
    Example:
        >>> team_perf = compute_team_circuit_performance(driver_profiles)
        >>> print(team_perf[team_perf['team'] == 'Red Bull Racing'].head())
    """
    if 'team' not in driver_profiles.columns:
        raise ValueError("DataFrame must have 'team' column")
    
    # Determine position column
    if 'qualifying_position' in driver_profiles.columns:
        position_col = 'qualifying_position'
    elif 'race_position' in driver_profiles.columns:
        position_col = 'race_position'
    else:
        raise ValueError("DataFrame must have position column")
    
    df = driver_profiles[['year', 'event', 'team', 'driver', position_col]].copy()
    df = df.dropna(subset=[position_col])
    
    results = []
    
    for (team, event), group in df.groupby(['team', 'event']):
        group = group.sort_values('year')
        
        for year in group['year'].unique():
            historical = group[
                (group['year'] < year) &
                (group['year'] >= year - lookback_years)
            ]
            
            if len(historical) > 0:
                results.append({
                    'team': team,
                    'event': event,
                    'year': year,
                    'team_circuit_avg_position': historical[position_col].mean(),
                    'team_circuit_best_position': historical[position_col].min(),
                    'team_circuit_drivers': historical['driver'].nunique()
                })
    
    return pd.DataFrame(results)


def compute_team_momentum(
    driver_profiles: pd.DataFrame,
    window_size: int = 5
) -> pd.DataFrame:
    """
    Compute team development trajectory over recent races.
    
    Measures whether team is improving or declining based on rolling
    average of both drivers' positions.
    
    Args:
        driver_profiles: DataFrame with year, event, team, session_date, position
        window_size: Number of races for rolling window (default: 5)
    
    Returns:
        DataFrame with team momentum:
        - team, event, year: Identifiers
        - team_recent_avg: Rolling average of team positions
        - team_momentum: Trend coefficient (negative = improving)
        
    Example:
        >>> momentum = compute_team_momentum(driver_profiles)
        >>> improving = momentum[momentum['team_momentum'] < -0.5]
        >>> print(improving[['team', 'event', 'team_momentum']])
    """
    if 'team' not in driver_profiles.columns or 'session_date' not in driver_profiles.columns:
        raise ValueError("DataFrame must have 'team' and 'session_date' columns")
    
    # Determine position column
    if 'qualifying_position' in driver_profiles.columns:
        position_col = 'qualifying_position'
    elif 'race_position' in driver_profiles.columns:
        position_col = 'race_position'
    else:
        raise ValueError("DataFrame must have position column")
    
    df = driver_profiles[['year', 'event', 'team', 'session_date', position_col]].copy()
    df = df.dropna(subset=[position_col])
    df['session_date'] = pd.to_datetime(df['session_date'])
    df = df.sort_values(['team', 'session_date'])
    
    results = []
    
    for team, group in df.groupby('team'):
        # Average both drivers per event
        team_avg = group.groupby(['year', 'event', 'session_date'])[position_col].mean().reset_index()
        team_avg = team_avg.sort_values('session_date').reset_index(drop=True)
        
        for idx, row in team_avg.iterrows():
            if idx == 0:
                continue
            
            window = team_avg.loc[max(0, idx - window_size):idx - 1]
            
            if len(window) >= 2:
                positions = window[position_col].values
                x = np.arange(len(positions))
                trend = np.polyfit(x, positions, 1)[0]
                
                results.append({
                    'team': team,
                    'event': row['event'],
                    'year': row['year'],
                    'team_recent_avg': positions.mean(),
                    'team_momentum': trend
                })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# =============================================================================

def compute_historical_features(
    driver_profiles: pd.DataFrame,
    circuit_profiles: Optional[pd.DataFrame] = None,
    lookback_years: int = 3,
    form_window: int = 5,
    rain_threshold: float = 0.1,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute all historical features for ML model training.
    
    Master function that orchestrates all historical feature computation:
    - Circuit-specific history
    - Recent form and momentum
    - Weather-adjusted performance
    - Team performance at circuit
    - Team development trajectory
    
    Args:
        driver_profiles: Main driver dataset with all sessions
        circuit_profiles: Optional circuit characteristics (track type, etc.)
        lookback_years: Years of history for circuit features (default: 3)
        form_window: Races for recent form calculation (default: 5)
        rain_threshold: Rainfall threshold for wet classification (default: 0.1 mm/h)
        start_year: First year for merging classification targets (required)
        end_year: Last year for merging classification targets (required)
    
    Returns:
        DataFrame with base data + all historical features merged
        
    Example:
        >>> from helpers.general_utils import load_or_build_profiles
        >>> drivers, _ = load_or_build_profiles(2022, 2024, 'driver')
        >>> circuits, _ = load_or_build_profiles(2022, 2024, 'circuit')
        >>> 
        >>> features = compute_historical_features(
        ...     drivers, circuits,
        ...     start_year=2022, end_year=2024
        ... )
        >>> print(features.columns)
        
    Note:
        - Automatically merges with classification targets
        - Handles missing data (NaN for insufficient history)
        - All features computed per driver-event-year combination
    """
    from .general_utils import merge_driver_features_with_targets
    from tqdm import tqdm
    
    if start_year is None or end_year is None:
        raise ValueError("start_year and end_year are required for merging classification targets")
    
    print("🏗️  Computing historical features...")
    
    # ========== FIX: AGGREGATE DRIVER PROFILES FIRST ==========
    print("  📊 Aggregating driver profiles to session level...")
    
    # Define aggregation functions for telemetry/weather features
    agg_functions = {
        'max_throttle_ratio': 'mean',
        'braking_events': 'sum',
        'brake_max_g': 'max',
        'brake_avg_g': 'mean',
        'drs_activations': 'sum',
        'degradation_slope': 'mean',
        'tyre_age': 'max',
        'is_fresh_tyre': 'max',
        'avg_rainfall': 'mean',
        'avg_track_temp': 'mean',
        'avg_air_temp': 'mean',
        'team': 'first',
        'compound': 'first',
        'session_date': 'first',
    }
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_functions.items() if k in driver_profiles.columns}
    
    # Aggregate to one row per (year, event, driver, session)
    df_driver_agg = driver_profiles.groupby(
        ['year', 'event', 'driver', 'session'],
        as_index=False
    ).agg(agg_dict)
    
    print(f"     Before aggregation: {len(driver_profiles):,} rows")
    print(f"     After aggregation: {len(df_driver_agg):,} rows")
    
    # Use aggregated data
    driver_profiles = df_driver_agg
    # ========== END FIX ==========
    
    # STEP 0: Merge driver features with classification targets
    print("  🔗 Merging features with targets...")
    merged_data = merge_driver_features_with_targets(driver_profiles, start_year, end_year)
    
    print(f"   Loaded {len(merged_data):,} classification records")
    print(f"   Merged dataset: {merged_data.shape}")
    
    # ========== VALIDATE: Check for duplicates ==========
    duplicate_keys = merged_data.duplicated(
        subset=['year', 'event', 'driver', 'session'],
        keep=False
    )
    
    if duplicate_keys.any():
        n_dupes = duplicate_keys.sum()
        print(f"   ⚠️  WARNING: {n_dupes} duplicate rows detected after merge!")
        
        # De-duplicate by keeping first occurrence
        merged_data = merged_data.drop_duplicates(
            subset=['year', 'event', 'driver', 'session'],
            keep='first'
        )
        print(f"   ✅ Deduplicated: {len(merged_data):,} rows")
    # ========== END VALIDATION ==========
    
    # Filter to only sessions with positions (Q, R, Sprint)
    data_with_positions = merged_data[
        merged_data[['qualifying_position', 'race_position']].notna().any(axis=1)
    ].copy()
    
    print(f"   Sessions with positions: {len(data_with_positions):,}")
    
    # Count teams
    if 'team' in data_with_positions.columns:
        n_teams = data_with_positions['team'].nunique()
        print(f"   Teams represented: {n_teams}")
    
    # Verify no duplicates in filtered data
    filtered_dupes = data_with_positions.duplicated(
        subset=['year', 'event', 'driver', 'session']
    ).sum()
    
    if filtered_dupes > 0:
        print(f"   ⚠️  WARNING: {filtered_dupes} duplicates remain after filtering!")
        data_with_positions = data_with_positions.drop_duplicates(
            subset=['year', 'event', 'driver', 'session'],
            keep='first'
        )
        print(f"     Filtered to {len(data_with_positions):,} sessions with position data")
    
    # 1. Circuit history
    print("  📍 Circuit-specific history...")
    circuit_hist = compute_circuit_history(data_with_positions, lookback_years)
    
    # 2. Recent form
    print("  📈 Recent form and momentum...")
    recent_form = compute_recent_form(data_with_positions, form_window)
    
    # 3. Weather performance
    print("  🌧️  Weather-adjusted performance...")
    weather_perf = compute_weather_performance(data_with_positions, rain_threshold)
    
    # 4. Team circuit performance
    print("  🏎️  Team circuit performance...")
    team_circuit = compute_team_circuit_performance(data_with_positions, lookback_years)
    
    # 5. Team momentum
    print("  📊 Team development momentum...")
    team_momentum = compute_team_momentum(data_with_positions, form_window)
    
    # Merge all features
    print("  🔗 Merging all features...")
    
    # Start with data that has positions
    features = data_with_positions.copy()
    
    # Merge circuit history
    features = features.merge(
        circuit_hist,
        on=['driver', 'event', 'year'],
        how='left'
    )
    
    # Merge recent form
    features = features.merge(
        recent_form,
        on=['driver', 'event', 'year'],
        how='left'
    )
    
    # Merge weather performance
    features = features.merge(
        weather_perf,
        on=['driver', 'year'],
        how='left'
    )
    
    # Merge team features
    if 'team' in features.columns:
        features = features.merge(
            team_circuit,
            on=['team', 'event', 'year'],
            how='left'
        )
        
        features = features.merge(
            team_momentum,
            on=['team', 'event', 'year'],
            how='left'
        )
    
    # Final deduplication check
    final_dupes = features.duplicated(subset=['year', 'event', 'driver', 'session']).sum()
    if final_dupes > 0:
        print(f"   ⚠️  Final deduplication: removing {final_dupes} duplicates...")
        features = features.drop_duplicates(
            subset=['year', 'event', 'driver', 'session'],
            keep='first'
        )
    
    print(f"✅ Historical features computed: {features.shape}")
    
    return features