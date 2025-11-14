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
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
from .general_utils import merge_driver_features_with_targets


# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


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
    
    Returns average and best qualifying/race positions at each circuit.
    
    Args:
        driver_profiles: Driver session data with positions
        lookback_years: Years of history to consider
        min_sessions: Minimum sessions required for valid history
        
    Returns:
        DataFrame with circuit-specific performance metrics
    """
    if 'qualifying_position' not in driver_profiles.columns:
        raise ValueError("DataFrame must have 'qualifying_position' column")
    
    # Filter to sessions with position data
    valid_data = driver_profiles[
        driver_profiles['qualifying_position'].notna()
    ].copy()
    
    if valid_data.empty:
        return pd.DataFrame()
    
    # Build aggregations
    agg_dict = {
        'qualifying_position': ['mean', 'min', 'count']
    }
    
    if 'race_position' in valid_data.columns:
        agg_dict['race_position'] = ['mean', 'min']
    
    # Group and aggregate
    history = valid_data.groupby(
        ['driver', 'event', 'year'], 
        as_index=False
    ).agg(agg_dict)
    
    # CRITICAL: Flatten MultiIndex columns properly
    if isinstance(history.columns, pd.MultiIndex):
        # Flatten: ('qualifying_position', 'mean') -> 'circuit_avg_position'
        new_cols = ['driver', 'event', 'year']
        
        for col in history.columns[3:]:  # Skip driver, event, year
            if col[0] == 'qualifying_position':
                if col[1] == 'mean':
                    new_cols.append('circuit_avg_position')
                elif col[1] == 'min':
                    new_cols.append('circuit_best_position')
                elif col[1] == 'count':
                    new_cols.append('circuit_sessions')
            elif col[0] == 'race_position':
                if col[1] == 'mean':
                    new_cols.append('circuit_avg_race')
                elif col[1] == 'min':
                    new_cols.append('circuit_best_race')
        
        history.columns = new_cols
    
    # Filter minimum sessions
    if 'circuit_sessions' in history.columns:
        history = history[history['circuit_sessions'] >= min_sessions]
    
    # Verify required columns
    required = ['driver', 'event', 'year', 'circuit_avg_position']
    missing = [col for col in required if col not in history.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return history


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
# RACE-SPECIFIC PERFORMANCE
# =============================================================================

def compute_race_pace_vs_quali(
    driver_profiles: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute driver's race pace relative to qualifying pace.
    
    Some drivers are better in races (tire management, consistency)
    vs qualifying (one-lap pace). This captures that difference.
    
    Args:
        driver_profiles: DataFrame with qualifying_position and race_position
        
    Returns:
        DataFrame with race vs quali performance metrics:
        - driver, year: Identifiers
        - race_vs_quali_delta: Average position change (negative = gains in race)
        - race_consistency: Std dev of position changes
        - races_improved: Count of races where position improved
        - races_declined: Count of races where position declined
        
    Example:
        >>> pace = compute_race_pace_vs_quali(driver_profiles)
        >>> strong_racers = pace[pace['race_vs_quali_delta'] < -1]  # Gain positions
        >>> print(strong_racers[['driver', 'race_vs_quali_delta']])
    """
    # Filter to sessions with both quali and race positions
    df = driver_profiles[
        driver_profiles['qualifying_position'].notna() &
        driver_profiles['race_position'].notna()
    ].copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Calculate position change (race - quali)
    # Negative = gained positions (better in race)
    df['position_change'] = df['race_position'] - df['qualifying_position']
    
    results = []
    
    for (driver, year), group in df.groupby(['driver', 'year']):
        if len(group) < 3:  # Need at least 3 races for stats
            continue
        
        position_changes = group['position_change'].values
        
        results.append({
            'driver': driver,
            'year': year,
            'race_vs_quali_delta': position_changes.mean(),
            'race_consistency': position_changes.std(),
            'races_improved': (position_changes < 0).sum(),
            'races_declined': (position_changes > 0).sum(),
            'races_maintained': (position_changes == 0).sum(),
            'best_position_gain': position_changes.min(),
            'worst_position_loss': position_changes.max(),
            'races_sampled': len(position_changes)
        })
    
    return pd.DataFrame(results)


def compute_overtaking_metrics(
    driver_profiles: pd.DataFrame,
    min_races: int = 5
) -> pd.DataFrame:
    """
    Compute driver overtaking and defending capabilities.
    
    Measures how often drivers gain positions in races and how well
    they defend their starting position.
    
    Args:
        driver_profiles: DataFrame with qualifying and race positions
        min_races: Minimum races to compute metrics (default: 5)
        
    Returns:
        DataFrame with overtaking metrics:
        - driver, year: Identifiers
        - avg_positions_gained: Average positions gained per race
        - overtaking_frequency: Percentage of races where positions gained
        - avg_positions_lost: Average positions lost per race
        - defending_success: Percentage of races where position maintained/improved
        
    Example:
        >>> overtaking = compute_overtaking_metrics(driver_profiles)
        >>> best_overtakers = overtaking.nlargest(10, 'avg_positions_gained')
        >>> print(best_overtakers[['driver', 'avg_positions_gained']])
    """
    df = driver_profiles[
        driver_profiles['qualifying_position'].notna() &
        driver_profiles['race_position'].notna()
    ].copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    df['position_change'] = df['race_position'] - df['qualifying_position']
    
    results = []
    
    for (driver, year), group in df.groupby(['driver', 'year']):
        if len(group) < min_races:
            continue
        
        changes = group['position_change'].values
        gains = changes[changes < 0]  # Negative = gained positions
        losses = changes[changes > 0]  # Positive = lost positions
        
        results.append({
            'driver': driver,
            'year': year,
            'avg_positions_gained': -gains.mean() if len(gains) > 0 else 0.0,
            'overtaking_frequency': len(gains) / len(changes),
            'avg_positions_lost': losses.mean() if len(losses) > 0 else 0.0,
            'defending_success': len(changes[changes <= 0]) / len(changes),
            'max_positions_gained': -changes.min() if changes.min() < 0 else 0,
            'max_positions_lost': changes.max() if changes.max() > 0 else 0
        })
    
    return pd.DataFrame(results)


def compute_circuit_overtaking_difficulty(
    driver_profiles: pd.DataFrame
) -> pd.DataFrame:
    """
    Rate circuits by how difficult overtaking is.
    
    Analyzes historical data to determine which circuits allow more
    position changes during races.
    
    Args:
        driver_profiles: DataFrame with qualifying and race positions by circuit
        
    Returns:
        DataFrame with circuit overtaking metrics:
        - event, year: Identifiers
        - avg_position_changes: Average |position change| per driver
        - overtake_frequency: % of drivers who gain/lose positions
        - top_10_volatility: Position changes in top 10
        - overtaking_difficulty: 1-10 scale (1=easy, 10=hard)
        
    Example:
        >>> difficulty = compute_circuit_overtaking_difficulty(driver_profiles)
        >>> hardest = difficulty.nlargest(5, 'overtaking_difficulty')
        >>> print(hardest[['event', 'overtaking_difficulty']])
        # Likely shows: Monaco, Singapore, Hungary...
    """
    df = driver_profiles[
        driver_profiles['qualifying_position'].notna() &
        driver_profiles['race_position'].notna()
    ].copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    df['position_change'] = df['race_position'] - df['qualifying_position']
    df['abs_position_change'] = np.abs(df['position_change'])
    
    results = []
    
    for (event, year), group in df.groupby(['event', 'year']):
        if len(group) < 10:  # Need reasonable sample
            continue
        
        changes = group['position_change'].values
        abs_changes = group['abs_position_change'].values
        
        # Focus on top 10 (where overtaking matters most for points)
        top10 = group[group['qualifying_position'] <= 10]
        top10_volatility = np.abs(top10['position_change']).mean() if len(top10) > 0 else 0
        
        # Overtaking difficulty: inverse of average position changes
        # More changes = easier overtaking
        avg_change = abs_changes.mean()
        
        # Scale to 1-10 (will normalize later)
        difficulty = 10 - (avg_change * 2)  # Rough scaling
        difficulty = max(1, min(10, difficulty))  # Clamp to 1-10
        
        results.append({
            'event': event,
            'year': year,
            'avg_position_changes': abs_changes.mean(),
            'overtake_frequency': (changes != 0).sum() / len(changes),
            'top_10_volatility': top10_volatility,
            'overtaking_difficulty': difficulty,
            'total_position_changes': abs_changes.sum()
        })
    
    df_result = pd.DataFrame(results)
    
    # Normalize overtaking_difficulty to 1-10 scale across all circuits
    if len(df_result) > 0:
        min_val = df_result['avg_position_changes'].min()
        max_val = df_result['avg_position_changes'].max()
        
        if max_val > min_val:
            # Invert: more changes = lower difficulty
            df_result['overtaking_difficulty'] = 10 - (
                9 * (df_result['avg_position_changes'] - min_val) / (max_val - min_val)
            )
    
    return df_result


def compute_dnf_probability(
    driver_profiles: pd.DataFrame,
    lookback_races: int = 10
) -> pd.DataFrame:
    """
    Compute probability of Did Not Finish (DNF) for drivers and teams.
    
    Args:
        driver_profiles: DataFrame with race completion data
        lookback_races: Number of recent races to analyze (default: 10)
        
    Returns:
        DataFrame with DNF probabilities:
        - driver, year: Identifiers
        - driver_dnf_rate: Driver's DNF rate
        - team_dnf_rate: Team's DNF rate
        - recent_dnf_count: DNFs in last N races
        - reliability_score: 0-1 (1 = very reliable)
        
    Example:
        >>> dnf = compute_dnf_probability(driver_profiles)
        >>> unreliable = dnf.nlargest(10, 'driver_dnf_rate')
        >>> print(unreliable[['driver', 'team', 'driver_dnf_rate']])
    """
    # Assume DNF if race_position > 20 or is NaN (but quali position exists)
    df = driver_profiles[
        driver_profiles['qualifying_position'].notna()
    ].copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Mark DNF
    df['is_dnf'] = (
        df['race_position'].isna() | 
        (df['race_position'] > 20)
    )
    
    results = []
    
    # Driver DNF rate
    for (driver, year), group in df.groupby(['driver', 'year']):
        if len(group) < 3:
            continue
        
        dnf_rate = group['is_dnf'].mean()
        
        # Recent DNF rate (last N races)
        recent = group.tail(lookback_races)
        recent_dnf_rate = recent['is_dnf'].mean()
        recent_dnf_count = recent['is_dnf'].sum()
        
        results.append({
            'driver': driver,
            'year': year,
            'driver_dnf_rate': dnf_rate,
            'recent_dnf_count': recent_dnf_count,
            'recent_dnf_rate': recent_dnf_rate,
            'races_started': len(group),
            'races_finished': (~group['is_dnf']).sum(),
            'reliability_score': 1 - dnf_rate
        })
    
    df_driver_dnf = pd.DataFrame(results)
    
    # Team DNF rate
    if 'team' in df.columns:
        team_results = []
        
        for (team, year), group in df.groupby(['team', 'year']):
            if len(group) < 5:
                continue
            
            team_results.append({
                'team': team,
                'year': year,
                'team_dnf_rate': group['is_dnf'].mean(),
                'team_reliability_score': 1 - group['is_dnf'].mean()
            })
        
        df_team_dnf = pd.DataFrame(team_results)
        
        # Merge team DNF rate back to driver data
        if len(df_team_dnf) > 0 and 'team' in df.columns:
            # Add team info to driver DNF data
            driver_team_map = df[['driver', 'year', 'team']].drop_duplicates()
            df_driver_dnf = df_driver_dnf.merge(
                driver_team_map,
                on=['driver', 'year'],
                how='left'
            )
            df_driver_dnf = df_driver_dnf.merge(
                df_team_dnf,
                on=['team', 'year'],
                how='left'
            )
    
    return df_driver_dnf

# =============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# =============================================================================

def compute_historical_features(
    driver_profiles: pd.DataFrame,
    circuit_profiles: pd.DataFrame,
    lookback_years: int = 3,
    form_window: int = 5,
    rain_threshold: float = 0.1,
    start_year: int = 2022,
    end_year: int = 2025,
    include_race_features: bool = True
) -> pd.DataFrame:
    """
    Compute all historical features for ML model training.
    
    Combines circuit-specific history, recent form, weather performance,
    team metrics, and optionally race-specific features (overtaking, DNF).
    
    Args:
        driver_profiles: Raw driver session data
        circuit_profiles: Circuit characteristics
        lookback_years: Years of history for circuit performance
        form_window: Number of races for recent form
        rain_threshold: Rainfall threshold for wet sessions (mm/h)
        start_year: First year to include in output
        end_year: Last year to include in output
        include_race_features: Whether to compute race-specific features
        
    Returns:
        DataFrame with all features merged, ready for ML training
    """
    logger.info("🔮 Computing historical features...")
    
    # ========================================================================
    # STEP 1: Merge with positions
    # ========================================================================

    driver_with_positions = merge_driver_features_with_targets(
        driver_profiles,
        start_year=start_year,
        end_year=end_year
    )
    
    if driver_with_positions.empty:
        logger.error("❌ No data after merging with positions!")
        return pd.DataFrame()
        
    # Validate position columns
    has_quali = 'qualifying_position' in driver_with_positions.columns
    has_race = 'race_position' in driver_with_positions.columns
    
    if not has_quali and not has_race:
        logger.error("❌ No position columns found!")
        return driver_with_positions    
    
    # ========================================================================
    # STEP 2: AGGREGATE: Reduce to ONE ROW per (year, event, driver, session)
    # ========================================================================

    initial_rows = len(driver_with_positions)

    # Define aggregation groups
    group_keys = ['year', 'event', 'driver']
    if 'session' in driver_with_positions.columns:
        group_keys.append('session')

    # Build aggregation dictionary
    agg_dict = {}

    # Preserve these columns with 'first' (should be same within group)
    preserve_cols = ['team', 'session_date', 'qualifying_position', 'race_position', 'event']
    for col in preserve_cols:
        if col in driver_with_positions.columns and col not in group_keys:
            agg_dict[col] = 'first'

    # Average all numeric columns not already handled
    numeric_cols = driver_with_positions.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Skip if already in preserve_cols or group_keys
        if col not in group_keys and col not in agg_dict and col not in ['year']:
            agg_dict[col] = 'mean'

    # Perform aggregation
    driver_with_positions = driver_with_positions.groupby(
        group_keys, 
        as_index=False,
        dropna=False
    ).agg(agg_dict)

    final_rows = len(driver_with_positions)

    # Verify critical columns
    missing = []
    for col in ['session_date', 'team']:
        if col not in driver_with_positions.columns:
            missing.append(col)

    if missing:
        logger.error(f"❌ Missing columns after aggregation: {missing}")
        return pd.DataFrame()
    else:
        logger.info(f"✅ All critical columns preserved")

    if final_rows == 0:
        logger.error("❌ No rows after aggregation!")
        return pd.DataFrame()
    
    # ========================================================================
    # STEP 3: INITIALIZE ALL FEATURE VARIABLES (prevents UnboundLocalError)
    # ========================================================================
    circuit_history = pd.DataFrame()
    recent_form = pd.DataFrame()
    weather_perf = pd.DataFrame()
    team_circuit = pd.DataFrame()
    team_momentum = pd.DataFrame()
    race_pace = pd.DataFrame()
    overtaking = pd.DataFrame()
    circuit_overtaking = pd.DataFrame()
    dnf_prob = pd.DataFrame()
    
    # ========================================================================
    # STEP 4: Compute features (with error handling)
    # ========================================================================
    try:
        circuit_history = compute_circuit_history(
            driver_with_positions,
            lookback_years=lookback_years
        )
    except Exception as e:
        logger.error(f"❌ Failed to compute circuit history: {e}")
    
    try:
        recent_form = compute_recent_form(
            driver_with_positions,
            window_size=form_window
        )
    except Exception as e:
        logger.error(f"❌ Failed to compute recent form: {e}")
    
    try:
        weather_perf = compute_weather_performance(
            driver_with_positions,
            rain_threshold=rain_threshold
        )
    except Exception as e:
        logger.error(f"❌ Failed to compute weather performance: {e}")
    
    try:
        team_circuit = compute_team_circuit_performance(
            driver_with_positions,
            lookback_years=lookback_years
        )
    except Exception as e:
        logger.error(f"❌ Failed to compute team circuit: {e}")
    
    try:
        team_momentum = compute_team_momentum(
            driver_with_positions,
            window_size=form_window
        )
    except Exception as e:
        logger.error(f"❌ Failed to compute team momentum: {e}")
    
    # Race-specific features
    if include_race_features and has_race:        
        try:
            race_pace = compute_race_pace_vs_quali(driver_with_positions)
        except Exception as e:
            logger.warning(f"❌ Race pace failed: {e}")
        
        try:
            overtaking = compute_overtaking_metrics(driver_with_positions)
        except Exception as e:
            logger.warning(f"❌ Overtaking failed: {e}")
        
        try:
            circuit_overtaking = compute_circuit_overtaking_difficulty(driver_with_positions)
        except Exception as e:
            logger.warning(f"❌ Circuit overtaking failed: {e}")
        
        try:
            dnf_prob = compute_dnf_probability(driver_with_positions)
        except Exception as e:
            logger.warning(f"❌ DNF probability failed: {e}")
    
    # ========================================================================
    # STEP 5: Smart merging - different granularities
    # ========================================================================

    result = driver_with_positions.copy()
    initial_rows = len(result)

    # 1. EVENT-LEVEL features (year + event + driver)
    event_level_features = []

    if not circuit_history.empty:
        event_level_features.append(('circuit_history', circuit_history, ['year', 'event', 'driver']))

    # 2. DRIVER-YEAR features (year + driver)
    driver_year_features = []

    if not recent_form.empty:
        driver_year_features.append(('recent_form', recent_form, ['year', 'driver']))

    if not weather_perf.empty:
        driver_year_features.append(('weather_perf', weather_perf, ['year', 'driver']))

    if not dnf_prob.empty:
        driver_year_features.append(('dnf_prob', dnf_prob, ['year', 'driver']))

    if not race_pace.empty:
        driver_year_features.append(('race_pace', race_pace, ['year', 'driver']))
    
    if not overtaking.empty:
        driver_year_features.append(('overtaking', overtaking, ['year', 'driver']))

    # 3. CIRCUIT-YEAR features (year + event)
    circuit_year_features = []

    if not circuit_overtaking.empty:
        circuit_year_features.append(('circuit_overtaking', circuit_overtaking, ['year', 'event']))

    # 4. TEAM features
    team_features = []

    if not team_circuit.empty and 'team' in result.columns:
        team_features.append(('team_circuit', team_circuit, ['team', 'year', 'event']))

    if not team_momentum.empty and 'team' in result.columns:
        team_features.append(('team_momentum', team_momentum, ['team', 'year']))

    # PERFORM MERGES
    for name, df, keys in event_level_features:
        result = result.merge(df, on=keys, how='left', suffixes=('', f'_{name}'))
    
        # Check for 'event' column
        if 'event' not in result.columns:
            logger.error(f"❌ 'event' column missing after {name} merge!")
            raise KeyError(f"'event' column lost after merging {name}")
        
        # CRITICAL: Check for explosion
        if len(result) > initial_rows * 1.1:
            logger.error(f"❌ Merge explosion detected! {initial_rows:,} → {len(result):,}")
            result = result.drop_duplicates(subset=['year', 'event', 'driver', 'session'])
            logger.warning(f"Deduped to: {len(result):,} rows")

    for name, df, keys in driver_year_features:
        result = result.merge(df, on=keys, how='left', suffixes=('', f'_{name}'))
        
        if len(result) > initial_rows * 1.1:
            logger.error(f"❌ Merge explosion detected! {initial_rows:,} → {len(result):,}")
            result = result.drop_duplicates(subset=['year', 'event', 'driver', 'session'])
            logger.warning(f"Deduped to: {len(result):,} rows")

    for name, df, keys in circuit_year_features:
        result = result.merge(df, on=keys, how='left', suffixes=('', f'_{name}'))
        
        if len(result) > initial_rows * 1.1:
            logger.error(f"❌ Merge explosion detected!")
            result = result.drop_duplicates(subset=['year', 'event', 'driver', 'session'])
            logger.warning(f"Deduped to: {len(result):,} rows")

    for name, df, keys in team_features:
        result = result.merge(df, on=keys, how='left', suffixes=('', f'_{name}'))
        
        if len(result) > initial_rows * 1.1:
            logger.error(f"❌ Merge explosion detected!")
            result = result.drop_duplicates(subset=['year', 'event', 'driver', 'session'])
            logger.warning(f"Deduped to: {len(result):,} rows")

    # Circuit profiles (event-level)
    if not circuit_profiles.empty:
        circuit_cols = ['event', 'year', 'slow_corner_pct', 'medium_corner_pct', 
                        'fast_corner_pct', 'total_corners', 'chicanes',
                        'avg_speed_circuit', 'top_speed_circuit']
        available_cols = [col for col in circuit_cols if col in circuit_profiles.columns]
        
        if len(available_cols) > 2:
            result = result.merge(
                circuit_profiles[available_cols].drop_duplicates(),
                on=['event', 'year'],
                how='left',
                suffixes=('', '_circuit')
            )

    # ========================================================================
    # STEP 6: Final check
    # ========================================================================
    final_rows = len(result)
    if final_rows != initial_rows:
        logger.warning(f"⚠️  Row count changed: {initial_rows:,} → {final_rows:,}")
        
        # Force dedupe if needed
        if final_rows > initial_rows * 1.1:
            logger.error(f"❌ Excessive duplication detected! Deduplicating...")
            result = result.drop_duplicates(subset=['year', 'event', 'driver', 'session'])
            logger.warning(f"Deduped to: {len(result):,} rows")

    logger.info(f"✅ Final feature dataset: {result.shape}")

    return result