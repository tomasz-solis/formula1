"""
Team-Based Priors for Rookie Feature Imputation

Provides intelligent NaN filling for drivers with missing historical features
(rookies, returnees, drivers at new circuits) using team performance baselines.

Key principle: Rookie at Red Bull ≠ Rookie at Williams

Usage:
    # 1. Compute baselines from training data (once)
    baselines = compute_team_baselines(training_df)
    save_team_baselines(baselines, 'models/team_baselines.json')
    
    # 2. Load at runtime (once at startup)
    baselines = load_team_baselines('models/team_baselines.json')
    
    # 3. Fill features for each prediction
    features = fill_rookie_features(features, driver_team, baselines)
    prediction = model.predict([features])

Author: Tomasz Solis
Date: November 2025
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# TEAM BASELINE COMPUTATION

def compute_team_baselines(
    df: pd.DataFrame,
    lookback_years: int = 2,
    min_samples: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Compute team performance baselines from historical data.
    
    For each team, calculates median qualifying position and variance
    using ONLY veteran drivers (to avoid rookie contamination).
    
    Args:
        df: Training DataFrame with columns:
            - team: Team name
            - year: Season year
            - qualifying_position: Target variable
            - (optional) career_races: For filtering veterans
        lookback_years: How many recent years to use (default: 2)
        min_samples: Minimum races required per team (default: 10)
    
    Returns:
        Dictionary mapping team name to baseline stats:
        {
            'Red Bull Racing': {
                'median_position': 2.0,
                'std_position': 1.5,
                'samples': 42
            },
            ...
        }
    
    Example:
        >>> baselines = compute_team_baselines(training_df)
        >>> print(baselines['Red Bull Racing']['median_position'])
        2.0
    """
    # Filter to recent years
    max_year = df['year'].max()
    recent_data = df[df['year'] >= max_year - lookback_years].copy()

    # warn if there are obviously raw marketing names left

    if 'team' not in df.columns:
        raise ValueError("Expected 'team' column in dataframe for team baselines")

    suspicious = df['team'].dropna().unique()
    if any('Alfa Romeo' in t or 'AlphaTauri' in t for t in suspicious):
        logger.warning("Non-canonical team names detected in baselines input: %s", suspicious)
    
    # Filter to veterans only (if career_races available)
    if 'career_races' in df.columns:
        veteran_mask = recent_data['career_races'] > 20
        recent_data = recent_data[veteran_mask]
        logger.info(f"Using {len(recent_data)} veteran driver-races for team baselines")
    else:
        logger.warning("No 'career_races' column - using all drivers for baselines")
    
    # Filter to valid qualifying positions
    recent_data = recent_data[recent_data['qualifying_position'].notna()]
    
    if len(recent_data) == 0:
        logger.error("No valid data for computing team baselines")
        return {}
    
    # Compute baselines per team
    baselines = {}
    
    for team, group in recent_data.groupby('team'):
        if len(group) < min_samples:
            logger.warning(f"Team '{team}' has only {len(group)} samples (min: {min_samples}), skipping")
            continue
        
        baselines[team] = {
            'median_position': float(group['qualifying_position'].median()),
            'mean_position': float(group['qualifying_position'].mean()),
            'std_position': float(group['qualifying_position'].std()),
            'best_position': float(group['qualifying_position'].min()),
            'worst_position': float(group['qualifying_position'].max()),
            'samples': int(len(group)),
            'years': sorted(group['year'].unique().tolist())
        }
    
    logger.info(f"✅ Computed baselines for {len(baselines)} teams")
    
    # Add overall grid baseline (for unknown teams)
    baselines['_default'] = {
        'median_position': float(recent_data['qualifying_position'].median()),
        'mean_position': float(recent_data['qualifying_position'].mean()),
        'std_position': float(recent_data['qualifying_position'].std()),
        'samples': len(recent_data)
    }
    
    return baselines

def save_team_baselines(baselines: Dict, filepath: str) -> None:
    """
    Save team baselines to JSON file.
    
    Args:
        baselines: Team baseline dictionary
        filepath: Path to save JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(baselines, f, indent=2)
    
    logger.info(f" Saved team baselines to {filepath}")

def load_team_baselines(filepath: str) -> Dict:
    """
    Load team baselines from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Team baseline dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"Team baselines file not found: {filepath}")
        raise FileNotFoundError(f"No team baselines at {filepath}")
    
    with open(filepath, 'r') as f:
        baselines = json.load(f)
    
    logger.info(f" Loaded baselines for {len(baselines)} teams")
    
    return baselines

# FEATURE FILLING

def fill_rookie_features(
    features: Dict[str, Any],
    team: str,
    team_baselines: Dict[str, Dict[str, float]],
    rookie_penalty: float = 1.5
) -> Dict[str, Any]:
    """
    Fill missing historical features using team-based priors.
    
    For each NaN feature, fills with appropriate value based on:
    - Feature type (position, trend, count, etc.)
    - Team performance baseline
    - Rookie uncertainty penalty
    
    Args:
        features: Feature dictionary (may contain NaN values)
        team: Team name (e.g., "Red Bull Racing")
        team_baselines: Team baseline dictionary from compute_team_baselines()
        rookie_penalty: Positions to add to team baseline for uncertainty (default: 1.5)
    
    Returns:
        Feature dictionary with all NaNs filled
    
    Example:
        >>> features = {'circuit_avg_position': np.nan, 'brake_max_g': 5.2, ...}
        >>> filled = fill_rookie_features(features, 'Red Bull Racing', baselines)
        >>> print(filled['circuit_avg_position'])
        3.5  # Red Bull baseline (2.0) + rookie penalty (1.5)
    """
    # Get team baseline (fallback to grid average if team unknown)
    if team in team_baselines:
        baseline = team_baselines[team]['median_position']
        team_std = team_baselines[team]['std_position']
    else:
        logger.warning(f"Unknown team '{team}', using grid average")
        baseline = team_baselines.get('_default', {}).get('median_position', 10.5)
        team_std = team_baselines.get('_default', {}).get('std_position', 3.0)
    
    # Count how many features we fill
    filled_count = 0
    
    # Position features: team baseline + rookie penalty
    position_features = [
        'circuit_avg_position',
        'circuit_best_position', 
        'circuit_worst_position',
        'recent_avg_position',
        'recent_best_position',
        'recent_worst_position',
        'wet_avg_position',
        'dry_avg_position',
        'team_circuit_avg_position',
        'team_recent_avg'
    ]
    
    for feat in position_features:
        if feat in features and (pd.isna(features[feat]) or features[feat] is None):
            # For "best" positions, be optimistic (baseline only)
            # For "avg/worst" positions, be conservative (baseline + penalty)
            if 'best' in feat:
                features[feat] = baseline
            else:
                features[feat] = baseline + rookie_penalty
            filled_count += 1
    
    # Trend/momentum features: assume neutral
    trend_features = [
        'form_trend',
        'team_momentum',
        'circuit_avg_position_change',
        'circuit_std_position_change',
        'driver_avg_position_change',
        'driver_std_position_change'
    ]
    
    for feat in trend_features:
        if feat in features and (pd.isna(features[feat]) or features[feat] is None):
            features[feat] = 0.0  # Neutral trend
            filled_count += 1
    
    # Count features: zero (no history)
    count_features = [
        'circuit_sessions',
        'races_in_window',
        'wet_sessions',
        'dry_sessions',
        'team_circuit_sessions',
        'circuit_overtaking_samples',
        'driver_overtaking_samples'
    ]
    
    for feat in count_features:
        if feat in features and (pd.isna(features[feat]) or features[feat] is None):
            features[feat] = 0
            filled_count += 1
    
    # Standard deviation features: high uncertainty
    std_features = [
        'circuit_std_position',
        'wet_dry_delta'
    ]
    
    for feat in std_features:
        if feat in features and (pd.isna(features[feat]) or features[feat] is None):
            features[feat] = max(team_std, 2.5)  # At least 2.5 positions variance
            filled_count += 1
    
    # Success rate features: neutral (50%)
    rate_features = [
        'driver_overtaking_success_rate',
        'driver_defensive_success_rate'
    ]
    
    for feat in rate_features:
        if feat in features and (pd.isna(features[feat]) or features[feat] is None):
            features[feat] = 0.5  # 50% success rate = average
            filled_count += 1
    
    # Absolute value features: use median
    abs_features = [
        'circuit_abs_position_change',
        'circuit_max_gain',
        'circuit_max_loss'
    ]
    
    for feat in abs_features:
        if feat in features and (pd.isna(features[feat]) or features[feat] is None):
            features[feat] = 3.0  # Typical position change
            filled_count += 1
    
    # Circuit race position (if different from qualifying)
    if 'circuit_avg_race' in features and (pd.isna(features['circuit_avg_race']) or features['circuit_avg_race'] is None):
        # Assume slight degradation from qualifying to race
        features['circuit_avg_race'] = baseline + rookie_penalty + 1.0
        filled_count += 1
    
    if 'circuit_best_race' in features and (pd.isna(features['circuit_best_race']) or features['circuit_best_race'] is None):
        features['circuit_best_race'] = baseline
        filled_count += 1
    
    if filled_count > 0:
        logger.debug(f"Filled {filled_count} features for team '{team}'")
    
    return features

def fill_features_batch(
    df: pd.DataFrame,
    team_baselines: Dict[str, Dict[str, float]],
    rookie_penalty: float = 1.5
) -> pd.DataFrame:
    """
    Fill missing features for entire DataFrame (training or prediction set).
    
    Applies fill_rookie_features() to each row, handling team-specific logic.
    
    Args:
        df: DataFrame with potential NaN values
        team_baselines: Team baseline dictionary
        rookie_penalty: Positions to add for uncertainty
    
    Returns:
        DataFrame with all historical features filled
    
    Example:
        >>> df = fill_features_batch(test_df, baselines)
        INFO - Filled features for 1888 rows
        >>> assert df.isnull().sum().sum() == 0
    """
    if 'team' not in df.columns:
        raise ValueError("DataFrame must have 'team' column for team-based filling")
    
    # Track how many features we fill
    initial_nulls = df.isnull().sum().sum()
    
    # Apply filling row by row
    for idx, row in df.iterrows():
        team = row['team']
        
        # Convert row to dict, fill, and update
        row_dict = row.to_dict()
        filled_dict = fill_rookie_features(row_dict, team, team_baselines, rookie_penalty)
        
        # Update only the filled values
        for key, value in filled_dict.items():
            if key in df.columns:
                df.at[idx, key] = value
    
    final_nulls = df.isnull().sum().sum()
    filled = initial_nulls - final_nulls
    
    logger.info(f"✅ Filled {filled} NaN values across {len(df)} rows")
    
    return df

# VALIDATION & DIAGNOSTICS

def validate_team_baselines(baselines: Dict) -> bool:
    """
    Validate that team baselines are reasonable.
    
    Args:
        baselines: Team baseline dictionary
    
    Returns:
        True if valid, False if suspicious
    """
    issues = []
    
    for team, stats in baselines.items():
        if team == '_default':
            continue
        
        median = stats['median_position']
        
        # Check if positions are in valid range
        if not (1 <= median <= 20):
            issues.append(f"{team}: median position {median:.1f} outside [1,20]")
        
        # Check if sample size is reasonable
        if stats['samples'] < 5:
            issues.append(f"{team}: only {stats['samples']} samples (might be unreliable)")
    
    if issues:
        logger.warning("Team baseline validation issues:")
        for issue in issues:
            logger.warning(f" - {issue}")
        return False
    
    logger.info("✅ Team baselines validation passed")
    return True

def compare_filling_strategies(
    df: pd.DataFrame,
    team_baselines: Dict,
    feature_name: str = 'circuit_avg_position'
) -> pd.DataFrame:
    """
    Compare different NaN filling strategies.
    
    Shows how team-based filling differs from naive approaches.
    
    Args:
        df: DataFrame with NaN values
        team_baselines: Team baseline dictionary
        feature_name: Feature to analyze
    
    Returns:
        Comparison DataFrame with different filling strategies
    """
    # Find rows with NaN for this feature
    nan_rows = df[df[feature_name].isna()].copy()
    
    if len(nan_rows) == 0:
        logger.info(f"No NaN values in {feature_name}")
        return pd.DataFrame()
    
    results = []
    
    for idx, row in nan_rows.iterrows():
        team = row['team']
        driver = row.get('driver', 'Unknown')
        
        # Strategy 1: Global median
        global_median = df[feature_name].median()
        
        # Strategy 2: Team-based
        team_baseline = team_baselines.get(team, team_baselines['_default'])['median_position']
        team_fill = team_baseline + 1.5  # With rookie penalty
        
        results.append({
            'driver': driver,
            'team': team,
            'global_median': global_median,
            'team_based': team_fill,
            'difference': team_fill - global_median
        })
    
    comparison_df = pd.DataFrame(results)
    
    logger.info(f"\n Filling Strategy Comparison for {feature_name}:")
    logger.info(f"  Global median: {global_median:.1f}")
    logger.info(f"  Team-based range: {comparison_df['team_based'].min():.1f} - {comparison_df['team_based'].max():.1f}")
    logger.info(f"  Average difference: {comparison_df['difference'].abs().mean():.1f} positions")
    
    return comparison_df

# MAIN WORKFLOW

def generate_and_save_baselines(
    training_data_path: str,
    output_path: str = 'models/team_baselines.json',
    lookback_years: int = 2
) -> Dict:
    """
    Complete workflow: load data, compute baselines, save to file.
    
    Args:
        training_data_path: Path to training data (CSV or parquet)
        output_path: Where to save baselines JSON
        lookback_years: Years of recent data to use
    
    Returns:
        Team baseline dictionary
    
    Example:
        >>> baselines = generate_and_save_baselines(
        ...     'data/features/ml_features.parquet',
        ...     'models/team_baselines.json'
        ... )
        INFO - Loaded 1888 rows
        INFO - ✅ Computed baselines for 10 teams
        INFO -  Saved team baselines to models/team_baselines.json
    """
    # Load training data
    logger.info(f" Loading training data from {training_data_path}")
    
    if training_data_path.endswith('.parquet'):
        df = pd.read_parquet(training_data_path)
    else:
        df = pd.read_csv(training_data_path)
    
    from helpers.team_name_map import TEAM_NAME_MAP
    # Canonicalize team names
    if 'team' in df.columns:
        df['team'] = (
            df['team']
            .map(TEAM_NAME_MAP)
            .fillna(df['team'].astype(str).str.upper())
    )
    logger.info(f"  Loaded {len(df)} rows")
    
    # Compute baselines
    baselines = compute_team_baselines(df, lookback_years=lookback_years)
    
    # Validate
    validate_team_baselines(baselines)
    
    # Save
    save_team_baselines(baselines, output_path)
    
    return baselines

if __name__ == "__main__":
    """
    Test the module with sample data.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate team performance baselines")
    parser.add_argument('--data', required=True, help='Path to training data')
    parser.add_argument('--output', default='models/team_baselines.json', help='Output path')
    parser.add_argument('--lookback', type=int, default=2, help='Years of data to use')
    
    args = parser.parse_args()
    
    # Generate baselines
    baselines = generate_and_save_baselines(
        args.data,
        args.output,
        args.lookback
    )
    
    # Print summary
    print("\n" + "="*60)
    print("TEAM PERFORMANCE BASELINES")
    print("="*60 + "\n")
    
    # Sort teams by performance
    teams_sorted = sorted(
        [(team, stats) for team, stats in baselines.items() if team != '_default'],
        key=lambda x: x[1]['median_position']
    )
    
    print(f"{'Team':<25} {'Median':<8} {'Std':<6} {'Samples':<8}")
    print("-"*60)
    
    for team, stats in teams_sorted:
        print(f"{team:<25} {stats['median_position']:>6.1f}   {stats['std_position']:>5.1f}   {stats['samples']:>6}")
    
    print("\n" + "="*60)
    print(f"✅ Baselines saved to {args.output}")
    print("="*60 + "\n")