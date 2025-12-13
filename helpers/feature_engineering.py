"""
Feature engineering for F1 qualifying position prediction.

This module transforms raw F1 telemetry data into ML-ready features by:
1. Loading driver profiles, circuit characteristics, and qualifying results
2. Filtering to practice sessions only (avoiding data leakage)
3. Aggregating multiple practice sessions per driver-race
4. Handling sprint vs normal weekend differences
5. Imputing missing circuit data from adjacent years

The output is a clean dataset with one row per driver-race combination,
ready for machine learning model training.

Example:
    >>> from helpers.feature_engineering import prepare_qualifying_dataset
    >>> df = prepare_qualifying_dataset([2022, 2023, 2024])
    >>> print(df.shape)
    (1353, 37)  # 1353 examples, 37 columns (36 features + 1 target)

Author: Tomasz Solis
Date: November 2025
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Final

# CONFIGURATION & CONSTANTS

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s' #OPTIONAL: %(asctime)s - %(levelname)s - 
)
logger = logging.getLogger(__name__)

# F1 Season constraints
MIN_F1_YEAR: Final[int] = 1950
MAX_F1_YEAR: Final[int] = 2030  # Allow future years for planning

# Session types
PRACTICE_SESSIONS: Final[List[str]] = ['FP1', 'FP2', 'FP3', 'SQ']
QUALIFYING_SESSIONS: Final[List[str]] = ['Q', 'SQ']
RACE_SESSIONS: Final[List[str]] = ['R', 'S']
ALL_SESSIONS: Final[List[str]] = PRACTICE_SESSIONS + QUALIFYING_SESSIONS + RACE_SESSIONS

# Data paths (relative to project root)
DATA_DIR: Final[Path] = Path('data')
DRIVER_PROFILE_DIR: Final[Path] = DATA_DIR / 'driver'
CIRCUIT_PROFILE_DIR: Final[Path] = DATA_DIR / 'circuit'
RESULTS_DIR: Final[Path] = DATA_DIR / 'predictions' / 'ssot'
PROCESSED_DIR: Final[Path] = DATA_DIR / 'processed'

# Feature column groups
DRIVER_FEATURE_COLS: Final[List[str]] = [
    'max_throttle_ratio', 'brake_max_g', 'brake_avg_g',
    'drs_activations', 'degradation_slope'
]

CIRCUIT_FEATURE_COLS: Final[List[str]] = [
    'slow_corners', 'medium_corners', 'fast_corners',
    'avg_speed', 'top_speed', 'chicanes'
]

WEATHER_FEATURE_COLS: Final[List[str]] = [
    'avg_rainfall', 'avg_track_temp', 'avg_air_temp'
]

# Merge keys
MERGE_KEYS: Final[List[str]] = ['year', 'event', 'session']

# Validation thresholds
MIN_CORNERS_PER_CIRCUIT: Final[int] = 5
MAX_CORNERS_PER_CIRCUIT: Final[int] = 25
MIN_THROTTLE_RATIO: Final[float] = 0.0
MAX_THROTTLE_RATIO: Final[float] = 1.0
MIN_QUALIFYING_POSITION: Final[int] = 1
MAX_QUALIFYING_POSITION: Final[int] = 20

# INPUT VALIDATION FUNCTIONS

def validate_years(years: List[int]) -> None:
    """
    Validate that years list is valid for F1 data processing.
    
    Args:
        years: List of season years to validate
        
    Raises:
        ValueError: If years list is invalid (empty, out of range, etc.)
    """
    if not years:
        raise ValueError("Years list cannot be empty")
    
    if not isinstance(years, list):
        raise ValueError(f"Years must be a list, got {type(years)}")
    
    if not all(isinstance(y, int) for y in years):
        raise ValueError(f"All years must be integers, got {years}")
    
    invalid_years = [y for y in years if not (MIN_F1_YEAR <= y <= MAX_F1_YEAR)]
    if invalid_years:
        raise ValueError(
            f"Years must be between {MIN_F1_YEAR}-{MAX_F1_YEAR}, "
            f"invalid: {invalid_years}"
        )
    
    logger.debug("Years validation passed: %s", years)

def validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    context: str
) -> None:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must exist
        context: Description of DataFrame for error messages
        
    Raises:
        ValueError: If required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{context} DataFrame missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    
    logger.debug("%s validation passed: %d columns", context, len(df.columns))

# DATA LOADING FUNCTIONS

def load_driver_profiles(years: List[int]) -> pd.DataFrame:
    """
    Load and concatenate driver telemetry profiles across multiple seasons.
    
    Reads CSV files from data/driver/ directory containing practice session
    telemetry for each driver (throttle usage, braking, tire degradation, etc.).
    
    Args:
        years: List of F1 season years to load (e.g., [2022, 2023, 2024]).
               Each year must have a corresponding CSV file.
    
    Returns:
        Combined DataFrame with all driver sessions across specified years.
        
        Key columns include:
        - driver: 3-letter driver code (VER, HAM, etc.)
        - max_throttle_ratio: Percentage of lap at full throttle
        - brake_max_g: Peak braking deceleration (g-force)
        - drs_activations: Number of DRS zone activations
        - degradation_slope: Tire wear rate (seconds per lap)
        - session: Session type (FP1, FP2, FP3, Q, R, SQ, S)
        - year, event, location: Race identifiers
    
    Raises:
        ValueError: If years list is invalid
        FileNotFoundError: If no driver profile files exist for any provided year
    
    Example:
        >>> profiles = load_driver_profiles([2024])
        INFO - Loaded 2337 rows from 2024
        INFO - Total driver sessions: 2337
        >>> print(profiles[['driver', 'session', 'max_throttle_ratio']].head())
    """
    validate_years(years)
        
    dfs = []
    for year in years:
        path = DRIVER_PROFILE_DIR / f'{year}_driver_profiles.csv'
        
        try:
            if path.exists():
                df = pd.read_csv(path)
                dfs.append(df)
                logger.info("✅ Loaded %d rows from %d", len(df), year)
            else:
                logger.warning("⚠️ Missing file: %s", path)
        except Exception as e:
            logger.error("Failed to load %s: %s", path, str(e))
            raise
    
    if not dfs:
        raise FileNotFoundError(
            f"No driver profiles found in {DRIVER_PROFILE_DIR} "
            f"for years {years}"
        )
    
    combined = pd.concat(dfs, ignore_index=True)

    from helpers.team_name_mapping import normalize_team_column
    # Canonicalize team names   
    combined = normalize_team_column(combined, col='team')
    
    # Validate expected columns exist
    expected_cols = ['driver', 'session', 'year', 'event']
    validate_dataframe_columns(combined, expected_cols, "Driver profiles")
    
    logger.info("Total driver sessions: %d", len(combined))
    logger.info("Sessions: %s", combined['session'].unique().tolist())
    
    return combined

def load_circuit_profiles(years: List[int]) -> pd.DataFrame:
    """
    Load and concatenate circuit characteristic profiles across multiple seasons.
    
    Reads CSV files from data/circuit/ directory containing track layout
    characteristics (corners, speeds, elevation, etc.) for each session.
    
    Args:
        years: List of F1 season years to load (e.g., [2022, 2023, 2024]).
    
    Returns:
        Combined DataFrame with track characteristics per session.
        
        Key columns include:
        - slow_corners, medium_corners, fast_corners: Corner counts by speed
        - chicanes: Number of chicane sequences
        - avg_speed, top_speed: Track speed characteristics (km/h)
        - real_altitude: Circuit elevation above sea level (meters)
        - air_temp_avg, track_temp_avg: Average temperatures (°C)
        - rain_detected: Boolean indicating rainfall during session
        - year, event, session: Identifiers
    
    Raises:
        ValueError: If years list is invalid
        FileNotFoundError: If no circuit profile files exist for any provided year
    """
    validate_years(years)
        
    dfs = []
    for year in years:
        path = CIRCUIT_PROFILE_DIR / f'{year}_circuit_profiles.csv'
        
        try:
            if path.exists():
                df = pd.read_csv(path)
                dfs.append(df)
                logger.info("✅ Loaded %d rows from %d", len(df), year)
            else:
                logger.warning("⚠️ Missing file: %s", path)
        except Exception as e:
            logger.error("Failed to load %s: %s", path, str(e))
            raise
    
    if not dfs:
        raise FileNotFoundError(
            f"No circuit profiles found in {CIRCUIT_PROFILE_DIR} "
            f"for years {years}"
        )
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Validate expected columns
    expected_cols = ['year', 'event', 'session', 'slow_corners', 'fast_corners']
    validate_dataframe_columns(combined, expected_cols, "Circuit profiles")
    
    logger.info("Total circuit sessions: %d", len(combined))
    
    return combined

def load_qualifying_results(years: List[int]) -> pd.DataFrame:
    """
    Load official FIA qualifying results across multiple seasons.
    
    Reads CSV files from data/predictions/ssot/ directory containing
    official qualifying classification data (the target variable for ML).
    
    Args:
        years: List of F1 season years to load (e.g., [2022, 2023, 2024]).
    
    Returns:
        Combined DataFrame with qualifying results (target variable).
        
        Key columns include:
        - qualifying_position: Final qualifying position (1-20) - TARGET VARIABLE
        - Abbreviation: 3-letter driver code (VER, HAM, etc.)
        - TeamName: Constructor name (Red Bull Racing, Mercedes, etc.)
        - EventName: Race name (Bahrain Grand Prix, etc.)
        - Q1, Q2, Q3: Qualifying session lap times (when available)
        - year: Season year (added by this function)
    
    Raises:
        ValueError: If years list invalid or 'Position' column missing
        FileNotFoundError: If no qualifying result files exist for any provided year
    """
    validate_years(years)
        
    dfs = []
    for year in years:
        path = RESULTS_DIR / f'{year}_qualifying.csv'
        
        try:
            if path.exists():
                df = pd.read_csv(path)
                df['year'] = year

                df['Position'] = df['qualifying_position']

                # Use 'Position' as our target (not ClassifiedPosition)
                if 'Position' not in df.columns:
                    raise ValueError(
                        f"'Position' column not found in {path}. "
                        f"Available columns: {df.columns.tolist()}"
                    )
                
                dfs.append(df)
                logger.info("✅ Loaded %d results from %d", len(df), year)
            else:
                logger.warning("⚠️ Missing file: %s", path)
        except Exception as e:
            logger.error("Failed to load %s: %s", path, str(e))
            raise
    
    if not dfs:
        raise FileNotFoundError(
            f"No qualifying results found in {RESULTS_DIR} "
            f"for years {years}"
        )
    
    combined = pd.concat(dfs, ignore_index=True)

    # keep only rows with a valid quali position
    valid_mask = (
        combined['qualifying_position'].notna()
        & combined['qualifying_position'].between(
            MIN_QUALIFYING_POSITION,
            MAX_QUALIFYING_POSITION
        )
    )

    dropped = len(combined) - valid_mask.sum()
    if dropped > 0:
        logger.warning(
            "Dropping %d rows with invalid qualifying_position (NaN or out of [%d, %d])",
            dropped, MIN_QUALIFYING_POSITION, MAX_QUALIFYING_POSITION
        )

    combined = combined[valid_mask].copy()

    logger.info("Total qualifying results: %d", len(combined))
    logger.info(
        "Position range: %.0f to %.0f",
        combined['qualifying_position'].min(),
        combined['qualifying_position'].max()
    )

    #team name cleanup + mapping
    from helpers.team_name_mapping import normalize_team_column
    # Canonicalize team names
    combined = normalize_team_column(combined, col='team')

    # Validate target variable
    if not combined['qualifying_position'].between(
        MIN_QUALIFYING_POSITION, 
        MAX_QUALIFYING_POSITION
    ).all():
        invalid = combined[
            ~combined['qualifying_position'].between(
                MIN_QUALIFYING_POSITION, 
                MAX_QUALIFYING_POSITION
            )
        ]
        logger.warning(
            "Found %d qualifying positions outside range %d-%d",
            len(invalid), MIN_QUALIFYING_POSITION, MAX_QUALIFYING_POSITION
        )
    
    return combined

# DATA MERGING FUNCTIONS

def merge_driver_circuit_data(
    drivers: pd.DataFrame,
    circuits: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge driver telemetry with circuit characteristics for each session.
    
    Combines driver-level performance data (throttle, braking, etc.) with
    track-level characteristics (corners, speeds, etc.) by matching on
    year + event + session.
    
    Args:
        drivers: Driver telemetry DataFrame from load_driver_profiles().
                 Must contain columns: year, event, session, driver, etc.
        circuits: Circuit characteristics DataFrame from load_circuit_profiles().
                  Must contain columns: year, event, session, slow_corners, etc.
    
    Returns:
        Merged DataFrame with both driver and circuit features.
        
        Shape: Same number of rows as drivers (one row per driver-session).
        Columns: All driver columns + all circuit columns (except duplicates).
    
    Raises:
        ValueError: If required merge keys missing from either DataFrame
        RuntimeError: If merge produces unexpected results
    
    Example:
        >>> merged = merge_driver_circuit_data(drivers, circuits)
        INFO - Merging driver telemetry with circuit data...
        INFO - Merged shape: (3501, 36)
        INFO - Missing circuit data: 233 rows
    """
    
    # Validate merge keys exist
    try:
        validate_dataframe_columns(drivers, MERGE_KEYS, "Drivers")
        validate_dataframe_columns(circuits, MERGE_KEYS, "Circuits")
    except ValueError as e:
        logger.error("Merge validation failed: %s", str(e))
        raise
    
    # Drop duplicate columns from circuits before merge
    circuits_clean = circuits.drop(columns=['location'], errors='ignore')
    
    # Merge with validation
    try:
        merged = drivers.merge(
            circuits_clean,
            on=MERGE_KEYS,
            how='left',
            suffixes=('_driver', '_circuit'),
            validate='m:1'  # Many drivers to one circuit per session
        )
    except Exception as e:
        logger.error("Merge failed: %s", str(e))
        raise RuntimeError(f"Failed to merge driver and circuit data: {e}")
    
    # Validate merge didn't lose rows
    if len(merged) != len(drivers):
        logger.error(
            "Merge changed row count: %d  %d",
            len(drivers), len(merged)
        )
        raise RuntimeError(
            f"Merge lost {len(drivers) - len(merged)} rows. "
            "Check for duplicate merge keys."
        )
    
    missing_circuit = merged['avg_speed'].isna().sum()
    if missing_circuit > 0:
        logger.warning(
            "⚠️ Missing circuit data for %d/%d rows (%.1f%%)",
            missing_circuit, len(merged),
            100 * missing_circuit / len(merged)
        )
    
    logger.info("✅ Merged shape: %s", merged.shape)
    
    return merged

def merge_with_qualifying_results(
    features: pd.DataFrame,
    results: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge practice session features with qualifying results (target variable).
    
    Final merge step that adds the target variable (qualifying_position) to
    the feature matrix. Only keeps rows where both features AND target exist.
    
    Args:
        features: Aggregated practice features (one row per driver-race).
                  Must contain: year, event, driver (3-letter code).
        results: Qualifying results from load_qualifying_results().
                 Must contain: year, EventName, Abbreviation, qualifying_position.
    
    Returns:
        Complete ML-ready DataFrame with features + target variable.
        
        Rows: Fewer than input features (drops drivers without quali results).
        Columns: All feature columns + 'qualifying_position' + 'TeamName'.
    
    Raises:
        ValueError: If merge keys missing or target variable invalid
    
    Example:
        >>> final = merge_with_qualifying_results(features, results)
        INFO - Merging with qualifying results (target variable)...
        INFO - Features shape: (502, 35)
        INFO - Results shape: (479, 31)
        INFO - ✅ Merged shape: (479, 37)
        INFO - Dropped 23 rows without qualifying result
    """
    logger.info("Features shape: %s", features.shape)
    logger.info("Results shape: %s", results.shape)
    
    # Validate required columns
    try:
        validate_dataframe_columns(
            features, ['year', 'event', 'driver'], "Features"
        )
        validate_dataframe_columns(
            results, ['year', 'event', 'driver', 'qualifying_position'],
            "Results"
        )
    except ValueError as e:
        logger.error("Merge validation failed: %s", str(e))
        raise
    
    # Rename for merge
    results_clean = results[[
        'year', 'event', 'driver',
        'qualifying_position', 'team'
    ]].copy()
    
    # Merge
    try:
        merged = features.merge(
            results_clean,
            on=['year', 'event', 'driver'],
            how='inner',  # Only keep rows with both features AND target
            suffixes=('', '_results'),
            validate='1:1'  # One-to-one relationship expected
        )
    except Exception as e:
        logger.error("Results merge failed: %s", str(e))
        raise RuntimeError(f"Failed to merge with qualifying results: {e}")
    
    dropped = len(features) - len(merged)
    if dropped > 0:
        logger.warning(
            "Dropped %d rows without qualifying result (%.1f%%)",
            dropped, 100 * dropped / len(features)
        )
    
    # Validate target variable
    missing_target = merged['qualifying_position'].isna().sum()
    if missing_target > 0:
        raise ValueError(
            f"Target variable has {missing_target} missing values after merge"
        )
    
    invalid_positions = ~merged['qualifying_position'].between(
        MIN_QUALIFYING_POSITION,
        MAX_QUALIFYING_POSITION
    )
    if invalid_positions.any():
        bad_vals = merged.loc[invalid_positions, 'qualifying_position'].unique()
        raise ValueError(
            f"Target variable has invalid positions: {bad_vals}. "
            f"Must be between {MIN_QUALIFYING_POSITION}-{MAX_QUALIFYING_POSITION}"
        )
    
    logger.info("✅ Merged shape: %s", merged.shape)
    logger.info("Target variable range: %.0f to %.0f",
                merged['qualifying_position'].min(),
                merged['qualifying_position'].max())
    
    return merged

def compute_circuit_overtaking_features(
    df: pd.DataFrame,
    lookback_years: int = 3
) -> pd.DataFrame:
    """
    Compute circuit-specific overtaking difficulty metrics.
    
    For each circuit, computes historical position changes to identify:
    - Processional tracks (Monaco, Hungary): Low position change
    - Overtaking tracks (Monza, Spa): High position change
    
    This is CRITICAL for race prediction because:
    - At Monaco: Qualifying ≈ Race (hard to overtake)
    - At Monza: Many position changes (easy to overtake)
    
    Args:
        df: DataFrame with race and qualifying positions
        lookback_years: Years of history to use (default: 3)
        
    Returns:
        DataFrame with circuit overtaking metrics per event-year
        
    Features created:
        - circuit_avg_position_change: Mean position change (- = net gains)
        - circuit_std_position_change: Variability of position changes
        - circuit_abs_position_change: Average absolute changes (overtaking ease)
        - circuit_max_gain: Biggest position gain seen historically
        - circuit_max_loss: Biggest position loss seen historically
        
    Example:
        >>> overtaking = compute_circuit_overtaking_features(df)
        >>> print(overtaking[overtaking['event'] == 'Monaco Grand Prix'])
        # circuit_abs_position_change: 1.2 (low - hard to overtake)
        >>> print(overtaking[overtaking['event'] == 'Italian Grand Prix'])
        # circuit_abs_position_change: 3.8 (high - easy to overtake)
    """
    import pandas as pd
    
    # Need both qualifying and race positions
    df_race = df[
        (df['qualifying_position'].notna()) & 
        (df['race_position'].notna())
    ].copy()
    
    if df_race.empty:
        return pd.DataFrame()
    
    # Compute position change
    df_race['position_change'] = (
        df_race['race_position'] - df_race['qualifying_position']
    )
    
    results = []
    
    # For each circuit-year, compute stats from previous years
    unique_combos = df_race[['event', 'year']].drop_duplicates()
    
    for _, row in unique_combos.iterrows():
        event = row['event']
        target_year = row['year']
        
        # Get historical data (previous years only, no leakage!)
        historical = df_race[
            (df_race['event'] == event) &
            (df_race['year'] < target_year) &
            (df_race['year'] >= target_year - lookback_years)
        ]
        
        if len(historical) < 5:  # Need minimum data
            continue
        
        position_changes = historical['position_change']
        
        results.append({
            'event': event,
            'year': target_year,
            'circuit_avg_position_change': position_changes.mean(),
            'circuit_std_position_change': position_changes.std(),
            'circuit_abs_position_change': position_changes.abs().mean(),
            'circuit_max_gain': position_changes.min(),  # Most negative = biggest gain
            'circuit_max_loss': position_changes.max(),  # Most positive = biggest loss
            'circuit_overtaking_samples': len(historical)
        })
    
    return pd.DataFrame(results)

def compute_driver_overtaking_skill(
    df: pd.DataFrame,
    lookback_years: int = 3
) -> pd.DataFrame:
    """
    Compute driver-specific overtaking skill.
    
    Some drivers are better at overtaking than others. This computes
    historical average position gains/losses per driver.
    
    Args:
        df: DataFrame with race and qualifying positions
        lookback_years: Years of history to use
        
    Returns:
        DataFrame with driver overtaking metrics per driver-year
        
    Features:
        - driver_avg_position_change: Average position change (- = gains positions)
        - driver_overtaking_success_rate: % of races where gained positions
        - driver_defensive_success_rate: % of races where lost <2 positions
    """
    import pandas as pd
    
    df_race = df[
        (df['qualifying_position'].notna()) & 
        (df['race_position'].notna())
    ].copy()
    
    if df_race.empty:
        return pd.DataFrame()
    
    df_race['position_change'] = (
        df_race['race_position'] - df_race['qualifying_position']
    )
    
    results = []
    
    # For each driver-year
    unique_combos = df_race[['driver', 'year']].drop_duplicates()
    
    for _, row in unique_combos.iterrows():
        driver = row['driver']
        target_year = row['year']
        
        # Historical data (previous years only)
        historical = df_race[
            (df_race['driver'] == driver) &
            (df_race['year'] < target_year) &
            (df_race['year'] >= target_year - lookback_years)
        ]
        
        if len(historical) < 3:  # Minimum races
            continue
        
        position_changes = historical['position_change']
        
        results.append({
            'driver': driver,
            'year': target_year,
            'driver_avg_position_change': position_changes.mean(),
            'driver_std_position_change': position_changes.std(),
            'driver_overtaking_success_rate': (position_changes < 0).mean(),
            'driver_defensive_success_rate': (position_changes.abs() <= 2).mean(),
            'driver_overtaking_samples': len(historical)
        })
    
    return pd.DataFrame(results)

# FEATURE AGGREGATION FUNCTION

def aggregate_practice_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple practice sessions into single feature vector per driver-race.
    
    Transforms data from "one row per driver-session" to "one row per driver-race"
    by intelligently aggregating FP1/FP2/FP3/SQ sessions. Handles sprint vs normal
    weekends differently due to different practice session availability.
    
    Strategy:
        1. Best performance: max() across all sessions (teams show true pace at least once)
        2. Average performance: mean() for consistency indicator
        3. Session-specific: FP3 for normal weekends, Sprint Quali for sprint weekends
        4. Weather: aggregate rain detection and temperature stats
        5. Circuit: use static values (same for all sessions at a track)
    
    Args:
        df: Merged driver+circuit data with one row per driver-session.
            Must contain: year, event, driver, session, max_throttle_ratio,
            brake_max_g, slow_corners, etc.
    
    Returns:
        Aggregated DataFrame with one row per driver-race combination.
        
        New features created:
        - best_throttle_ratio: Peak performance across all practice sessions
        - avg_throttle_ratio: Average consistency indicator
        - fp3_throttle_ratio: FP3 specific (NaN for sprint weekends)
        - sprint_quali_throttle: Sprint Qualifying specific (NaN for normal weekends)
        - is_sprint_weekend: Boolean flag for weekend format
        - has_sprint_quali_data: Boolean indicating Sprint Quali data availability
        - rain_in_practice: Boolean if any rainfall detected
        - slow_corner_pct, medium_corner_pct, fast_corner_pct: Corner distribution
        - total_corners: Sum of all corner types
    
    Raises:
        ValueError: If required columns missing
    
    Example:
        >>> aggregated = aggregate_practice_sessions(merged_data)
        INFO - Aggregating practice sessions per driver-race...
        INFO - Session patterns: 62 normal weekends, 6 sprint weekends
        INFO - ✅ Aggregated to 1421 driver-race combinations
    """
    
    # Validate required columns
    required = ['year', 'event', 'driver', 'session', 'max_throttle_ratio']
    try:
        validate_dataframe_columns(df, required, "Practice sessions")
    except ValueError as e:
        logger.error("Aggregation validation failed: %s", str(e))
        raise
    
    # Show session patterns
    session_counts = df.groupby(['year', 'event'])['session'].unique()
    normal_count = sum(1 for sessions in session_counts if 'SQ' not in sessions)
    sprint_count = sum(1 for sessions in session_counts if 'SQ' in sessions)
    
    logger.info("Session patterns: %d normal weekends, %d sprint weekends",
                normal_count, sprint_count)
    
    # Group by driver + race
    grouped = df.groupby(['year', 'event', 'driver'], dropna=False)
    logger.info("Unique driver-race combinations: %d", len(grouped))
    
    aggregated_rows = []
    
    for (year, event, driver), group in grouped:
        try:
            row = _aggregate_single_driver_race(year, event, driver, group)
            aggregated_rows.append(row)
        except Exception as e:
            logger.error(
                "Failed to aggregate %s - %s - %s: %s",
                year, event, driver, str(e)
            )
            # Continue processing other rows instead of failing completely
            continue
    
    if not aggregated_rows:
        raise RuntimeError("No rows successfully aggregated")
    
    result = pd.DataFrame(aggregated_rows)
    
    logger.info("✅ Aggregated to %d driver-race combinations", len(result))
    logger.info("Sprint weekends: %d", result['is_sprint_weekend'].sum())
    logger.info("Normal weekends: %d", (~result['is_sprint_weekend']).sum())
    
    return result

def _aggregate_single_driver_race(
    year: int,
    event: str,
    driver: str,
    group: pd.DataFrame
) -> Dict[str, Any]:
    """
    Helper function to aggregate single driver-race combination.
    
    Args:
        year: Season year
        event: Race name
        driver: 3-letter driver code
        group: DataFrame of all sessions for this driver-race
        
    Returns:
        Dictionary with aggregated features
    """
    row = {
        'year': year,
        'event': event,
        'driver': driver,
        'grand_prix': group['grand_prix'].iloc[0],
        'location': group['location'].iloc[0],
    }
    
    # Detect weekend format
    available_sessions = group['session'].tolist()
    is_sprint = 'SQ' in available_sessions
    row['is_sprint_weekend'] = is_sprint
    row['sessions_available'] = ','.join(sorted(available_sessions))
    
    # Best performance across all sessions
    row['best_throttle_ratio'] = group['max_throttle_ratio'].max()
    row['best_brake_max_g'] = group['brake_max_g'].max()
    row['best_drs_activations'] = group['drs_activations'].max()
    
    # Average performance (consistency)
    row['avg_throttle_ratio'] = group['max_throttle_ratio'].mean()
    row['avg_brake_max_g'] = group['brake_max_g'].mean()
    row['avg_degradation_slope'] = group['degradation_slope'].mean()
    
    # Session-specific features
    if is_sprint:
        row = _add_sprint_weekend_features(row, group)
    else:
        row = _add_normal_weekend_features(row, group)
    
    # Weather features
    row['rain_in_practice'] = (group['avg_rainfall'].max() > 0)
    row['avg_track_temp'] = group['avg_track_temp'].mean()
    row['track_temp_std'] = group['avg_track_temp'].std()
    
    # Circuit features
    row['real_altitude'] = group['real_altitude'].iloc[0]
    row['slow_corners'] = group['slow_corners'].iloc[0]
    row['medium_corners'] = group['medium_corners'].iloc[0]
    row['fast_corners'] = group['fast_corners'].iloc[0]
    
    total_corners = row['slow_corners'] + row['medium_corners'] + row['fast_corners']
    row['total_corners'] = total_corners
    
    if total_corners > 0:
        row['slow_corner_pct'] = row['slow_corners'] / total_corners
        row['medium_corner_pct'] = row['medium_corners'] / total_corners
        row['fast_corner_pct'] = row['fast_corners'] / total_corners
    else:
        row['slow_corner_pct'] = np.nan
        row['medium_corner_pct'] = np.nan
        row['fast_corner_pct'] = np.nan
    
    row['chicanes'] = group['chicanes'].iloc[0]
    row['avg_speed_circuit'] = group['avg_speed'].iloc[0]
    row['top_speed_circuit'] = group['top_speed'].iloc[0]
    
    return row

def _add_sprint_weekend_features(
    row: Dict[str, Any],
    group: pd.DataFrame
) -> Dict[str, Any]:
    """
    Add features specific to sprint weekends (FP1 + Sprint Qualifying).
    
    Args:
        row: Existing row dictionary
        group: DataFrame of sessions for this driver-race
        
    Returns:
        Updated row dictionary with sprint-specific features
    """
    fp1 = group[group['session'] == 'FP1']
    sq = group[group['session'] == 'SQ']
    
    # FP1 features
    if not fp1.empty:
        row['fp1_throttle_ratio'] = fp1['max_throttle_ratio'].iloc[0]
        row['fp1_track_temp'] = fp1['avg_track_temp'].iloc[0]
    else:
        row['fp1_throttle_ratio'] = np.nan
        row['fp1_track_temp'] = np.nan
    
    # Sprint Qualifying features (most predictive for sprint weekends)
    if not sq.empty:
        row['sprint_quali_throttle'] = sq['max_throttle_ratio'].iloc[0]
        row['sprint_quali_brake_g'] = sq['brake_max_g'].iloc[0]
        row['has_sprint_quali_data'] = True
    else:
        row['sprint_quali_throttle'] = np.nan
        row['sprint_quali_brake_g'] = np.nan
        row['has_sprint_quali_data'] = False
    
    # No FP3 on sprint weekends
    row['fp3_throttle_ratio'] = np.nan
    row['fp3_brake_max_g'] = np.nan
    
    return row

def _add_normal_weekend_features(
    row: Dict[str, Any],
    group: pd.DataFrame
) -> Dict[str, Any]:
    """
    Add features specific to normal weekends (FP1 + FP2 + FP3).
    
    Args:
        row: Existing row dictionary
        group: DataFrame of sessions for this driver-race
        
    Returns:
        Updated row dictionary with normal weekend features
    """
    fp1 = group[group['session'] == 'FP1']
    fp3 = group[group['session'] == 'FP3']
    
    # FP3 features (most predictive for normal weekends)
    if not fp3.empty:
        row['fp3_throttle_ratio'] = fp3['max_throttle_ratio'].iloc[0]
        row['fp3_brake_max_g'] = fp3['brake_max_g'].iloc[0]
        row['fp3_track_temp'] = fp3['avg_track_temp'].iloc[0]
    else:
        row['fp3_throttle_ratio'] = np.nan
        row['fp3_brake_max_g'] = np.nan
        row['fp3_track_temp'] = np.nan
    
    # FP1 features for comparison
    if not fp1.empty:
        row['fp1_throttle_ratio'] = fp1['max_throttle_ratio'].iloc[0]
        row['fp1_track_temp'] = fp1['avg_track_temp'].iloc[0]
    else:
        row['fp1_throttle_ratio'] = np.nan
        row['fp1_track_temp'] = np.nan 

    # No sprint qualifying on normal weekends
    row['sprint_quali_throttle'] = np.nan
    row['sprint_quali_brake_g'] = np.nan
    row['has_sprint_quali_data'] = False
    
    return row

# DATA CLEANING FUNCTIONS

def fix_missing_circuit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix missing circuit features by imputing from same track in other years.
    
    Handles cases where circuit profile extraction failed for specific races
    (e.g., 2023 Abu Dhabi) by copying circuit characteristics from the same
    track in adjacent years. Also drops altitude feature due to low variance.
    
    Strategy:
        1. Drop real_altitude feature (only 2/24 tracks affected, low value)
        2. Identify rows with missing circuit data (slow_corners == NaN)
        3. For each missing row, find same track from different year
        4. Copy circuit features from donor row
    
    Args:
        df: Feature DataFrame with potential missing circuit data.
            Must contain: year, event, slow_corners, fast_corners, etc.
    
    Returns:
        Cleaned DataFrame with all circuit features filled.
        
        Changes:
        - real_altitude column removed
        - Missing circuit features imputed from adjacent years
        - All corner counts and percentages filled
    
    Raises:
        RuntimeError: If imputation fails to fix all missing data
    
    Example:
        >>> fixed = fix_missing_circuit_data(df)
        INFO - Fixing missing circuit data...
        INFO - Dropped real_altitude feature
        INFO - ⚠️ Rows with missing circuit data: 119
        INFO - ✅ Imputed circuit features
        INFO - Remaining missing: 0
    
    Notes:
        - Assumes tracks don't change significantly year-to-year (valid assumption)
        - If same track not found in other years, those rows remain NaN
        - Typically affects <10% of data (e.g., 119/1353 rows = 8.8%)
        - Most common causes: API failures, session cancellations, data pipeline issues
    """
    
    # 1. Drop altitude (not useful anyway)
    if 'real_altitude' in df.columns:
        df = df.drop(columns=['real_altitude'])
        logger.info("✅ Dropped real_altitude feature")
    
    # 2. Identify rows with missing circuit data
    circuit_cols = [
        'slow_corners', 'fast_corners', 'avg_speed_circuit',
        'slow_corner_pct', 'medium_corner_pct', 'fast_corner_pct'
    ]
    
    missing_mask = df[circuit_cols[0]].isna()
    missing_count = missing_mask.sum()
    
    if missing_count > 0:
        logger.warning(
            "⚠️ Rows with missing circuit data: %d (%.1f%%)",
            missing_count, 100 * missing_count / len(df)
        )
    else:
        logger.info("✅ No missing circuit data!")
        return df
    
    # 3. For each missing row, find same track from different year
    imputed_count = 0
    failed_imputation = []
    
    for idx in df[missing_mask].index:
        event = df.loc[idx, 'event']
        year = df.loc[idx, 'year']
        
        # Find same event from other years with valid data
        same_track_other_years = df[
            (df['event'] == event) &
            (df['year'] != year) &
            (df[circuit_cols[0]].notna())
        ]
        
        if len(same_track_other_years) > 0:
            # Use circuit features from first available year
            donor_row = same_track_other_years.iloc[0]
            
            for col in circuit_cols:
                df.loc[idx, col] = donor_row[col]
            
            # Also update related columns
            for col in ['total_corners', 'chicanes', 'top_speed_circuit']:
                if col in df.columns and col in donor_row.index:
                    df.loc[idx, col] = donor_row[col]
            
            imputed_count += 1
        else:
            failed_imputation.append(f"{year} {event}")
            logger.warning(
                "No donor data found for %s %s",
                year, event
            )
    
    # 4. Verify fix
    remaining_missing = df[circuit_cols[0]].isna().sum()
    
    if remaining_missing > 0:
        logger.error(
            "Failed to impute all missing data. Remaining: %d",
            remaining_missing
        )
        if failed_imputation:
            logger.error("Could not impute: %s", failed_imputation)
        raise RuntimeError(
            f"⚠️ Imputation incomplete. {remaining_missing} rows still missing circuit data"
        )
    
    logger.info("✅ Imputed circuit features for %d rows", imputed_count)
    logger.info("Remaining missing: %d", remaining_missing)
    
    return df

# HISTORICAL FEATURES (PLACEHOLDER)

def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical performance features for each driver-track combination.
    
    PLACEHOLDER FUNCTION - Not yet implemented.
    
    Planned features:
        1. driver_track_avg_quali_3yr: Driver's avg quali position at this track (last 3 years)
        2. team_track_avg_quali_3yr: Team's avg quali position at this track (last 3 years)
        3. driver_recent_form: Driver's avg quali position in last 3 races
    
    Args:
        df: Dataset with year, event, driver, qualifying_position columns.
            Must be sorted by date to avoid data leakage.
    
    Returns:
        Dataset with 3 new historical features (currently unchanged).
    
    Important:
        Must avoid data leakage by only using PAST data to predict future.
        For 2022 data: no historical features available (will be NaN).
        For 2023+: look back to previous years only.
    
    Example:
        >>> df_with_history = add_historical_features(df)
        # Currently returns df unchanged - implementation pending
    
    TODO:
        - Implement temporal lookback logic with proper data leakage guards
        - Add driver historical performance at each track
        - Add team historical performance at each track
        - Add rolling 3-race form for each driver
        - Handle cold start problem (new drivers/teams)
    """
    logger.warning("Historical features not yet implemented - skipping")
    # Implementation pending - Phase 2
    return df

# MAIN PIPELINE ORCHESTRATOR

def prepare_qualifying_dataset(
    years: List[int] = [2022, 2023, 2024]
) -> pd.DataFrame:
    """
    Build complete ML-ready feature matrix for qualifying position prediction.
    
    End-to-end pipeline that transforms raw F1 telemetry into clean dataset
    with one row per driver-race and 36+ features ready for model training.
    
    Pipeline steps:
        1. Load driver profiles (telemetry from practice sessions)
        2. Load circuit profiles (track characteristics)
        3. Load qualifying results (target variable)
        4. Filter to practice sessions only (FP1/FP2/FP3/SQ - no data leakage)
        5. Merge driver telemetry with circuit characteristics
        6. Aggregate multiple practice sessions per driver-race
        7. Merge with qualifying results to add target variable
        8. Fix missing circuit data via imputation
    
    Args:
        years: List of F1 seasons to include (default: [2022, 2023, 2024]).
               Each year should have corresponding data files in data/ directory.
    
    Returns:
        Complete ML-ready DataFrame with features + target variable.
        
        Shape: ~440-480 rows per year (20 drivers × 22-24 races).
               Example: 3 years → ~1,350 rows.
        
        Columns (~37 total):
        - Identifiers: year, event, driver, grand_prix, location, TeamName
        - Target: qualifying_position (1-20)
        - Driver performance: best/avg throttle ratio, brake_max_g, drs_activations
        - Session-specific: fp3_throttle_ratio, sprint_quali_throttle
        - Circuit: slow/medium/fast_corner_pct, total_corners, chicanes, avg_speed
        - Weather: rain_in_practice, avg_track_temp, track_temp_std
        - Flags: is_sprint_weekend, has_sprint_quali_data
    
    Raises:
        ValueError: If years list invalid or data validation fails
        FileNotFoundError: If no data files exist for any provided year
        RuntimeError: If pipeline fails at any stage
    
    Example:
        >>> # Build dataset for 3 seasons
        >>> df = prepare_qualifying_dataset([2022, 2023, 2024])
        ============================================================
        🏗️  Building Qualifying Prediction Dataset
        ============================================================
        
        INFO - Loading driver profiles...
        INFO - Loaded 2089 rows from 2022
        INFO - Loaded 2123 rows from 2023
        INFO - Loaded 2337 rows from 2024
        ...
        ============================================================
        ✅ DATASET COMPLETE!
        ============================================================
        
        >>> print(df.shape)
        (1353, 37)
        
        >>> # Check for missing values
        >>> print(df.isnull().sum().sum())
        0  # Zero missing values after imputation
    
    Notes:
        - Output is deterministic (same input  same output)
        - Safe to re-run (idempotent pipeline)
        - Handles sprint vs normal weekends automatically
        - Drops drivers without qualifying results (DNS/DSQ cases)
        - Imputes missing circuit data from adjacent years
        - No data leakage (only uses practice session data, not qualifying itself)
    
    Data Quality:
        - Typical data loss: ~5% of rows (missing qualifying results)
        - Missing circuit data: ~8-10% (fixed via imputation)
        - Final dataset: 100% complete (zero missing values)
    
    Performance:
        - Runtime: ~30-60 seconds for 3 years of data
        - Memory: ~50-100 MB for full dataset
        - Output CSV: ~1-2 MB (1,350 rows × 37 columns)
    """
    # Visual header (print)
    print("\n" + "=" * 60)
    print("🏗️  Building Qualifying Prediction Dataset")
    print("=" * 60 + "\n")
    
    try:
        # Step 1: Load raw data
        logger.info("Step 1/7: Loading driver profiles for years %s", years)
        drivers = load_driver_profiles(years)
        
        logger.info("Step 2/7: Loading circuit profiles")
        circuits = load_circuit_profiles(years)
        
        logger.info("Step 3/7: Loading qualifying results")
        results = load_qualifying_results(years)
        
        # Step 2: Filter to practice sessions only
        logger.info("Step 4/7: Filtering to practice sessions only")
        drivers_practice = drivers[drivers['session'].isin(PRACTICE_SESSIONS)].copy()
        circuits_practice = circuits[circuits['session'].isin(PRACTICE_SESSIONS)].copy()
        logger.info("   Driver practice: %d rows, Circuit practice: %d rows",
                   len(drivers_practice), len(circuits_practice))
        
        # Step 3: Merge and aggregate
        logger.info("Step 5/7: Merging and aggregating features")
        combined = merge_driver_circuit_data(drivers_practice, circuits_practice)
        combined = aggregate_practice_sessions(combined)
        
        # Step 4: Add target variable
        logger.info("Step 6/7: Adding target variable and fixing missing data")
        final = merge_with_qualifying_results(combined, results)
        final = fix_missing_circuit_data(final)

        logger.info("Step 6.5/7: Filling rookie features with team-based priors")
        try:
            try:
                # When run as a module: python -m helpers.feature_engineering
                from helpers.team_priors import load_team_baselines, fill_features_batch
            except ModuleNotFoundError:
                # When run as a script: python helpers/feature_engineering.py
                from team_priors import load_team_baselines, fill_features_batch

            team_baselines = load_team_baselines('models/team_baselines.json')
            final = fill_features_batch(final, team_baselines, rookie_penalty=1.5)
            logger.info("   ✅ Rookie features filled using team baselines")
        except FileNotFoundError:
            logger.warning("   ⚠️ Team baselines not found, using fallback filling")
            for col in final.columns:
                if final[col].isnull().any() and final[col].dtype in ['float64', 'int64']:
                    final[col] = final[col].fillna(final[col].median())

        
        # Step 5: Validate final dataset
        logger.info("Step 7/7: Validating dataset quality")
        try:
            from helpers.validation import validate_feature_dataframe
        except ModuleNotFoundError:
            from validation import validate_feature_dataframe
        
        required_features = [
            'best_throttle_ratio', 'avg_throttle_ratio', 'best_brake_max_g',
            'slow_corner_pct', 'fast_corner_pct', 'is_sprint_weekend'
        ]
        
        try:
            validate_feature_dataframe(final, required_features, 'qualifying_position')
        except ValueError as e:
            logger.error("Dataset validation failed: %s", str(e))
            raise
        
        # Visual footer (print)
        print("\n" + "=" * 60)
        print(f"✅ DATASET COMPLETE - {final.shape[0]} rows × {final.shape[1]} columns")
        print("=" * 60 + "\n")
        
        return final
        
    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise

# COMMAND-LINE INTERFACE

# COMMAND-LINE INTERFACE

def parse_arguments():
    """
    Parse command-line arguments for feature engineering pipeline.
    
    Returns:
        Namespace with parsed arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate ML-ready features for F1 qualifying prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate features for default years (2022-2024)
  python helpers/feature_engineering.py
  
  # Generate features for specific years
  python helpers/feature_engineering.py --years 2023 2024
  
  # Specify custom output path
  python helpers/feature_engineering.py --output data/custom/features.csv
  
  # Enable verbose logging
  python helpers/feature_engineering.py --verbose
  
  # Disable validation step
  python helpers/feature_engineering.py --no-validate
        """
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=[2022, 2023, 2024],
        help='F1 season years to process (default: 2022 2023 2024)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/qualifying_features.csv',
        help='Output CSV file path (default: data/processed/qualifying_features.csv)'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip dataset validation step'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    parser.add_argument(
        '--show-summary',
        action='store_true',
        help='Show detailed dataset summary after generation'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    """
    Command-line interface for generating ML features.
    
    Usage:
        python helpers/feature_engineering.py [OPTIONS]
    
    Output:
        - Generates qualifying_features.csv in data/processed/
        - Prints dataset summary statistics
        - Shows year-by-year breakdown
    
    Examples:
        # Default: generate features for 2022-2024
        $ python helpers/feature_engineering.py
        
        # Custom years
        $ python helpers/feature_engineering.py --years 2023 2024
        
        # Custom output path
        $ python helpers/feature_engineering.py --output custom_features.csv
        
        # Verbose mode
        $ python helpers/feature_engineering.py --verbose
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configure logging based on arguments
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    try:
        # Generate features
        df = prepare_qualifying_dataset(args.years)
        
        # Show summary statistics
        if not args.quiet:
            logger.info("\n Dataset Summary:")
            logger.info("   Shape: %s", df.shape)
            logger.info("\n   Rows per year:")
            for year, count in df.groupby('year').size().items():
                logger.info("      %d: %d", year, count)
            
            if args.show_summary:
                logger.info("\n   Column types:")
                logger.info("\n%s", df.dtypes.value_counts())
                logger.info("\n   Missing values per column:")
                missing = df.isnull().sum()
                if missing.sum() == 0:
                    logger.info("      None! ✅")
                else:
                    for col, count in missing[missing > 0].items():
                        logger.info("      %s: %d", col, count)
        
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        if not args.quiet:
            logger.info("\n Saved to: %s", output_path.absolute())
            logger.info("   File size: %.2f MB", output_path.stat().st_size / 1_000_000)
        
        logger.info("\n✅ Feature engineering complete!")
        
    except Exception as e:
        logger.error("\n❌ Feature engineering failed: %s", str(e), exc_info=args.verbose)
        raise