"""
Feature engineering for F1 qualifying position prediction.
Combines driver telemetry, circuit data, and results into single feature matrix.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


def load_driver_profiles(years: List[int]) -> pd.DataFrame:
    """
    Load and concatenate driver profiles across multiple years.
    
    Returns:
        Combined DataFrame with all driver session data (FP1, FP2, FP3, Q, R, SQ, S)
    """
    dfs = []
    for year in years:
        path = f'data/driver/{year}_driver_profiles.csv'
        if Path(path).exists():
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"✅ Loaded {len(df)} rows from {year}")
        else:
            print(f"⚠️  Missing: {path}")
    
    if not dfs:
        raise FileNotFoundError("No driver profiles found!")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Total driver sessions: {len(combined)}")
    print(f"   Sessions: {combined['session'].unique().tolist()}")
    return combined


def load_circuit_profiles(years: List[int]) -> pd.DataFrame:
    """
    Load and concatenate circuit profiles across multiple years.
    
    Returns:
        Combined DataFrame with track characteristics per session
    """
    dfs = []
    for year in years:
        path = f'data/circuit/{year}_circuit_profiles.csv'
        if Path(path).exists():
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"✅ Loaded {len(df)} rows from {year}")
        else:
            print(f"⚠️  Missing: {path}")
    
    if not dfs:
        raise FileNotFoundError("No circuit profiles found!")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Total circuit sessions: {len(combined)}")
    return combined


def load_qualifying_results(years: List[int]) -> pd.DataFrame:
    """
    Load qualifying results (target variable = Position column).
    
    Returns:
        DataFrame with EventName, Abbreviation, Position (1-20), TeamName, etc.
    """
    dfs = []
    for year in years:
        path = f'data/predictions/ssot/{year}_qualifying.csv'
        if Path(path).exists():
            df = pd.read_csv(path)
            df['year'] = year
            
            # Use 'Position' as our target (not ClassifiedPosition)
            if 'Position' in df.columns:
                df['qualifying_position'] = df['Position']
            else:
                raise ValueError(f"No 'Position' column in {path}")
            
            dfs.append(df)
            print(f"✅ Loaded {len(df)} results from {year}")
        else:
            print(f"⚠️  Missing: {path}")
    
    if not dfs:
        raise FileNotFoundError("No qualifying results found!")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Total qualifying results: {len(combined)}")
    print(f"   Position range: {combined['qualifying_position'].min():.0f} to {combined['qualifying_position'].max():.0f}")
    return combined


def prepare_qualifying_dataset(
    years: List[int] = [2022, 2023, 2024]
) -> pd.DataFrame:
    """
    Build complete feature matrix for qualifying prediction.
    
    Pipeline:
    1. Load driver profiles (all sessions)
    2. Load circuit profiles (all sessions)
    3. Load qualifying results (target variable)
    4. Filter to ONLY practice sessions (FP1, FP2, FP3, SQ)
    5. Merge driver + circuit data
    6. Aggregate FP sessions per driver-race
    7. Merge with qualifying results (add target)
    
    Args:
        years: List of years to include
        
    Returns:
        DataFrame ready for modeling (features + target)
    """
    print("=" * 60)
    print("🏗️  Building Qualifying Prediction Dataset")
    print("=" * 60)
    
    # Step 1: Load raw data
    print("\n📂 Loading driver profiles...")
    drivers = load_driver_profiles(years)
    
    print("\n📂 Loading circuit profiles...")
    circuits = load_circuit_profiles(years)
    
    print("\n📂 Loading qualifying results...")
    results = load_qualifying_results(years)
    
    # Step 2: Filter to practice sessions only
    print("\n🔍 Filtering to practice sessions...")
    practice_sessions = ['FP1', 'FP2', 'FP3', 'SQ']
    drivers_practice = drivers[drivers['session'].isin(practice_sessions)].copy()
    circuits_practice = circuits[circuits['session'].isin(practice_sessions)].copy()
    
    print(f"   Driver practice data: {len(drivers_practice)} rows")
    print(f"   Circuit practice data: {len(circuits_practice)} rows")
    
    # Step 3: Merge driver telemetry with circuit characteristics
    combined = merge_driver_circuit_data(drivers_practice, circuits_practice)
    
    # Step 4: Aggregate FP sessions per driver-race
    combined = aggregate_practice_sessions(combined)
    
    # Step 5: Add target variable (qualifying position)
    final = merge_with_qualifying_results(combined, results)

    # Step 6: Fix missing circuit data
    final = fix_missing_circuit_data(final)
    
    print("\n" + "=" * 60)
    print("✅ DATASET COMPLETE!")
    print("=" * 60)
    
    return final


def fix_missing_circuit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix missing circuit features by imputing from other years.
    
    Strategy:
    - Drop real_altitude (low variance, not useful)
    - For missing circuit features: use same track from different year
    """
    print("\n🔧 Fixing missing circuit data...")
    
    # 1. Drop altitude (not useful anyway)
    if 'real_altitude' in df.columns:
        df = df.drop(columns=['real_altitude'])
        print("   ✅ Dropped real_altitude feature")
    
    # 2. Identify rows with missing circuit data
    circuit_cols = ['slow_corners', 'fast_corners', 'avg_speed_circuit', 
                    'slow_corner_pct', 'medium_corner_pct', 'fast_corner_pct']
    
    missing_mask = df[circuit_cols[0]].isna()
    print(f"   Rows with missing circuit data: {missing_mask.sum()}")
    
    if missing_mask.sum() == 0:
        print("   ✅ No missing circuit data!")
        return df
    
    # 3. For each missing row, find same track from different year
    for idx in df[missing_mask].index:
        event = df.loc[idx, 'event']
        year = df.loc[idx, 'year']
        
        # Find same event from other years
        same_track_other_years = df[
            (df['event'] == event) & 
            (df['year'] != year) & 
            (df[circuit_cols[0]].notna())
        ]
        
        if len(same_track_other_years) > 0:
            # Use circuit features from most recent available year
            donor_row = same_track_other_years.iloc[0]
            
            for col in circuit_cols:
                df.loc[idx, col] = donor_row[col]
            
            # Also update related columns
            for col in ['total_corners', 'chicanes', 'top_speed_circuit']:
                if col in df.columns and col in donor_row.index:
                    df.loc[idx, col] = donor_row[col]
    
    # 4. Verify fix
    remaining_missing = df[circuit_cols[0]].isna().sum()
    print(f"   ✅ Imputed circuit features")
    print(f"   Remaining missing: {remaining_missing}")
    
    return df


def merge_driver_circuit_data(
    drivers: pd.DataFrame,
    circuits: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge driver practice session data with circuit characteristics.
    
    Merge keys: year + event + session
    
    Args:
        drivers: Driver telemetry from practice sessions
        circuits: Circuit characteristics from practice sessions
        
    Returns:
        Combined DataFrame with both driver and track features
        
    Example:
        Before merge:
        - drivers: 1287 rows (one per driver-session)
        - circuits: 66 rows (one per circuit-session)
        
        After merge:
        - combined: 1287 rows (drivers enriched with track data)
    """
    print("\n🔗 Merging driver telemetry with circuit data...")
    
    # Drop duplicate columns from circuits before merge
    circuits_clean = circuits.drop(columns=['location'], errors='ignore')
    
    # Merge on: year + event + session
    merged = drivers.merge(
        circuits_clean,
        on=['year', 'event', 'session'],
        how='left',
        suffixes=('_driver', '_circuit')
    )
    
    print(f"   ✅ Merged shape: {merged.shape}")
    print(f"   Missing circuit data: {merged['avg_speed'].isna().sum()} rows")
    
    return merged


def aggregate_practice_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate practice sessions into single row per driver-race.
    
    Strategy (revised based on F1 reality):
    1. Use BEST performance across all available sessions (not just FP3)
    2. For sprint weekends: use Sprint Qualifying as a feature
    3. Calculate average to capture consistency
    4. Flag which data is available (important for model to learn)
    
    Args:
        df: Merged driver+circuit data (one row per driver-session)
        
    Returns:
        Aggregated data (one row per driver-race)
    """
    print("\n📊 Aggregating practice sessions per driver-race...")
    
    # Show session patterns
    session_counts = df.groupby(['year', 'event'])['session'].unique()
    print(f"\n   Session patterns in dataset:")
    normal_count = 0
    sprint_count = 0
    for (year, event), sessions in session_counts.items():
        sessions_sorted = sorted(sessions)
        if 'SQ' in sessions:
            sprint_count += 1
            if sprint_count <= 2:  # Show first 2 examples
                print(f"      Sprint: {event} → {sessions_sorted}")
        else:
            normal_count += 1
            if normal_count <= 2:  # Show first 2 examples
                print(f"      Normal: {event} → {sessions_sorted}")
    
    print(f"   Total: {normal_count} normal weekends, {sprint_count} sprint weekends")
    
    # Group by driver + race
    grouped = df.groupby(['year', 'event', 'driver'], dropna=False)
    print(f"   Unique driver-race combinations: {len(grouped)}")
    
    aggregated_rows = []
    
    for (year, event, driver), group in grouped:
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
        
        # ==============================================================
        # STRATEGY 1: Best performance across ALL available sessions
        # (Teams always show their true pace at least once)
        # ==============================================================
        row['best_throttle_ratio'] = group['max_throttle_ratio'].max()
        row['best_brake_max_g'] = group['brake_max_g'].max()
        row['best_drs_activations'] = group['drs_activations'].max()
        
        # ==============================================================
        # STRATEGY 2: Average performance (consistency indicator)
        # ==============================================================
        row['avg_throttle_ratio'] = group['max_throttle_ratio'].mean()
        row['avg_brake_max_g'] = group['brake_max_g'].mean()
        row['avg_degradation_slope'] = group['degradation_slope'].mean()
        
        # ==============================================================
        # STRATEGY 3: Session-specific features
        # ==============================================================
        
        if is_sprint:
            # Sprint weekend: Use FP1 + Sprint Qualifying
            fp1 = group[group['session'] == 'FP1']
            sq = group[group['session'] == 'SQ']
            
            if not fp1.empty:
                row['fp1_throttle_ratio'] = fp1['max_throttle_ratio'].iloc[0]
                row['fp1_track_temp'] = fp1['avg_track_temp'].iloc[0]
            else:
                row['fp1_throttle_ratio'] = np.nan
                row['fp1_track_temp'] = np.nan
            
            # Sprint Qualifying is VERY predictive for sprint weekends
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
            
        else:
            # Normal weekend: Use FP3 as most recent
            fp3 = group[group['session'] == 'FP3']
            
            if not fp3.empty:
                row['fp3_throttle_ratio'] = fp3['max_throttle_ratio'].iloc[0]
                row['fp3_brake_max_g'] = fp3['brake_max_g'].iloc[0]
                row['fp3_track_temp'] = fp3['avg_track_temp'].iloc[0]
            else:
                # Rare case: FP3 cancelled (weather, etc.)
                row['fp3_throttle_ratio'] = np.nan
                row['fp3_brake_max_g'] = np.nan
                row['fp3_track_temp'] = np.nan
            
            # No sprint qualifying on normal weekends
            row['sprint_quali_throttle'] = np.nan
            row['sprint_quali_brake_g'] = np.nan
            row['has_sprint_quali_data'] = False
            
            # Also capture FP1 for comparison
            fp1 = group[group['session'] == 'FP1']
            if not fp1.empty:
                row['fp1_throttle_ratio'] = fp1['max_throttle_ratio'].iloc[0]
                row['fp1_track_temp'] = fp1['avg_track_temp'].iloc[0]
            else:
                row['fp1_throttle_ratio'] = np.nan
                row['fp1_track_temp'] = np.nan
        
        # ==============================================================
        # Weather features
        # ==============================================================
        row['rain_in_practice'] = (group['avg_rainfall'].max() > 0)
        row['avg_track_temp'] = group['avg_track_temp'].mean()
        row['track_temp_std'] = group['avg_track_temp'].std()  # temp variation
        
        # ==============================================================
        # Circuit features (static per race)
        # ==============================================================
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
        
        aggregated_rows.append(row)
    
    result = pd.DataFrame(aggregated_rows)
    
    print(f"\n   ✅ Aggregated to {len(result)} driver-race combinations")
    print(f"      Sprint weekends: {result['is_sprint_weekend'].sum()}")
    print(f"      Normal weekends: {(~result['is_sprint_weekend']).sum()}")
    print(f"      Sprint Quali data available: {result['has_sprint_quali_data'].sum()}")
    
    return result


def merge_with_qualifying_results(
    features: pd.DataFrame,
    results: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge practice features with qualifying results (target variable).
    
    Merge keys: year + event + driver abbreviation
    
    Args:
        features: Aggregated practice session features (502 rows)
        results: Qualifying results with Position column (479 rows)
        
    Returns:
        Complete dataset with features + target (qualifying_position)
        
    Note:
        Some drivers may have practice data but no qualifying result (DNS/DSQ)
        We'll drop these rows since we can't train without a target.
    """
    print("\n🎯 Merging with qualifying results (target variable)...")
    
    # The driver column in features is 3-letter code (VER, HAM, etc.)
    # The results have 'Abbreviation' column
    # Also need to match on year + event
    
    print(f"   Features shape: {features.shape}")
    print(f"   Results shape: {results.shape}")
    
    # Rename for merge
    results_clean = results[['year', 'EventName', 'Abbreviation', 'qualifying_position', 'TeamName']].copy()
    results_clean = results_clean.rename(columns={
        'EventName': 'event',
        'Abbreviation': 'driver'
    })
    
    # Merge
    merged = features.merge(
        results_clean,
        on=['year', 'event', 'driver'],
        how='inner',  # Only keep rows with both features AND target
        suffixes=('', '_results')
    )
    
    print(f"   ✅ Merged shape: {merged.shape}")
    print(f"   Dropped rows (no quali result): {len(features) - len(merged)}")
    
    # Validate target variable
    print(f"\n   Target variable (qualifying_position):")
    print(f"      Min: {merged['qualifying_position'].min():.0f}")
    print(f"      Max: {merged['qualifying_position'].max():.0f}")
    print(f"      Missing: {merged['qualifying_position'].isna().sum()}")
    
    return merged


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical performance features for each driver-track combination.
    
    Features added:
    1. driver_track_avg_quali_3yr: Driver's avg quali position at this track (last 3 years)
    2. team_track_avg_quali_3yr: Team's avg quali position at this track (last 3 years)
    3. driver_recent_form: Driver's avg quali position in last 3 races
    
    Args:
        df: Dataset with year, event, driver, qualifying_position
        
    Returns:
        Dataset with 3 new historical features
        
    Implementation notes:
    - Must sort by date to avoid data leakage!
    - For 2022 data: no historical features (NaN or impute with season avg)
    - For 2023+: look back to previous years
    """

    pass # TBD

if __name__ == "__main__":
    years_used = [2022, 2023, 2024]
    df = prepare_qualifying_dataset(years_used)

    print(f"\n📊 Dataset by year:")
    print(df.groupby('year').size())
    
    # Create directory if it doesn't exist
    from pathlib import Path
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV for next steps
    output_path = output_dir / 'qualifying_features.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")