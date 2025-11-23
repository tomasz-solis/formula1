"""
F1 Session Results Export Utilities

Provides functions to export session results (qualifying, race, sprint) from
FastF1 API to CSV files for ML model training. Implements append-only strategy
to avoid duplicate data and supports incremental updates.

Key Features:
- Append-only CSV exports (no overwrites)
- Duplicate detection and prevention
- Sprint weekend handling
- Multi-season batch exports
- Team name canonicalization
- Comprehensive error tracking

Example:
    >>> from helpers.prediction import export_completed_classifications_csv_range
    >>> results = export_completed_classifications_csv_range(2022, 2024)
    >>> print(results[2024]['Qualifying'].status)
    'appended'

Author: Tomasz Solis
Date: November 2025
"""

import os
import pandas as pd
import fastf1 as ff1
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Optional
from tqdm import tqdm


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExportResult:
    """
    Result of a classification export operation.
    
    Attributes:
        status: Operation outcome ('created', 'appended', 'skipped', 'error')
        written_path: File path if successful (None if skipped/error)
        reason: Explanation for skip/error (None if successful)
    """
    status: str
    written_path: Optional[str] = None
    reason: Optional[str] = None


# =============================================================================
# MULTI-SEASON EXPORT
# =============================================================================

def export_completed_classifications_csv_range(
    start_year: int,
    end_year: int,
    include_sprint: bool = True,
    up_to_utc: Optional[datetime] = None
) -> Dict[int, Dict[str, ExportResult]]:
    """
    Export classification CSVs for multiple seasons (append-only).
    
    For each season and session type (Qualifying, Race, Sprint, Sprint Qualifying),
    checks existing CSV files and appends only new events not already present.
    Prevents duplicate data while enabling incremental updates.
    
    Args:
        start_year: First season to export (inclusive)
        end_year: Last season to export (inclusive)
        include_sprint: Whether to include sprint sessions (default: True)
        up_to_utc: Optional cutoff time - only export sessions before this
        
    Returns:
        Nested dictionary: {year: {session_type: ExportResult}}
        
    Example:
        >>> results = export_completed_classifications_csv_range(2022, 2024)
        >>> print(results[2024]['Qualifying'].status)
        'appended'
        >>> print(results[2024]['Qualifying'].written_path)
        'data/predictions/ssot/2024_qualifying.csv'
    """
    results_by_season = {}
    
    # Iterate through years
    for year in tqdm(
        range(start_year, end_year + 1),
        desc="📦 Exporting seasons",
        colour="blue",
        leave=True
    ):
        session_types = ['Qualifying', 'Race']
        if include_sprint:
            session_types.extend(['Sprint', 'Sprint Qualifying'])
        
        year_results = {}
        
        # Iterate through session types for this year
        for session_type in tqdm(
            session_types,
            desc=f"  {year}",
            colour="cyan",
            leave=False,
            position=1
        ):
            try:
                result = export_session_classification(
                    year=year,
                    session_type=session_type,
                    up_to_utc=up_to_utc
                )
                year_results[session_type] = result
            except Exception as e:
                year_results[session_type] = ExportResult(
                    status='error',
                    reason=str(e),
                    written_path=None
                )
        
        results_by_season[year] = year_results
    
    return results_by_season


# =============================================================================
# SINGLE-SESSION EXPORT
# =============================================================================

def export_session_classification(
    year: int,
    session_type: str,
    up_to_utc: Optional[datetime] = None
) -> ExportResult:
    """
    Export one session type for one year (append-only).
    
    Workflow:
    1. Check if CSV exists
    2. Load existing data (if present)
    3. Find new events to add
    4. Append only missing rows
    5. Save updated CSV
    
    Args:
        year: Season year
        session_type: 'Qualifying', 'Race', 'Sprint', or 'Sprint Qualifying'
        up_to_utc: Optional cutoff time (timezone-aware datetime)
        
    Returns:
        ExportResult with status and file path
        
    Example:
        >>> result = export_session_classification(2024, 'Qualifying')
        >>> print(result.status)
        'appended'
    """
    # Map session type to filename
    file_mapping = {
        'Qualifying': f'{year}_qualifying.csv',
        'Race': f'{year}_race.csv',
        'Sprint': f'{year}_sprint.csv',
        'Sprint Qualifying': f'{year}_sprint_qualifying.csv'
    }
    
    if session_type not in file_mapping:
        return ExportResult(
            status='error',
            reason=f"Unknown session type: {session_type}"
        )
    
    output_dir = 'data/predictions/ssot'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, file_mapping[session_type])
    
    # Get event schedule for this year
    try:
        schedule = ff1.get_event_schedule(year)
        
        # Ensure timezone-aware datetime
        if 'Session1DateUtc' in schedule.columns:
            schedule['Session1DateUtc'] = pd.to_datetime(
                schedule['Session1DateUtc'],
                utc=True
            )
    except Exception as e:
        return ExportResult(
            status='error',
            reason=f"Failed to load schedule: {e}"
        )
    
    # Filter to completed events only
    now = datetime.now(timezone.utc)
    
    if up_to_utc:
        # Ensure cutoff time is timezone-aware
        if up_to_utc.tzinfo is None:
            up_to_utc = up_to_utc.replace(tzinfo=timezone.utc)
        schedule = schedule[schedule['Session1DateUtc'] < up_to_utc]
    else:
        schedule = schedule[schedule['Session1DateUtc'] < now]
    
    if schedule.empty:
        if os.path.exists(output_file):
            return ExportResult(status='skipped', written_path=output_file)
        else:
            return ExportResult(status='skipped', reason='No completed events')
    
    # Map session type to FastF1 session name
    session_map = {
        'Qualifying': 'Qualifying',
        'Race': 'Race',
        'Sprint': 'Sprint',
        'Sprint Qualifying': 'Sprint Qualifying'
    }
    ff1_session = session_map.get(session_type)
    
    if not ff1_session:
        return ExportResult(
            status='error',
            reason=f"No FastF1 mapping for {session_type}"
        )
    
    # Collect new data from FastF1 API
    new_data = []
    errors = []
    
    for _, event in schedule.iterrows():
        event_name = event['EventName']
        
        # Skip testing events
        if 'test' in event_name.lower():
            continue
        
        # Check if session exists for this event format
        event_format = str(event.get('EventFormat', 'conventional')).lower()
        
        # Sprint Qualifying only in sprint_qualifying format
        if session_type == 'Sprint Qualifying' and event_format != 'sprint_qualifying':
            continue
        
        # Sprint only in sprint formats
        if session_type == 'Sprint' and 'sprint' not in event_format:
            continue
        
        try:
            # Load session from FastF1
            session = ff1.get_session(year, event_name, ff1_session)
            session.load(laps=False, telemetry=False, weather=False)
            
            results = session.results
            
            if results is None or results.empty:
                continue
            
            # Extract relevant columns
            if session_type in ['Qualifying', 'Sprint Qualifying']:
                if 'Position' not in results.columns:
                    continue
                
                event_data = results[[
                    'DriverNumber', 'Abbreviation', 'TeamName', 'Position'
                ]].copy()
                event_data.columns = [
                    'driver_number', 'driver', 'team', 'qualifying_position'
                ]
                
            elif session_type in ['Race', 'Sprint']:
                if 'Position' not in results.columns:
                    continue
                
                event_data = results[[
                    'DriverNumber', 'Abbreviation', 'TeamName', 'Position'
                ]].copy()
                event_data.columns = [
                    'driver_number', 'driver', 'team', 'race_position'
                ]
            
            # Add metadata
            event_data['year'] = year
            event_data['event'] = event_name
            event_data['session'] = session_type
            
            new_data.append(event_data)
            
        except Exception as e:
            errors.append(f"{event_name}: {str(e)}")
            continue
    
    # Handle case with no new data
    if not new_data:
        if os.path.exists(output_file):
            return ExportResult(status='skipped', written_path=output_file)
        else:
            error_msg = (
                f"No data extracted. Sample errors: {errors[:3]}" 
                if errors else "No data available"
            )
            return ExportResult(status='error', reason=error_msg)
    
    new_df = pd.concat(new_data, ignore_index=True)
    
    # Merge with existing data (if file exists)
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        
        # Canonicalize team names in both DataFrames
        from helpers.team_name_mapping import normalize_team_column
        existing_df = normalize_team_column(existing_df, col='team')
        new_df = normalize_team_column(new_df, col='team')
        
        # Find new entries (not in existing data)
        existing_keys = set(
            existing_df[['year', 'event', 'driver']].itertuples(
                index=False, name=None
            )
        )
        new_keys = set(
            new_df[['year', 'event', 'driver']].itertuples(
                index=False, name=None
            )
        )
        
        keys_to_add = new_keys - existing_keys
        
        if not keys_to_add:
            return ExportResult(status='skipped', written_path=output_file)
        
        # Filter to only new rows
        new_rows_mask = new_df[['year', 'event', 'driver']].apply(
            tuple, axis=1
        ).isin(keys_to_add)
        rows_to_add = new_df[new_rows_mask]
        
        # Append and save
        updated_df = pd.concat([existing_df, rows_to_add], ignore_index=True)
        updated_df.to_csv(output_file, index=False)
        
        return ExportResult(status='appended', written_path=output_file)
    
    else:
        # Create new file
        new_df.to_csv(output_file, index=False)
        return ExportResult(status='created', written_path=output_file)