"""
Classification export utilities for F1 ML pipeline.

Provides functions to export session results (qualifying positions, race results)
to CSV files for downstream ML model training and evaluation.

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


@dataclass
class ExportResult:
    """Result of a classification export operation."""
    status: str  # 'created', 'appended', 'skipped', 'error'
    written_path: Optional[str] = None
    reason: Optional[str] = None


def export_completed_classifications_csv_range(
    start_year: int,
    end_year: int,
    include_sprint: bool = True,
    up_to_utc: Optional[datetime] = None
) -> Dict[int, Dict[str, ExportResult]]:
    """
    Export classification CSVs for multiple seasons (append-only).
    
    For each season and session type (Qualifying, Race, Sprint, Sprint Qualifying),
    checks existing CSV files and appends only new events that aren't already present.
    
    Args:
        start_year: First season to export (inclusive)
        end_year: Last season to export (inclusive)
        include_sprint: Whether to include sprint sessions (default: True)
        up_to_utc: Optional cutoff time (only export sessions before this)
    
    Returns:
        Nested dict: {year: {session_type: ExportResult}}
        
    Example:
        >>> results = export_completed_classifications_csv_range(2022, 2024)
        >>> print(results[2024]['Qualifying'].status)
        'appended'
        >>> print(results[2024]['Qualifying'].written_path)
        'data/predictions/ssot/2024_qualifying.csv'
    """
    results_by_season = {}
    
    # Level 1: Iterate through years
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
        
        # Level 2: Iterate through session types for this year
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


def export_session_classification(
    year: int,
    session_type: str,
    up_to_utc: Optional[datetime] = None
) -> ExportResult:
    """
    Export one session type for one year (append-only).
    
    Checks if CSV exists, loads existing data, finds new events to add,
    and appends only missing rows.
    
    Args:
        year: Season year
        session_type: 'Qualifying', 'Race', 'Sprint', or 'Sprint Qualifying'
        up_to_utc: Optional cutoff time
    
    Returns:
        ExportResult with status and path
    """
    # Map session type to file name
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
    
    # Get all events for this year and session type
    try:
        schedule = ff1.get_event_schedule(year)
        
        # CRITICAL FIX: Ensure Session1DateUtc is timezone-aware
        if 'Session1DateUtc' in schedule.columns:
            schedule['Session1DateUtc'] = pd.to_datetime(schedule['Session1DateUtc'], utc=True)
        
    except Exception as e:
        return ExportResult(
            status='error',
            reason=f"Failed to load schedule: {e}"
        )
    
    # Filter to only completed events
    now = datetime.now(timezone.utc)
    
    # Filter by cutoff time if provided
    if up_to_utc:
        # Ensure up_to_utc is timezone-aware
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
    
    # Collect new data
    new_data = []
    errors = []
    
    for _, event in schedule.iterrows():
        event_name = event['EventName']
        
        # Skip testing events
        if 'test' in event_name.lower():
            continue
        
        # Check if this session type exists for this event format
        event_format = str(event.get('EventFormat', 'conventional')).lower()
        
        # Sprint Qualifying only exists in sprint_qualifying format
        if session_type == 'Sprint Qualifying' and event_format != 'sprint_qualifying':
            continue
        
        # Sprint exists in all sprint formats
        if session_type == 'Sprint' and 'sprint' not in event_format:
            continue
        
        try:
            # Load session
            session = ff1.get_session(year, event_name, ff1_session)
            session.load(laps=False, telemetry=False, weather=False)
            
            # Get results
            results = session.results
            
            if results is None or results.empty:
                continue
            
            # Extract relevant columns
            if session_type in ['Qualifying', 'Sprint Qualifying']:
                # Check if required columns exist
                if 'Position' not in results.columns:
                    continue
                
                event_data = results[['DriverNumber', 'Abbreviation', 'TeamName', 'Position']].copy()
                event_data.columns = ['driver_number', 'driver', 'team', 'qualifying_position']
                
            elif session_type in ['Race', 'Sprint']:
                # Check if required columns exist
                if 'Position' not in results.columns:
                    continue
                
                event_data = results[['DriverNumber', 'Abbreviation', 'TeamName', 'Position']].copy()
                event_data.columns = ['driver_number', 'driver', 'team', 'race_position']
            
            # Add metadata
            event_data['year'] = year
            event_data['event'] = event_name
            event_data['session'] = session_type
            
            new_data.append(event_data)
            
        except Exception as e:
            # Track errors for debugging
            errors.append(f"{event_name}: {str(e)}")
            continue
    
    # If no new data collected
    if not new_data:
        if os.path.exists(output_file):
            return ExportResult(status='skipped', written_path=output_file)
        else:
            # Show first few errors if we have them
            error_msg = f"No data extracted. Sample errors: {errors[:3]}" if errors else "No data available"
            return ExportResult(status='error', reason=error_msg)
    
    new_df = pd.concat(new_data, ignore_index=True)
    
    # Check if file exists
    if os.path.exists(output_file):
        # Load existing data
        existing_df = pd.read_csv(output_file)
        
        # Find what's new (based on year + event + driver)
        existing_keys = set(
            existing_df[['year', 'event', 'driver']].itertuples(index=False, name=None)
        )
        new_keys = set(
            new_df[['year', 'event', 'driver']].itertuples(index=False, name=None)
        )
        
        keys_to_add = new_keys - existing_keys
        
        if not keys_to_add:
            # Nothing new to add
            return ExportResult(status='skipped', written_path=output_file)
        
        # Filter to only new rows
        mask = new_df[['year', 'event', 'driver']].apply(tuple, axis=1).isin(keys_to_add)
        rows_to_add = new_df[mask]
        
        # Append and save
        updated_df = pd.concat([existing_df, rows_to_add], ignore_index=True)
        updated_df.to_csv(output_file, index=False)
        
        return ExportResult(status='appended', written_path=output_file)
    
    else:
        # Create new file
        new_df.to_csv(output_file, index=False)
        return ExportResult(status='created', written_path=output_file)