"""
Data Validation Utilities for F1 ML Pipeline

Provides comprehensive validation functions for DataFrame quality assurance
at critical pipeline stages. Validates data integrity, completeness, and
correctness before model training or evaluation.

Key Features:
- Feature DataFrame validation (shape, columns, ranges)
- Merge key compatibility checking
- Session data validation
- Dataset summary statistics
- Comprehensive validation reports

Example:
    >>> from helpers.validation import validate_feature_dataframe
    >>> validate_feature_dataframe(df, REQUIRED_FEATURES, 'qualifying_position')
    ✅ Validation passed: 1353 rows, 36 features, 0 missing values

Author: Tomasz Solis
Date: November 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# CORE VALIDATION FUNCTIONS

def validate_feature_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    target_column: str = 'qualifying_position',
    min_rows: int = 100
) -> None:
    """
    Validate ML feature DataFrame meets quality standards.
    
    Performs comprehensive checks including shape validation, required columns,
    target variable integrity, missing values, duplicates, and feature ranges.
    
    Args:
        df: Feature DataFrame to validate
        required_columns: List of columns that must exist
        target_column: Name of target variable column
        min_rows: Minimum acceptable number of rows
        
    Raises:
        ValueError: If any validation check fails, with detailed error messages
        
    Example:
        >>> features = ['max_throttle_ratio', 'brake_max_g', 'avg_speed']
        >>> validate_feature_dataframe(df, features, 'qualifying_position')
        ✅ Validation passed: 1353 rows, 36 features, 0 missing values
    """
    errors = []
    warnings = []
    
    # Check 1: DataFrame shape
    if len(df) == 0:
        errors.append("DataFrame is empty (0 rows)")
    elif len(df) < min_rows:
        warnings.append(
            f"Only {len(df)} rows (minimum recommended: {min_rows})"
        )
    
    # Check 2: Required columns present
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check 3: Target variable integrity
    if target_column in df.columns:
        # Missing values in target
        missing_target = df[target_column].isna().sum()
        if missing_target > 0:
            pct_missing = 100 * missing_target / len(df)
            errors.append(
                f"Target '{target_column}' has {missing_target} missing values "
                f"({pct_missing:.1f}%)"
            )
        
        # Invalid values (F1 positions are 1-20)
        valid_range = (1, 20)
        invalid_mask = ~df[target_column].between(*valid_range)
        if invalid_mask.any():
            bad_values = df.loc[invalid_mask, target_column].unique()
            errors.append(
                f"Target '{target_column}' has invalid values: {bad_values}. "
                f"Must be between {valid_range[0]}-{valid_range[1]}"
            )
    else:
        errors.append(f"Target column '{target_column}' not found")
    
    # Check 4: Duplicate detection
    if all(col in df.columns for col in ['year', 'event', 'driver']):
        duplicate_count = df.duplicated(subset=['year', 'event', 'driver']).sum()
        if duplicate_count > 0:
            pct_dupes = 100 * duplicate_count / len(df)
            errors.append(
                f"Found {duplicate_count} duplicate driver-race combinations "
                f"({pct_dupes:.1f}%)"
            )
    
    # Check 5: Excessive missing values per column
    missing_per_col = df[required_columns].isnull().sum()
    high_missing = missing_per_col[missing_per_col > len(df) * 0.3]
    if not high_missing.empty:
        warnings.append(
            f"Columns with >30% missing: {high_missing.to_dict()}"
        )
    
    # Check 6: Feature value ranges
    if 'max_throttle_ratio' in df.columns:
        invalid_throttle = ~df['max_throttle_ratio'].between(0, 1)
        if invalid_throttle.any():
            errors.append(
                f"Throttle ratio outside [0,1]: "
                f"{df.loc[invalid_throttle, 'max_throttle_ratio'].describe()}"
            )
    
    if 'slow_corner_pct' in df.columns:
        invalid_pct = ~df['slow_corner_pct'].between(0, 1)
        if invalid_pct.any() and df['slow_corner_pct'].notna().any():
            errors.append(
                f"Corner percentage outside [0,1]: "
                f"{df.loc[invalid_pct, 'slow_corner_pct'].describe()}"
            )
    
    # Report results
    if errors:
        error_msg = "❌ Validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if warnings:
        for warning in warnings:
            logger.warning("⚠️  %s", warning)
    
    total_missing = df[required_columns].isnull().sum().sum()
    logger.info(
        "✅ Validation passed: %d rows, %d features, %d missing values",
        len(df), len(required_columns), total_missing
    )

def validate_merge_keys(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    merge_keys: List[str],
    context: str = "merge"
) -> None:
    """
    Validate merge keys exist and have compatible data types.
    
    Ensures safe DataFrame merging by checking key presence and type
    compatibility. Warns on type mismatches but doesn't fail.
    
    Args:
        df1: First DataFrame to merge
        df2: Second DataFrame to merge
        merge_keys: List of column names to merge on
        context: Description for error messages (e.g., "driver-circuit merge")
        
    Raises:
        ValueError: If required merge keys are missing from either DataFrame
        
    Example:
        >>> validate_merge_keys(drivers, circuits, ['year', 'event', 'session'])
        ✅ Merge keys validated for driver-circuit merge
    """
    errors = []
    
    # Check keys exist in both DataFrames
    missing_df1 = [k for k in merge_keys if k not in df1.columns]
    missing_df2 = [k for k in merge_keys if k not in df2.columns]
    
    if missing_df1:
        errors.append(f"DataFrame 1 missing keys: {missing_df1}")
    if missing_df2:
        errors.append(f"DataFrame 2 missing keys: {missing_df2}")
    
    if errors:
        error_msg = f"❌ {context} validation failed:\n" + "\n".join(errors)
        raise ValueError(error_msg)
    
    # Check data type compatibility (warn but don't fail)
    for key in merge_keys:
        dtype1 = df1[key].dtype
        dtype2 = df2[key].dtype
        
        if dtype1 != dtype2:
            logger.warning(
                "⚠️  Key '%s' has different types: %s vs %s",
                key, dtype1, dtype2
            )
    
    logger.info("✅ Merge keys validated for %s", context)

def validate_session_data(
    df: pd.DataFrame,
    expected_sessions: List[str]
) -> None:
    """
    Validate session data contains only expected session types.
    
    Ensures session codes are valid F1 session identifiers and flags
    any unexpected session types for investigation.
    
    Args:
        df: DataFrame with 'session' column
        expected_sessions: List of valid session codes (e.g., ['FP1', 'FP2', 'Q'])
        
    Raises:
        ValueError: If 'session' column missing or invalid sessions found
        
    Example:
        >>> validate_session_data(drivers, ['FP1', 'FP2', 'FP3', 'Q', 'R'])
        ✅ All sessions valid: ['FP1', 'FP2', 'FP3', 'Q', 'R']
    """
    if 'session' not in df.columns:
        raise ValueError("DataFrame missing 'session' column")
    
    actual_sessions = df['session'].unique().tolist()
    invalid_sessions = [s for s in actual_sessions if s not in expected_sessions]
    
    if invalid_sessions:
        raise ValueError(
            f"❌ Invalid session codes found: {invalid_sessions}\n"
            f"Expected: {expected_sessions}"
        )
    
    logger.info("✅ All sessions valid: %s", actual_sessions)

# SUMMARY STATISTICS

def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics for dataset.
    
    Computes key metrics including row counts, missing values, year breakdown,
    weekend types, and target variable statistics.
    
    Args:
        df: Feature DataFrame
        
    Returns:
        Dictionary containing:
        - total_rows, total_columns
        - missing_values, missing_pct
        - rows_per_year (if 'year' column exists)
        - sprint_weekends, normal_weekends (if 'is_sprint_weekend' exists)
        - target stats (if 'qualifying_position' exists)
        
    Example:
        >>> summary = summarize_dataset(df)
        >>> print(f"Rows: {summary['total_rows']}, Missing: {summary['missing_pct']:.1f}%")
        Rows: 1353, Missing: 2.3%
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'missing_pct': 100 * df.isnull().sum().sum() / (len(df) * len(df.columns)),
        'duplicate_rows': df.duplicated().sum(),
    }
    
    # Year breakdown
    if 'year' in df.columns:
        summary['rows_per_year'] = df['year'].value_counts().to_dict()
    
    # Weekend type breakdown
    if 'is_sprint_weekend' in df.columns:
        summary['sprint_weekends'] = df['is_sprint_weekend'].sum()
        summary['normal_weekends'] = (~df['is_sprint_weekend']).sum()
    
    # Target variable statistics
    if 'qualifying_position' in df.columns:
        summary['target_min'] = float(df['qualifying_position'].min())
        summary['target_max'] = float(df['qualifying_position'].max())
        summary['target_mean'] = float(df['qualifying_position'].mean())
        summary['target_std'] = float(df['qualifying_position'].std())
    
    return summary

def generate_validation_report(df: pd.DataFrame) -> str:
    """
    Generate formatted validation report for dataset.
    
    Creates human-readable summary including dataset overview, year breakdown,
    weekend types, target variable stats, and missing value analysis.
    
    Args:
        df: Feature DataFrame
        
    Returns:
        Multi-line formatted report string
        
    Example:
        >>> report = generate_validation_report(df)
        >>> print(report)
        ============================================================
        DATA VALIDATION REPORT
        ============================================================
         Dataset Overview:
           Rows: 1,353
           ...
    """
    summary = summarize_dataset(df)
    
    report = []
    report.append("=" * 60)
    report.append("DATA VALIDATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Dataset overview
    report.append(" Dataset Overview:")
    report.append(f"   Rows: {summary['total_rows']:,}")
    report.append(f"   Columns: {summary['total_columns']}")
    report.append(f"   Missing values: {summary['missing_values']} ({summary['missing_pct']:.2f}%)")
    report.append(f"   Duplicate rows: {summary['duplicate_rows']}")
    report.append("")
    
    # Year breakdown
    if 'rows_per_year' in summary:
        report.append(" Rows per year:")
        for year, count in sorted(summary['rows_per_year'].items()):
            report.append(f"   {year}: {count:,}")
        report.append("")
    
    # Weekend types
    if 'sprint_weekends' in summary:
        report.append(" Weekend types:")
        report.append(f"   Sprint: {summary['sprint_weekends']}")
        report.append(f"   Normal: {summary['normal_weekends']}")
        report.append("")
    
    # Target variable
    if 'target_min' in summary:
        report.append(" Target variable (qualifying_position):")
        report.append(f"   Range: {summary['target_min']:.0f} - {summary['target_max']:.0f}")
        report.append(f"   Mean: {summary['target_mean']:.2f}")
        report.append(f"   Std Dev: {summary['target_std']:.2f}")
        report.append("")
    
    # Missing values per column
    missing = df.isnull().sum()
    if missing.sum() > 0:
        report.append("⚠️  Missing values by column:")
        for col, count in missing[missing > 0].items():
            pct = 100 * count / len(df)
            report.append(f"   {col}: {count} ({pct:.1f}%)")
        report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)