"""
Data validation utilities for F1 ML pipeline.

Provides functions to validate DataFrames at critical pipeline stages,
ensuring data quality and catching issues early.

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


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_feature_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    target_column: str = 'qualifying_position',
    min_rows: int = 100
) -> None:
    """
    Validate that ML feature DataFrame meets quality standards.
    
    Performs comprehensive checks on the dataset including:
    - Shape validation (sufficient rows)
    - Required columns present
    - Target variable validity
    - Missing value detection
    - Duplicate detection
    - Feature value ranges
    
    Args:
        df: Feature DataFrame to validate
        required_columns: List of columns that must exist
        target_column: Name of target variable column
        min_rows: Minimum acceptable number of rows
        
    Raises:
        ValueError: If validation fails with detailed error messages
        
    Example:
        >>> features = ['best_throttle_ratio', 'avg_brake_max_g', ...]
        >>> validate_feature_dataframe(df, features, 'qualifying_position')
        INFO - ✅ Validation passed: 1353 rows, 36 features, 0 missing
    """
    errors = []
    warnings = []
    
    # 1. Check shape
    if len(df) == 0:
        errors.append("DataFrame is empty (0 rows)")
    elif len(df) < min_rows:
        warnings.append(
            f"Only {len(df)} rows (minimum recommended: {min_rows})"
        )
    
    # 2. Check required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # 3. Check target variable
    if target_column in df.columns:
        # Missing values in target
        missing_target = df[target_column].isna().sum()
        if missing_target > 0:
            errors.append(
                f"Target '{target_column}' has {missing_target} missing values "
                f"({100 * missing_target / len(df):.1f}%)"
            )
        
        # Invalid values in target
        valid_range = (1, 20)
        invalid = ~df[target_column].between(*valid_range)
        if invalid.any():
            bad_vals = df.loc[invalid, target_column].unique()
            errors.append(
                f"Target '{target_column}' has invalid values: {bad_vals}. "
                f"Must be between {valid_range[0]}-{valid_range[1]}"
            )
    else:
        errors.append(f"Target column '{target_column}' not found")
    
    # 4. Check for duplicates
    if 'year' in df.columns and 'event' in df.columns and 'driver' in df.columns:
        dupes = df.duplicated(subset=['year', 'event', 'driver']).sum()
        if dupes > 0:
            errors.append(
                f"Found {dupes} duplicate driver-race combinations "
                f"({100 * dupes / len(df):.1f}%)"
            )
    
    # 5. Check for excessive missing values
    missing_per_col = df[required_columns].isnull().sum()
    high_missing = missing_per_col[missing_per_col > len(df) * 0.3]
    if not high_missing.empty:
        warnings.append(
            f"Columns with >30% missing: {high_missing.to_dict()}"
        )
    
    # 6. Check feature value ranges
    if 'best_throttle_ratio' in df.columns:
        invalid_throttle = ~df['best_throttle_ratio'].between(0, 1)
        if invalid_throttle.any():
            errors.append(
                f"Throttle ratio outside [0,1]: "
                f"{df.loc[invalid_throttle, 'best_throttle_ratio'].describe()}"
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
    Validate that merge keys exist and have compatible data types.
    
    Args:
        df1: First DataFrame to merge
        df2: Second DataFrame to merge
        merge_keys: List of column names to merge on
        context: Description for error messages
        
    Raises:
        ValueError: If merge keys invalid or incompatible
        
    Example:
        >>> validate_merge_keys(drivers, circuits, ['year', 'event', 'session'])
        INFO - ✅ Merge keys validated for merge
    """
    errors = []
    
    # Check keys exist in both dataframes
    missing_df1 = [k for k in merge_keys if k not in df1.columns]
    missing_df2 = [k for k in merge_keys if k not in df2.columns]
    
    if missing_df1:
        errors.append(f"DataFrame 1 missing keys: {missing_df1}")
    if missing_df2:
        errors.append(f"DataFrame 2 missing keys: {missing_df2}")
    
    if errors:
        raise ValueError(f"❌ {context} validation failed:\n" + "\n".join(errors))
    
    # Check data types are compatible
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
    Validate that session data contains expected session types.
    
    Args:
        df: DataFrame with 'session' column
        expected_sessions: List of valid session codes
        
    Raises:
        ValueError: If invalid sessions found
        
    Example:
        >>> validate_session_data(drivers, ['FP1', 'FP2', 'FP3', 'Q', 'R'])
        INFO - ✅ All sessions valid: ['FP1', 'FP2', 'FP3', 'Q', 'R']
    """
    if 'session' not in df.columns:
        raise ValueError("DataFrame missing 'session' column")
    
    actual_sessions = df['session'].unique().tolist()
    invalid = [s for s in actual_sessions if s not in expected_sessions]
    
    if invalid:
        raise ValueError(
            f"❌ Invalid session codes found: {invalid}\n"
            f"Expected: {expected_sessions}"
        )
    
    logger.info("✅ All sessions valid: %s", actual_sessions)


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for dataset.
    
    Args:
        df: Feature DataFrame
        
    Returns:
        Dictionary with summary statistics
        
    Example:
        >>> summary = summarize_dataset(df)
        >>> print(summary['total_rows'])
        1353
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
    
    # Target variable stats
    if 'qualifying_position' in df.columns:
        summary['target_min'] = float(df['qualifying_position'].min())
        summary['target_max'] = float(df['qualifying_position'].max())
        summary['target_mean'] = float(df['qualifying_position'].mean())
        summary['target_std'] = float(df['qualifying_position'].std())
    
    return summary


# =============================================================================
# VALIDATION REPORT
# =============================================================================

def generate_validation_report(df: pd.DataFrame) -> str:
    """
    Generate comprehensive validation report for dataset.
    
    Args:
        df: Feature DataFrame
        
    Returns:
        Formatted report string
        
    Example:
        >>> report = generate_validation_report(df)
        >>> print(report)
        ============================================================
        DATA VALIDATION REPORT
        ============================================================
        ...
    """
    summary = summarize_dataset(df)
    
    report = []
    report.append("=" * 60)
    report.append("DATA VALIDATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Basic stats
    report.append("📊 Dataset Overview:")
    report.append(f"   Rows: {summary['total_rows']:,}")
    report.append(f"   Columns: {summary['total_columns']}")
    report.append(f"   Missing values: {summary['missing_values']} ({summary['missing_pct']:.2f}%)")
    report.append(f"   Duplicate rows: {summary['duplicate_rows']}")
    report.append("")
    
    # Year breakdown
    if 'rows_per_year' in summary:
        report.append("📅 Rows per year:")
        for year, count in sorted(summary['rows_per_year'].items()):
            report.append(f"   {year}: {count:,}")
        report.append("")
    
    # Weekend types
    if 'sprint_weekends' in summary:
        report.append("🏁 Weekend types:")
        report.append(f"   Sprint: {summary['sprint_weekends']}")
        report.append(f"   Normal: {summary['normal_weekends']}")
        report.append("")
    
    # Target variable
    if 'target_min' in summary:
        report.append("🎯 Target variable (qualifying_position):")
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