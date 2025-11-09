"""
Unit tests for feature engineering pipeline.

Tests cover:
- Data loading functions
- Validation functions
- Merge operations
- Aggregation logic
- Error handling

Run tests:
    pytest tests/test_feature_engineering.py -v

Author: Tomasz Solis
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.feature_engineering import (
    validate_years,
    validate_dataframe_columns,
    merge_driver_circuit_data,
    fix_missing_circuit_data,
    _aggregate_single_driver_race,
    _add_sprint_weekend_features,
    _add_normal_weekend_features,
    MIN_F1_YEAR,
    MAX_F1_YEAR,
    MERGE_KEYS,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_driver_data():
    """Create sample driver telemetry data."""
    return pd.DataFrame({
        'year': [2024, 2024, 2024],
        'event': ['Bahrain Grand Prix', 'Bahrain Grand Prix', 'Bahrain Grand Prix'],
        'session': ['FP1', 'FP2', 'FP3'],
        'driver': ['VER', 'VER', 'VER'],
        'grand_prix': ['Bahrain', 'Bahrain', 'Bahrain'],
        'location': ['Sakhir', 'Sakhir', 'Sakhir'],
        'max_throttle_ratio': [0.70, 0.72, 0.71],
        'brake_max_g': [5.2, 5.3, 5.4],
        'brake_avg_g': [2.1, 2.2, 2.3],
        'drs_activations': [10, 12, 11],
        'degradation_slope': [-0.05, -0.06, -0.055],
        'avg_rainfall': [0, 0, 0],
        'avg_track_temp': [35.0, 36.0, 37.0],
    })


@pytest.fixture
def sample_circuit_data():
    """Create sample circuit characteristics data."""
    return pd.DataFrame({
        'year': [2024, 2024, 2024],
        'event': ['Bahrain Grand Prix', 'Bahrain Grand Prix', 'Bahrain Grand Prix'],
        'session': ['FP1', 'FP2', 'FP3'],
        'slow_corners': [3, 3, 3],
        'medium_corners': [7, 7, 7],
        'fast_corners': [5, 5, 5],
        'chicanes': [2, 2, 2],
        'avg_speed': [210.5, 210.5, 210.5],
        'top_speed': [320.0, 320.0, 320.0],
        'real_altitude': [50, 50, 50],
    })


@pytest.fixture
def sample_sprint_data():
    """Create sample sprint weekend data."""
    return pd.DataFrame({
        'year': [2024, 2024],
        'event': ['Austrian Grand Prix', 'Austrian Grand Prix'],
        'session': ['FP1', 'SQ'],
        'driver': ['VER', 'VER'],
        'grand_prix': ['Austria', 'Austria'],
        'location': ['Spielberg', 'Spielberg'],
        'max_throttle_ratio': [0.68, 0.73],
        'brake_max_g': [4.8, 5.1],
        'brake_avg_g': [2.0, 2.2],
        'drs_activations': [8, 10],
        'degradation_slope': [-0.04, -0.05],
        'avg_rainfall': [0, 0],
        'avg_track_temp': [28.0, 30.0],
        'slow_corners': [4, 4],
        'medium_corners': [6, 6],
        'fast_corners': [3, 3],
        'chicanes': [1, 1],
        'avg_speed': [230.0, 230.0],
        'top_speed': [340.0, 340.0],
        'real_altitude': [700, 700],
    })


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidateYears:
    """Test year validation function."""
    
    def test_valid_years(self):
        """Test that valid years pass validation."""
        validate_years([2022, 2023, 2024])  # Should not raise
    
    def test_empty_years_raises_error(self):
        """Test that empty years list raises ValueError."""
        with pytest.raises(ValueError, match="Years list cannot be empty"):
            validate_years([])
    
    def test_invalid_year_too_old_raises_error(self):
        """Test that year before MIN_F1_YEAR raises ValueError."""
        with pytest.raises(ValueError, match="Years must be between"):
            validate_years([1900])
    
    def test_invalid_year_too_new_raises_error(self):
        """Test that year after MAX_F1_YEAR raises ValueError."""
        with pytest.raises(ValueError, match="Years must be between"):
            validate_years([3000])
    
    def test_non_list_raises_error(self):
        """Test that non-list input raises ValueError."""
        with pytest.raises(ValueError, match="Years must be a list"):
            validate_years(2024)  # type: ignore
    
    def test_non_integer_years_raises_error(self):
        """Test that non-integer years raise ValueError."""
        with pytest.raises(ValueError, match="All years must be integers"):
            validate_years([2024, "2025"])  # type: ignore


class TestValidateDataframeColumns:
    """Test DataFrame column validation."""
    
    def test_valid_columns(self):
        """Test that valid columns pass validation."""
        df = pd.DataFrame({'year': [2024], 'event': ['Test']})
        validate_dataframe_columns(df, ['year', 'event'], "Test")  # Should not raise
    
    def test_missing_columns_raises_error(self):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({'year': [2024]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe_columns(df, ['year', 'event'], "Test")


# =============================================================================
# MERGE TESTS
# =============================================================================

class TestMergeDriverCircuitData:
    """Test driver-circuit data merging."""
    
    def test_successful_merge(self, sample_driver_data, sample_circuit_data):
        """Test normal merge with matching keys."""
        result = merge_driver_circuit_data(sample_driver_data, sample_circuit_data)
        
        assert len(result) == len(sample_driver_data)
        assert 'slow_corners' in result.columns
        assert 'max_throttle_ratio' in result.columns
        assert result['slow_corners'].iloc[0] == 3
    
    def test_merge_preserves_all_driver_rows(self, sample_driver_data, sample_circuit_data):
        """Test that merge doesn't drop driver rows."""
        result = merge_driver_circuit_data(sample_driver_data, sample_circuit_data)
        assert len(result) == len(sample_driver_data)
    
    def test_missing_merge_key_raises_error(self):
        """Test that missing merge key raises ValueError."""
        drivers = pd.DataFrame({'driver': ['VER']})
        circuits = pd.DataFrame({'event': ['Bahrain GP']})
        
        with pytest.raises(ValueError, match="missing required columns"):
            merge_driver_circuit_data(drivers, circuits)


# =============================================================================
# AGGREGATION TESTS
# =============================================================================

class TestAggregateSingleDriverRace:
    """Test single driver-race aggregation."""
    
    def test_normal_weekend_aggregation(self, sample_driver_data, sample_circuit_data):
        """Test aggregation for normal weekend (FP1+FP2+FP3)."""
        # Merge data first
        merged = pd.merge(
            sample_driver_data, 
            sample_circuit_data, 
            on=['year', 'event', 'session']
        )
        
        result = _aggregate_single_driver_race(
            2024, 'Bahrain Grand Prix', 'VER', merged
        )
        
        assert result['year'] == 2024
        assert result['event'] == 'Bahrain Grand Prix'
        assert result['driver'] == 'VER'
        assert result['is_sprint_weekend'] == False
        assert result['best_throttle_ratio'] == 0.72  # Max of [0.70, 0.72, 0.71]
        assert 'fp3_throttle_ratio' in result
        assert result['sprint_quali_throttle'] is np.nan
    
    def test_sprint_weekend_aggregation(self, sample_sprint_data):
        """Test aggregation for sprint weekend (FP1+SQ)."""
        result = _aggregate_single_driver_race(
            2024, 'Austrian Grand Prix', 'VER', sample_sprint_data
        )
        
        assert result['is_sprint_weekend'] == True
        assert result['has_sprint_quali_data'] == True
        assert result['sprint_quali_throttle'] == 0.73
        assert result['fp3_throttle_ratio'] is np.nan
    
    def test_corner_percentage_calculation(self, sample_driver_data, sample_circuit_data):
        """Test that corner percentages are calculated correctly."""
        merged = pd.merge(
            sample_driver_data, 
            sample_circuit_data, 
            on=['year', 'event', 'session']
        )
        
        result = _aggregate_single_driver_race(
            2024, 'Bahrain Grand Prix', 'VER', merged
        )
        
        total = result['total_corners']
        assert total == 15  # 3 + 7 + 5
        assert result['slow_corner_pct'] == pytest.approx(3/15)
        assert result['medium_corner_pct'] == pytest.approx(7/15)
        assert result['fast_corner_pct'] == pytest.approx(5/15)


class TestAddSprintWeekendFeatures:
    """Test sprint weekend feature extraction."""
    
    def test_sprint_features_extracted(self, sample_sprint_data):
        """Test that sprint qualifying features are extracted."""
        row = {'year': 2024, 'event': 'Austrian GP', 'driver': 'VER'}
        result = _add_sprint_weekend_features(row, sample_sprint_data)
        
        assert result['has_sprint_quali_data'] == True
        assert result['sprint_quali_throttle'] == 0.73
        assert result['fp3_throttle_ratio'] is np.nan
    
    def test_missing_sprint_quali_handled(self):
        """Test handling of missing sprint qualifying data."""
        df = pd.DataFrame({
            'session': ['FP1'],
            'max_throttle_ratio': [0.70],
            'brake_max_g': [5.0],
            'avg_track_temp': [30.0],
        })
        
        row = {}
        result = _add_sprint_weekend_features(row, df)
        
        assert result['has_sprint_quali_data'] == False
        assert result['sprint_quali_throttle'] is np.nan


class TestAddNormalWeekendFeatures:
    """Test normal weekend feature extraction."""
    
    def test_normal_features_extracted(self, sample_driver_data):
        """Test that FP3 features are extracted."""
        row = {}
        result = _add_normal_weekend_features(row, sample_driver_data)
        
        assert result['fp3_throttle_ratio'] == 0.71
        assert result['fp3_brake_max_g'] == 5.4
        assert result['has_sprint_quali_data'] == False
        assert result['sprint_quali_throttle'] is np.nan


# =============================================================================
# DATA CLEANING TESTS
# =============================================================================

class TestFixMissingCircuitData:
    """Test circuit data imputation."""
    
    def test_imputes_from_other_year(self):
        """Test that missing circuit data is filled from other years."""
        df = pd.DataFrame({
            'year': [2023, 2024],
            'event': ['Bahrain Grand Prix', 'Bahrain Grand Prix'],
            'driver': ['VER', 'VER'],
            'slow_corners': [np.nan, 3.0],
            'fast_corners': [np.nan, 7.0],
            'avg_speed_circuit': [np.nan, 210.0],
            'slow_corner_pct': [np.nan, 0.2],
            'medium_corner_pct': [np.nan, 0.5],
            'fast_corner_pct': [np.nan, 0.3],
        })
        
        result = fix_missing_circuit_data(df)
        
        # 2023 should now have same values as 2024
        assert result.loc[0, 'slow_corners'] == 3.0
        assert result.loc[0, 'fast_corners'] == 7.0
        assert result.loc[0, 'avg_speed_circuit'] == 210.0
    
    def test_drops_altitude_column(self):
        """Test that real_altitude column is dropped."""
        df = pd.DataFrame({
            'year': [2024],
            'event': ['Bahrain GP'],
            'driver': ['VER'],
            'real_altitude': [50],
            'slow_corners': [3],
            'fast_corners': [7],
            'avg_speed_circuit': [210],
            'slow_corner_pct': [0.2],
            'medium_corner_pct': [0.5],
            'fast_corner_pct': [0.3],
        })
        
        result = fix_missing_circuit_data(df)
        
        assert 'real_altitude' not in result.columns
    
    def test_raises_error_if_cannot_impute(self):
        """Test that error is raised if imputation fails."""
        df = pd.DataFrame({
            'year': [2024],
            'event': ['New Track GP'],  # No other years available
            'driver': ['VER'],
            'slow_corners': [np.nan],
            'fast_corners': [np.nan],
            'avg_speed_circuit': [np.nan],
            'slow_corner_pct': [np.nan],
            'medium_corner_pct': [np.nan],
            'fast_corner_pct': [np.nan],
        })
        
        with pytest.raises(RuntimeError, match="Imputation incomplete"):
            fix_missing_circuit_data(df)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestEndToEndPipeline:
    """Test complete pipeline with realistic data."""
    
    def test_pipeline_produces_valid_output(self, sample_driver_data, sample_circuit_data):
        """Test that pipeline produces valid feature matrix."""
        # Merge
        merged = merge_driver_circuit_data(sample_driver_data, sample_circuit_data)
        
        # Aggregate
        result = _aggregate_single_driver_race(
            2024, 'Bahrain Grand Prix', 'VER', merged
        )
        
        # Validate output structure
        assert 'year' in result
        assert 'event' in result
        assert 'driver' in result
        assert 'best_throttle_ratio' in result
        assert 'avg_throttle_ratio' in result
        assert 'total_corners' in result
        assert 'slow_corner_pct' in result


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])