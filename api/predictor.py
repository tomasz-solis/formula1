"""
Prediction logic and feature engineering.
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from .config import MODEL_PATH, METADATA_PATH, DATA_PATH

logger = logging.getLogger(__name__)


class QualifyingPredictor:
    """
    Qualifying position predictor using trained ML model.
    """
    
    def __init__(self):
        """Initialize predictor by loading model and metadata."""
        self.model = None
        self.features = None
        self.metadata = None
        self.historical_data = None
        self._load_model()
        self._load_data()
    
    def _load_model(self):
        """Load trained model and metadata."""
        try:
            self.model = joblib.load(MODEL_PATH)
            logger.info(f"✅ Model loaded from {MODEL_PATH}")
            
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            
            self.features = self.metadata['features']
            logger.info(f"✅ Loaded {len(self.features)} features")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _load_data(self):
        """Load historical data for feature lookup."""
        try:
            self.historical_data = pd.read_parquet(DATA_PATH)
            logger.info(f"✅ Loaded {len(self.historical_data):,} historical records")
        except Exception as e:
            logger.error(f"❌ Failed to load historical data: {e}")
            raise
    
    def get_historical_features(
        self,
        driver: str,
        circuit: str,
        year: int
    ) -> Dict[str, float]:
        """
        Lookup historical features for driver at circuit.
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Session year
            
        Returns:
            Dictionary of feature values
        """
        # Filter to relevant historical data (before current year)
        hist_df = self.historical_data[
            (self.historical_data['driver'] == driver) &
            (self.historical_data['year'] < year)
        ].copy()
        
        features_dict = {}
        
        if len(hist_df) == 0:
            logger.warning(f"No historical data for {driver}")
            # Return median values as fallback
            for feat in self.features:
                if feat in self.historical_data.columns:
                    features_dict[feat] = self.historical_data[feat].median()
            return features_dict
        
        # Circuit-specific features
        circuit_df = hist_df[hist_df['event'].str.contains(circuit, case=False, na=False)]
        
        if len(circuit_df) > 0:
            for feat in ['circuit_avg_position', 'circuit_best_position', 'circuit_position_std']:
                if feat in circuit_df.columns:
                    features_dict[feat] = circuit_df[feat].iloc[-1]
        
        # Recent form (last 5 sessions)
        recent_df = hist_df.sort_values('year').tail(5)
        for feat in ['recent_avg_position', 'recent_best_position', 'form_trend']:
            if feat in recent_df.columns:
                features_dict[feat] = recent_df[feat].iloc[-1]
        
        # Team features
        for feat in ['team_circuit_avg_position', 'team_momentum']:
            if feat in hist_df.columns:
                features_dict[feat] = hist_df[feat].iloc[-1]
        
        # Weather performance
        if 'wet_dry_delta' in hist_df.columns:
            features_dict['wet_dry_delta'] = hist_df['wet_dry_delta'].iloc[-1]
        
        # Telemetry (use recent average)
        telemetry_features = ['max_throttle_ratio', 'brake_max_g', 'braking_events']
        for feat in telemetry_features:
            if feat in recent_df.columns:
                features_dict[feat] = recent_df[feat].mean()
        
        return features_dict
    
    def predict(
        self,
        driver: str,
        circuit: str,
        year: int,
        manual_features: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """
        Predict qualifying position.
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Session year
            manual_features: Optional manual feature overrides
            
        Returns:
            Tuple of (predicted_position, ci_lower, ci_upper, features_used)
        """
        # Get historical features
        features_dict = self.get_historical_features(driver, circuit, year)
        
        # Override with manual features if provided
        if manual_features:
            features_dict.update({k: v for k, v in manual_features.items() if v is not None})
        
        # Fill missing features with median
        for feat in self.features:
            if feat not in features_dict:
                features_dict[feat] = self.historical_data[feat].median()
        
        # Create feature vector
        X = pd.DataFrame([features_dict])[self.features]
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Compute confidence interval (using MAE as approximation)
        mae = self.metadata['mae']
        ci_lower = max(1.0, prediction - 1.96 * mae)
        ci_upper = min(20.0, prediction + 1.96 * mae)
        
        return prediction, ci_lower, ci_upper, features_dict
    
    def get_model_info(self) -> Dict:
        """Get model metadata."""
        return {
            'model_name': self.metadata['model_name'],
            'mae': self.metadata['mae'],
            'r2': self.metadata['r2'],
            'features': self.features,
            'feature_count': len(self.features)
        }