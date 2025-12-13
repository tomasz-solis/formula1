"""
Qualifying Predictor with Dynamic Model Loading.

Handles predictions for Q3, Top3, and Round classification.
Supports both static (backward compatible) and dynamic model loading.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class QualifyingPredictor:
    """
    Qualifying outcome predictor with multiple classification models.
    
    Models:
    - Q3 Binary: Will driver make Q3 (top 10)?
    - Top 3 Binary: Will driver finish top 3?
    - Round Multi-class: Which round will driver reach (Q1/Q2/Q3)?
    
    Features robust fallback when historical data is missing.
    """
    
    def __init__(self, use_dynamic_loading: bool = False):
        """
        Initialize predictor.
        
        Args:
            use_dynamic_loading: If True, use dynamic model loader (always latest version)
        """
        self.use_dynamic_loading = use_dynamic_loading
        
        if use_dynamic_loading:
            from dynamic_model_loader import load_latest_models, get_model_metadata
            models = load_latest_models()
            self.model_q3 = models['q3']
            self.model_top3 = models['top3']
            self.model_q2 = models['round']
            self.metadata = get_model_metadata()
            self._current_version = None
        else:
            # Static loading
            from api.config import (
                MODEL_Q3_PATH, MODEL_TOP3_PATH, MODEL_Q2_PATH,
                METADATA_PATH, DATA_PATH
            )
            
            self.model_q3 = joblib.load(MODEL_Q3_PATH)
            self.model_top3 = joblib.load(MODEL_TOP3_PATH)
            self.model_q2 = joblib.load(MODEL_Q2_PATH)
            
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
        
        # Load historical data
        from api.config import DATA_PATH
        self.historical_data = pd.read_parquet(DATA_PATH)
        
        # Get feature list
        self.features = self.metadata['features']
        
        logger.info(f"✅ Predictor initialized")
        logger.info(f"   Q3 accuracy: {self.metadata['models']['q3_binary']['accuracy']:.1%}")
        logger.info(f"   Top3 accuracy: {self.metadata['models']['top3_binary']['accuracy']:.1%}")
        logger.info(f"   Round accuracy: {self.metadata['models']['q2_multiclass']['accuracy']:.1%}")
        logger.info(f"   Features: {len(self.features)}")
    
    def reload_if_needed(self, force: bool = False):
        """
        Reload models if new version available (dynamic loading only).
        
        Args:
            force: Force reload even if version hasn't changed
        """
        if not self.use_dynamic_loading:
            return
        
        from dynamic_model_loader import get_model_version, load_latest_models, get_model_metadata
        
        current = get_model_version()
        
        if force or (self._current_version != current):
            logger.info(f" Reloading models: {self._current_version}  {current}")
            models = load_latest_models()
            self.model_q3 = models['q3']
            self.model_top3 = models['top3']
            self.model_q2 = models['round']
            self.metadata = get_model_metadata()
            self._current_version = current
    
    def _get_historical_features(self, driver: str, circuit: str, year: int) -> pd.Series:
        """
        Get historical features for a driver at a specific circuit.
        
        CRITICAL: This method ALWAYS returns valid features (never None).
        
        Fallback hierarchy:
        1. Exact match (driver, circuit, year)
        2. Driver at this circuit (any year - use most recent)
        3. Driver at other circuits (same year)
        4. Driver at other circuits (previous year)
        5. Median imputation (rookie/new driver baseline)
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Year
            
        Returns:
            Series with features (ALWAYS returns valid features)
        """
        # Fallback 1: Exact match
        filtered = self.historical_data[
            (self.historical_data['driver'] == driver) &
            (self.historical_data['event'] == circuit) &
            (self.historical_data['year'] == year)
        ]
        
        if not filtered.empty:
            logger.debug(f"✅ Found exact match for {driver} at {circuit} {year}")
            return filtered.iloc[-1][self.features]
        
        # Fallback 2: Same driver at this circuit (any year - use most recent)
        driver_circuit = self.historical_data[
            (self.historical_data['driver'] == driver) &
            (self.historical_data['event'] == circuit)
        ]
        
        if not driver_circuit.empty:
            most_recent = driver_circuit.nlargest(1, 'year')
            most_recent_year = most_recent['year'].values[0]
            logger.warning(f"⚠️  No {year} data for {driver} at {circuit}")
            logger.info(f"   Using {driver}'s {most_recent_year} data at {circuit}")
            return most_recent.iloc[0][self.features]
        
        # Fallback 3: Driver at other circuits (same year)
        driver_same_year = self.historical_data[
            (self.historical_data['driver'] == driver) &
            (self.historical_data['year'] == year)
        ]
        
        if not driver_same_year.empty:
            avg_features = driver_same_year[self.features].mean()
            logger.warning(f"⚠️  No {circuit} history for {driver}")
            logger.info(f"   Using {driver}'s average from {year} at other circuits")
            return avg_features
        
        # Fallback 4: Driver at other circuits (previous year)
        driver_prev_year = self.historical_data[
            (self.historical_data['driver'] == driver) &
            (self.historical_data['year'] == year - 1)
        ]
        
        if not driver_prev_year.empty:
            avg_features = driver_prev_year[self.features].mean()
            logger.warning(f"⚠️  No {year} data for {driver}")
            logger.info(f"   Using {driver}'s average from {year-1}")
            return avg_features
        
        # Fallback 5: Median imputation (rookie or completely new driver)
        logger.warning(f"⚠️  No historical data for {driver} anywhere")
        logger.info(f"   Using population median (rookie/new driver baseline)")
        median_features = self.historical_data[self.features].median()
        
        return median_features
    
    def _merge_features(self, historical: pd.Series, manual: Optional[Dict] = None) -> np.ndarray:
        """
        Merge historical and manual features.
        
        Args:
            historical: Historical features from data
            manual: Manual feature overrides (e.g., weather)
            
        Returns:
            Complete feature vector
        """
        # Start with historical features
        features_dict = historical.to_dict()
        
        # Override with manual features if provided
        if manual:
            for key, value in manual.items():
                if value is not None and key in features_dict:
                    features_dict[key] = value
        
        # Convert to array in correct order
        feature_array = np.array([features_dict[f] for f in self.features])
        
        # Validate
        if len(feature_array) != len(self.features):
            raise ValueError(
                f"Feature count mismatch: got {len(feature_array)}, expected {len(self.features)}"
            )
        
        return feature_array.reshape(1, -1)
    
    def _get_confidence(self, probability: float) -> str:
        """
        Get confidence level based on probability.
        
        Args:
            probability: Prediction probability
            
        Returns:
            Confidence level: high/medium/low
        """
        if probability >= 0.75 or probability <= 0.25:
            return "high"
        elif probability >= 0.60 or probability <= 0.40:
            return "medium"
        else:
            return "low"
    
    def predict_q3(
        self,
        driver: str,
        circuit: str,
        year: int,
        manual_features: Optional[Dict] = None
    ) -> Tuple[bool, float, Dict]:
        """
        Predict Q3 qualification (top 10).
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Year
            manual_features: Manual feature overrides
            
        Returns:
            Tuple of (will_make_q3, probability, feature_dict)
        """
        # Get features - NOW ALWAYS RETURNS VALID FEATURES
        features_series = self._get_historical_features(driver, circuit, year)
        
        # Merge with manual features
        X = self._merge_features(features_series, manual_features)
        
        # Predict
        probability = float(self.model_q3.predict_proba(X)[0][1])
        will_make_q3 = bool(probability >= 0.5)
        
        # Get feature importance
        if hasattr(self.model_q3, 'feature_importances_'):
            importance = dict(zip(self.features, self.model_q3.feature_importances_))
        else:
            importance = {f: 0.0 for f in self.features}
        
        return will_make_q3, probability, importance
    
    def predict_top3(
        self,
        driver: str,
        circuit: str,
        year: int,
        manual_features: Optional[Dict] = None
    ) -> Tuple[bool, float, Dict]:
        """
        Predict Top 3 finish.
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Year
            manual_features: Manual feature overrides
            
        Returns:
            Tuple of (will_make_top3, probability, feature_dict)
        """
        # Get features - NOW ALWAYS RETURNS VALID FEATURES
        features_series = self._get_historical_features(driver, circuit, year)
        
        # Merge with manual features
        X = self._merge_features(features_series, manual_features)
        
        # Predict
        probability = float(self.model_top3.predict_proba(X)[0][1])
        will_make_top3 = bool(probability >= 0.5)
        
        # Get feature importance
        if hasattr(self.model_top3, 'feature_importances_'):
            importance = dict(zip(self.features, self.model_top3.feature_importances_))
        else:
            importance = {f: 0.0 for f in self.features}
        
        return will_make_top3, probability, importance
    
    def predict_round(
        self,
        driver: str,
        circuit: str,
        year: int,
        manual_features: Optional[Dict] = None
    ) -> Tuple[str, Dict[str, float], Dict]:
        """
        Predict qualifying round (Q1/Q2/Q3).
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Year
            manual_features: Manual feature overrides
            
        Returns:
            Tuple of (predicted_round, probabilities_dict, feature_dict)
        """
        # Get features - NOW ALWAYS RETURNS VALID FEATURES
        features_series = self._get_historical_features(driver, circuit, year)
        
        # Merge with manual features
        X = self._merge_features(features_series, manual_features)
        
        # Predict
        probabilities = self.model_q2.predict_proba(X)[0]
        predicted_class = str(self.model_q2.predict(X)[0])
        
        # Map to round names
        class_names = self.model_q2.classes_
        prob_dict = {
            str(class_name): float(prob) 
            for class_name, prob in zip(class_names, probabilities)
        }
        
        # Get feature importance
        if hasattr(self.model_q2, 'feature_importances_'):
            importance = dict(zip(self.features, self.model_q2.feature_importances_))
        else:
            importance = {f: 0.0 for f in self.features}
        
        return predicted_class, prob_dict, importance
    
    def predict_all(
        self,
        driver: str,
        circuit: str,
        year: int,
        manual_features: Optional[Dict] = None
    ) -> Dict:
        """
        Get all predictions at once.
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Year
            manual_features: Manual feature overrides
            
        Returns:
            Dictionary with all predictions
        """
        # Q3 prediction
        q3_pred, q3_prob, q3_features = self.predict_q3(
            driver, circuit, year, manual_features
        )
        
        # Top 3 prediction
        top3_pred, top3_prob, top3_features = self.predict_top3(
            driver, circuit, year, manual_features
        )
        
        # Round prediction
        round_pred, round_probs, round_features = self.predict_round(
            driver, circuit, year, manual_features
        )
        
        return {
            'q3': {
                'prediction': q3_pred,
                'probability': q3_prob,
                'confidence': self._get_confidence(q3_prob),
                'top_features': dict(sorted(q3_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
            },
            'top3': {
                'prediction': top3_pred,
                'probability': top3_prob,
                'confidence': self._get_confidence(top3_prob),
                'top_features': dict(sorted(top3_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
            },
            'round': {
                'prediction': round_pred,
                'probabilities': round_probs,
                'confidence': self._get_confidence(max(round_probs.values())),
                'top_features': dict(sorted(round_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
            }
        }
    
    def get_model_info(self) -> Dict:
        """Get model information and metadata."""
        return {
            'models': self.metadata['models'],
            'features': self.features,
            'feature_count': len(self.features),
            'timestamp': self.metadata.get('timestamp')
        }