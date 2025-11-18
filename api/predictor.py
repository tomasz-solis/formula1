"""
Prediction logic for F1 Classification Models.
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from config import MODEL_Q3_PATH, MODEL_TOP3_PATH, MODEL_Q2_PATH, METADATA_PATH, DATA_PATH

logger = logging.getLogger(__name__)


class QualifyingPredictor:
    """
    Qualifying classification predictor using trained ML models.
    
    Provides three types of predictions:
    1. Q3 Binary: Will driver make top 10?
    2. Top 3 Binary: Will driver podium in quali?
    3. Q2 Multi-class: Which round will they reach (Q1/Q2/Q3)?
    """
    
    def __init__(self):
        """Initialize predictor by loading models and metadata."""
        self.model_q3 = None
        self.model_top3 = None
        self.model_q2 = None
        self.features = None
        self.metadata = None
        self.historical_data = None
        self._load_models()
        self._load_data()
    
    def _load_models(self):
        """Load all trained models and metadata."""
        try:
            # Load Q3 binary classifier
            self.model_q3 = joblib.load(MODEL_Q3_PATH)
            logger.info(f"✅ Q3 model loaded from {MODEL_Q3_PATH}")
            
            # Load Top 3 binary classifier
            self.model_top3 = joblib.load(MODEL_TOP3_PATH)
            logger.info(f"✅ Top 3 model loaded from {MODEL_TOP3_PATH}")
            
            # Load Q2 multi-class classifier
            self.model_q2 = joblib.load(MODEL_Q2_PATH)
            logger.info(f"✅ Q2 model loaded from {MODEL_Q2_PATH}")
            
            # Load metadata
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            
            self.features = self.metadata['features']
            logger.info(f"✅ Loaded {len(self.features)} features")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
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
            for feat in ['circuit_avg_position', 'circuit_best_position', 'circuit_worst_position']:
                if feat in circuit_df.columns:
                    features_dict[feat] = circuit_df[feat].iloc[-1]
        
        # Recent form (last 5 sessions)
        recent_df = hist_df.sort_values('year').tail(5)
        for feat in ['recent_avg_position', 'recent_best_position', 'form_trend']:
            if feat in recent_df.columns:
                features_dict[feat] = recent_df[feat].iloc[-1]
        
        # Team features
        for feat in ['team_circuit_avg_position', 'team_momentum', 'team_recent_avg']:
            if feat in hist_df.columns:
                features_dict[feat] = hist_df[feat].iloc[-1]
        
        # Weather performance
        for feat in ['wet_dry_delta', 'wet_avg_position', 'dry_avg_position']:
            if feat in hist_df.columns:
                features_dict[feat] = hist_df[feat].iloc[-1]
        
        # Telemetry (use recent average)
        telemetry_features = ['max_throttle_ratio', 'brake_max_g', 'brake_avg_g']
        for feat in telemetry_features:
            if feat in recent_df.columns:
                features_dict[feat] = recent_df[feat].mean()
        
        return features_dict
    
    def _prepare_features(
        self,
        driver: str,
        circuit: str,
        year: int,
        manual_features: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Prepare feature vector for prediction.
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Session year
            manual_features: Optional manual overrides
            
        Returns:
            DataFrame with feature vector
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
        return pd.DataFrame([features_dict])[self.features]
    
    def predict_q3(
        self,
        driver: str,
        circuit: str,
        year: int,
        manual_features: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Predict Q3 qualification (top 10).
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Session year
            manual_features: Optional overrides
            
        Returns:
            Tuple of (will_make_q3, probability, features_used)
        """
        X = self._prepare_features(driver, circuit, year, manual_features)
        
        # Predict
        prediction = self.model_q3.predict(X)[0]
        probability = self.model_q3.predict_proba(X)[0][1]  # Probability of class 1 (Q3)
        
        features_used = X.iloc[0].to_dict()
        
        return bool(prediction), float(probability), features_used
    
    def predict_top3(
        self,
        driver: str,
        circuit: str,
        year: int,
        manual_features: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Predict Top 3 finish in qualifying.
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Session year
            manual_features: Optional overrides
            
        Returns:
            Tuple of (will_make_top3, probability, features_used)
        """
        X = self._prepare_features(driver, circuit, year, manual_features)
        
        # Predict
        prediction = self.model_top3.predict(X)[0]
        probability = self.model_top3.predict_proba(X)[0][1]  # Probability of class 1 (Top 3)
        
        features_used = X.iloc[0].to_dict()
        
        return bool(prediction), float(probability), features_used
    
    def predict_round(
        self,
        driver: str,
        circuit: str,
        year: int,
        manual_features: Optional[Dict[str, float]] = None
    ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """
        Predict which qualifying round driver will reach (Q1/Q2/Q3).
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Session year
            manual_features: Optional overrides
            
        Returns:
            Tuple of (predicted_round, probabilities_dict, features_used)
        """
        X = self._prepare_features(driver, circuit, year, manual_features)
        
        # Predict
        prediction_code = self.model_q2.predict(X)[0]
        probabilities = self.model_q2.predict_proba(X)[0]
        
        # Map prediction code to round name
        round_map = {0: 'Q1', 1: 'Q2', 2: 'Q3'}
        predicted_round = round_map[prediction_code]
        
        # Create probabilities dict
        prob_dict = {
            'Q1': float(probabilities[0]),
            'Q2': float(probabilities[1]),
            'Q3': float(probabilities[2])
        }
        
        features_used = X.iloc[0].to_dict()
        
        return predicted_round, prob_dict, features_used
    
    def predict_all(
        self,
        driver: str,
        circuit: str,
        year: int,
        manual_features: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Get all predictions at once.
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Session year
            manual_features: Optional overrides
            
        Returns:
            Dictionary with all prediction results
        """
        # Get Q3 prediction
        q3_pred, q3_prob, q3_features = self.predict_q3(driver, circuit, year, manual_features)
        
        # Get Top 3 prediction
        top3_pred, top3_prob, top3_features = self.predict_top3(driver, circuit, year, manual_features)
        
        # Get Round prediction
        round_pred, round_probs, round_features = self.predict_round(driver, circuit, year, manual_features)
        
        return {
            'q3': {
                'prediction': q3_pred,
                'probability': q3_prob,
                'confidence': self._get_confidence(q3_prob),
                'features': q3_features
            },
            'top3': {
                'prediction': top3_pred,
                'probability': top3_prob,
                'confidence': self._get_confidence(top3_prob),
                'features': top3_features
            },
            'round': {
                'prediction': round_pred,
                'probabilities': round_probs,
                'confidence': self._get_confidence(max(round_probs.values())),
                'features': round_features
            }
        }
    
    def _get_confidence(self, probability: float) -> str:
        """
        Convert probability to confidence level.
        
        Args:
            probability: Prediction probability
            
        Returns:
            Confidence level: high/medium/low
        """
        if probability >= 0.75:
            return "high"
        elif probability >= 0.55:
            return "medium"
        else:
            return "low"
    
    def get_model_info(self) -> Dict:
        """Get model metadata."""
        return {
            'models': self.metadata['models'],
            'features': self.features,
            'feature_count': len(self.features)
        }