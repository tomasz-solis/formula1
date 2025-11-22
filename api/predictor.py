"""
Qualifying Predictor with Dynamic Model Loading.

Handles predictions for Q3, Top3, and Round classification.
Supports both static (backward compatible) and dynamic model loading.
"""

import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import sys

# Add parent directory to path for dynamic loader
sys.path.append(str(Path(__file__).parent.parent))

# Try to import dynamic loader (optional for backward compatibility)
try:
    from dynamic_model_loader import get_model_version, load_latest_models
    DYNAMIC_LOADING_AVAILABLE = True
except ImportError:
    DYNAMIC_LOADING_AVAILABLE = False


class QualifyingPredictor:
    """
    Predict qualifying outcomes with optional dynamic model loading.
    
    Features:
    - Q3 qualification (binary: top 10 or not)
    - Top 3 finish (binary: podium or not)
    - Qualifying round (multi-class: Q1/Q2/Q3)
    - Dynamic model reloading (if enabled)
    - Historical feature lookup
    """
    
    def __init__(
        self, 
        models_dir: str = "../models",
        data_dir: str = "../data",
        use_dynamic_loading: bool = False
    ):
        """
        Initialize predictor.
        
        Args:
            models_dir: Path to models directory
            data_dir: Path to data directory
            use_dynamic_loading: Enable dynamic model loading (auto-reload new versions)
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.use_dynamic_loading = use_dynamic_loading and DYNAMIC_LOADING_AVAILABLE
        
        # Dynamic loading state
        self._current_version = None
        self._last_reload_check = None
        self._reload_interval = 60  # Check every 60 seconds
        
        # Initialize models
        if self.use_dynamic_loading:
            print("🔄 Initializing with DYNAMIC model loading...")
            self._load_models_dynamic()
        else:
            print("📦 Initializing with STATIC model loading...")
            self._load_models_static()
        
        # Load historical data
        self._load_historical_data()
        
        print(f"✅ Predictor ready with {len(self.features)} features")
    
    def _load_models_static(self):
        """Load models statically (once at startup, never reload)."""
        try:
            # Load models
            self.model_q3 = joblib.load(self.models_dir / "q3_classifier.pkl")
            self.model_top3 = joblib.load(self.models_dir / "top3_classifier.pkl")
            self.model_q2 = joblib.load(self.models_dir / "round_classifier.pkl")  # Changed from q2
            
            # Load metadata
            metadata_file = self.models_dir / "classification_metadata.json"
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            self.features = self.metadata['features']
            
            print(f"   Loaded 3 models (static mode)")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")
    
    def _load_models_dynamic(self):
        """Load models dynamically (can reload when new version available)."""
        try:
            # Load latest model version
            models = load_latest_models()
            
            self.model_q3 = models['q3']
            self.model_top3 = models['top3']
            self.model_q2 = models['round']
            
            # Load metadata
            metadata_file = self.models_dir / "classification_metadata.json"
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            self.features = self.metadata['features']
            self._current_version = get_model_version()
            self._last_reload_check = datetime.now()
            
            print(f"   Loaded 3 models (dynamic mode): {self._current_version}")
            
        except Exception as e:
            # Fallback to static loading
            print(f"   ⚠️  Dynamic loading failed, falling back to static: {e}")
            self._load_models_static()
            self.use_dynamic_loading = False
    
    def reload_if_needed(self, force: bool = False):
        """
        Check if models need reloading and reload if necessary.
        
        Only works if dynamic loading is enabled.
        
        Args:
            force: Force reload even if interval hasn't elapsed
        """
        if not self.use_dynamic_loading:
            return  # Static loading, no reload
        
        # Check if enough time passed since last check
        now = datetime.now()
        if not force and self._last_reload_check:
            elapsed = (now - self._last_reload_check).total_seconds()
            if elapsed < self._reload_interval:
                return  # Too soon to check
        
        self._last_reload_check = now
        
        # Check if version changed
        try:
            current_version = get_model_version()
            
            if current_version != self._current_version or force:
                print(f"🔄 Reloading models: {self._current_version} → {current_version}")
                self._load_models_dynamic()
        except Exception as e:
            print(f"⚠️  Reload check failed: {e}")
    
    def _load_historical_data(self):
        """Load historical feature data for lookups."""
        try:
            features_file = self.data_dir / "features/ml_features.parquet"
            
            if features_file.exists():
                self.historical_data = pd.read_parquet(features_file)
                print(f"   Loaded historical data: {len(self.historical_data)} records")
            else:
                print(f"   ⚠️  Historical data not found: {features_file}")
                # Create empty dataframe with expected columns
                self.historical_data = pd.DataFrame(columns=self.features + ['driver', 'event', 'year'])
                
        except Exception as e:
            print(f"   ⚠️  Failed to load historical data: {e}")
            self.historical_data = pd.DataFrame(columns=self.features + ['driver', 'event', 'year'])
    
    def _get_historical_features(
        self,
        driver: str,
        circuit: str,
        year: int
    ) -> Dict[str, float]:
        """
        Lookup historical features for a driver at a circuit.
        
        Args:
            driver: Driver abbreviation (e.g., "VER")
            circuit: Circuit name (e.g., "Monza")
            year: Year
            
        Returns:
            Dictionary of feature values
        """
        # Filter to driver's past data (before this year)
        driver_data = self.historical_data[
            (self.historical_data['driver'] == driver) &
            (self.historical_data['year'] < year)
        ].copy()
        
        if len(driver_data) == 0:
            # New driver - use population medians
            feature_dict = {
                feat: self.historical_data[feat].median() 
                for feat in self.features 
                if feat in self.historical_data.columns
            }
        else:
            # Get most recent values
            driver_data = driver_data.sort_values('year')
            latest = driver_data.iloc[-1]
            
            feature_dict = {
                feat: latest[feat] if feat in latest.index else self.historical_data[feat].median()
                for feat in self.features
            }
        
        return feature_dict
    
    def _prepare_features(
        self,
        driver: str,
        circuit: str,
        year: int,
        manual_features: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Prepare feature vector for prediction.
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Year
            manual_features: Optional manual feature overrides (weather, etc.)
            
        Returns:
            DataFrame with single row of features
        """
        # Get historical features
        feature_dict = self._get_historical_features(driver, circuit, year)
        
        # Override with manual features if provided
        if manual_features:
            for key, value in manual_features.items():
                if value is not None:
                    feature_dict[key] = value
        
        # Fill any missing features with median
        for feat in self.features:
            if feat not in feature_dict or pd.isna(feature_dict[feat]):
                if feat in self.historical_data.columns:
                    feature_dict[feat] = self.historical_data[feat].median()
                else:
                    feature_dict[feat] = 0.0
        
        # Create feature vector in correct order
        return pd.DataFrame([feature_dict])[self.features]
    
    def _get_confidence(self, probability: float) -> str:
        """
        Convert probability to confidence level.
        
        Args:
            probability: Prediction probability (0-1)
            
        Returns:
            Confidence level: "high", "medium", or "low"
        """
        if probability >= 0.75 or probability <= 0.25:
            return "high"
        elif probability >= 0.55 and probability <= 0.45:
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
            manual_features: Optional manual features (weather, etc.)
            
        Returns:
            (will_make_q3, probability, features_used)
        """
        # Prepare features
        X = self._prepare_features(driver, circuit, year, manual_features)
        
        # Predict
        prediction = self.model_q3.predict(X)[0]
        probability = self.model_q3.predict_proba(X)[0][1]  # Probability of class 1 (Q3)
        
        # Get feature values for explanation
        features_used = X.iloc[0].to_dict()
        
        return bool(prediction), float(probability), features_used
    
    def predict_top3(
        self,
        driver: str,
        circuit: str,
        year: int,
        manual_features: Optional[Dict] = None
    ) -> Tuple[bool, float, Dict]:
        """
        Predict Top 3 finish in qualifying.
        
        Args:
            driver: Driver abbreviation
            circuit: Circuit name
            year: Year
            manual_features: Optional manual features (weather, etc.)
            
        Returns:
            (will_make_top3, probability, features_used)
        """
        # Prepare features
        X = self._prepare_features(driver, circuit, year, manual_features)
        
        # Predict
        prediction = self.model_top3.predict(X)[0]
        probability = self.model_top3.predict_proba(X)[0][1]  # Probability of class 1 (Top 3)
        
        # Get feature values for explanation
        features_used = X.iloc[0].to_dict()
        
        return bool(prediction), float(probability), features_used
    
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
            manual_features: Optional manual features (weather, etc.)
            
        Returns:
            (predicted_round, probabilities, features_used)
        """
        # Prepare features
        X = self._prepare_features(driver, circuit, year, manual_features)
        
        # Predict
        prediction_code = self.model_q2.predict(X)[0]
        probabilities_array = self.model_q2.predict_proba(X)[0]
        
        # Map prediction code to round name
        round_map = {0: 'Q1', 1: 'Q2', 2: 'Q3'}
        predicted_round = round_map[prediction_code]
        
        # Create probabilities dictionary
        probabilities = {
            'Q1': float(probabilities_array[0]),
            'Q2': float(probabilities_array[1]),
            'Q3': float(probabilities_array[2])
        }
        
        # Get feature values for explanation
        features_used = X.iloc[0].to_dict()
        
        return predicted_round, probabilities, features_used
    
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
            manual_features: Optional manual features (weather, etc.)
            
        Returns:
            Dictionary with all predictions
        """
        # Prepare features once
        X = self._prepare_features(driver, circuit, year, manual_features)
        
        # Q3 prediction
        q3_pred = self.model_q3.predict(X)[0]
        q3_prob = self.model_q3.predict_proba(X)[0][1]
        
        # Top 3 prediction
        top3_pred = self.model_top3.predict(X)[0]
        top3_prob = self.model_top3.predict_proba(X)[0][1]
        
        # Round prediction
        round_pred_code = self.model_q2.predict(X)[0]
        round_probs_array = self.model_q2.predict_proba(X)[0]
        
        round_map = {0: 'Q1', 1: 'Q2', 2: 'Q3'}
        round_pred = round_map[round_pred_code]
        
        round_probs = {
            'Q1': float(round_probs_array[0]),
            'Q2': float(round_probs_array[1]),
            'Q3': float(round_probs_array[2])
        }
        
        # Combine results
        return {
            'q3': {
                'prediction': bool(q3_pred),
                'probability': float(q3_prob),
                'confidence': self._get_confidence(q3_prob)
            },
            'top3': {
                'prediction': bool(top3_pred),
                'probability': float(top3_prob),
                'confidence': self._get_confidence(top3_prob)
            },
            'round': {
                'prediction': round_pred,
                'probabilities': round_probs,
                'confidence': self._get_confidence(max(round_probs.values()))
            }
        }
    
    def get_model_info(self) -> Dict:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary with model info
        """
        info = {
            'timestamp': self.metadata.get('timestamp', 'unknown'),
            'features_count': len(self.features),
            'models': {},
            'dynamic_loading': self.use_dynamic_loading
        }
        
        # Add model accuracies
        for model_name, model_info in self.metadata.get('models', {}).items():
            info['models'][model_name] = {
                'accuracy': model_info.get('accuracy', 0.0),
                'baseline': model_info.get('baseline', 0.0)
            }
        
        # Add version if dynamic loading
        if self.use_dynamic_loading:
            try:
                info['active_version'] = get_model_version()
            except:
                info['active_version'] = 'unknown'
        
        return info


# Example usage
if __name__ == "__main__":
    # Test predictor
    print("Testing QualifyingPredictor...\n")
    
    # Initialize with dynamic loading
    predictor = QualifyingPredictor(use_dynamic_loading=True)
    
    # Test prediction
    driver = "VER"
    circuit = "Monza"
    year = 2025
    
    print(f"\nPredicting for {driver} at {circuit} {year}:")
    print("-" * 50)
    
    # Get all predictions
    results = predictor.predict_all(
        driver=driver,
        circuit=circuit,
        year=year,
        manual_features={
            'avg_rainfall': 0.0,
            'avg_track_temp': 35.0,
            'avg_air_temp': 30.0
        }
    )
    
    # Display results
    print(f"\nQ3: {results['q3']['prediction']} ({results['q3']['probability']:.1%})")
    print(f"Top 3: {results['top3']['prediction']} ({results['top3']['probability']:.1%})")
    print(f"Round: {results['round']['prediction']}")
    print(f"  Q1: {results['round']['probabilities']['Q1']:.1%}")
    print(f"  Q2: {results['round']['probabilities']['Q2']:.1%}")
    print(f"  Q3: {results['round']['probabilities']['Q3']:.1%}")
    
    # Model info
    print("\n" + "="*50)
    print("Model Information:")
    print("="*50)
    info = predictor.get_model_info()
    for key, value in info.items():
        if key == 'models':
            print(f"\nModel Accuracies:")
            for model_name, model_data in value.items():
                print(f"  {model_name}: {model_data['accuracy']:.1%}")
        else:
            print(f"{key}: {value}")