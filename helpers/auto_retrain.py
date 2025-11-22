"""
Auto-Retraining Module

Integrated into data pipeline (main.py) to automatically:
1. Detect new race data
2. Retrain models if needed
3. Deploy new version
4. Track performance

Called automatically at the end of main.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoRetrainer:
    """
    Handles automatic model retraining when new data arrives.
    
    Triggered by main.py after feature engineering completes.
    """
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load continuous learning configuration."""
        config_file = self.models_dir / "learning_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default config
        return {
            'enabled': True,
            'min_new_races': 1,  # Retrain after 1 new race
            'improvement_threshold': 0.01,  # Must improve by 1%
            'auto_deploy': True
        }
    
    def _save_config(self):
        """Save configuration."""
        config_file = self.models_dir / "learning_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def detect_new_data(self, features_file: Path) -> dict:
        """
        Detect if new race data has been added.
        
        Args:
            features_file: Path to latest features file
            
        Returns:
            dict with new_data: bool, new_races: int, details: list
        """
        # Load current features
        current_df = pd.read_parquet(features_file)
        
        # Load training history
        history_file = self.models_dir / "training_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            # Get last training timestamp
            last_training = history.get('last_training_data', {})
            last_race_count = last_training.get('race_count', 0)
            last_timestamp = last_training.get('timestamp', None)
        else:
            last_race_count = 0
            last_timestamp = None
        
        # Count unique races in current data
        current_race_count = len(current_df.groupby(['year', 'event']))
        
        # Detect new races
        new_races = current_race_count - last_race_count
        
        result = {
            'new_data': new_races > 0,
            'new_races': new_races,
            'current_race_count': current_race_count,
            'last_race_count': last_race_count,
            'last_training': last_timestamp,
            'trigger_retrain': new_races >= self.config['min_new_races']
        }
        
        if result['new_data']:
            logger.info(f"✨ New data detected: {new_races} new race(s)")
            logger.info(f"   Total races: {last_race_count} → {current_race_count}")
        
        return result
    
    def should_retrain(self, new_data_info: dict) -> bool:
        """
        Decide if retraining should happen.
        
        Args:
            new_data_info: Output from detect_new_data()
            
        Returns:
            bool: True if should retrain
        """
        if not self.config['enabled']:
            logger.info("⏸️  Auto-retraining disabled in config")
            return False
        
        if not new_data_info['trigger_retrain']:
            logger.info(f"⏭️  Not enough new data ({new_data_info['new_races']} races, need {self.config['min_new_races']})")
            return False
        
        return True
    
    def retrain_models(self, features_df: pd.DataFrame, features_list: list) -> dict:
        """
        Retrain all models with updated data.
        
        Args:
            features_df: Full features dataframe
            features_list: List of feature column names
            
        Returns:
            dict with new models and metrics
        """
        logger.info("🔄 Starting automatic retraining...")
        
        # Prepare data - FILTER OUT NaN positions first
        valid_positions = features_df['qualifying_position'].notna()
        features_df = features_df[valid_positions].copy()
        
        logger.info(f"   Filtered to {len(features_df)} records with valid qualifying positions")
        
        X = features_df[features_list]
        y_q3 = (features_df['qualifying_position'] <= 10).astype(int)
        y_top3 = (features_df['qualifying_position'] <= 3).astype(int)
        
        # Round classification
        y_round = pd.cut(
            features_df['qualifying_position'],
            bins=[0, 15, 20, 25],
            labels=[2, 1, 0]
        ).astype(int)
        
        # Temporal split (last 20% for validation)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_q3_train, y_q3_val = y_q3[:split_idx], y_q3[split_idx:]
        y_top3_train, y_top3_val = y_top3[:split_idx], y_top3[split_idx:]
        y_round_train, y_round_val = y_round[:split_idx], y_round[split_idx:]
        
        # Train Q3 model
        logger.info("   Training Q3 classifier...")
        model_q3 = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        model_q3.fit(X_train, y_q3_train)
        q3_acc = accuracy_score(y_q3_val, model_q3.predict(X_val))
        
        # Train Top3 model
        logger.info("   Training Top3 classifier...")
        model_top3 = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=15,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        model_top3.fit(X_train, y_top3_train)
        top3_acc = accuracy_score(y_top3_val, model_top3.predict(X_val))
        
        # Train Round model
        logger.info("   Training Round classifier...")
        model_round = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        model_round.fit(X_train, y_round_train)
        round_acc = accuracy_score(y_round_val, model_round.predict(X_val))
        
        logger.info(f"   ✅ Q3: {q3_acc:.1%} accuracy")
        logger.info(f"   ✅ Top3: {top3_acc:.1%} accuracy")
        logger.info(f"   ✅ Round: {round_acc:.1%} accuracy")
        
        return {
            'q3': {'model': model_q3, 'accuracy': q3_acc},
            'top3': {'model': model_top3, 'accuracy': top3_acc},
            'round': {'model': model_round, 'accuracy': round_acc}
        }
    
    def compare_with_existing(self, new_metrics: dict) -> dict:
        """
        Compare new model performance with existing.
        
        Args:
            new_metrics: Metrics from newly trained models
            
        Returns:
            dict with comparison results and deploy decision
        """
        # Load current model metadata
        metadata_file = self.models_dir / "classification_metadata.json"
        
        # Check if models exist
        models_exist = all([
            (self.models_dir / "q3_classifier.pkl").exists(),
            (self.models_dir / "top3_classifier.pkl").exists(),
            (self.models_dir / "q2_classifier.pkl").exists()
        ])
        
        if not models_exist or not metadata_file.exists():
            # No existing models, this is initial training
            logger.info("🆕 No existing models found - this is initial training")
            return {
                'should_deploy': True,
                'reason': 'Initial model training (no existing models)',
                'avg_improvement': 1.0,  # 100% improvement from nothing
                'improvements': {
                    'q3': new_metrics['q3']['accuracy'],
                    'top3': new_metrics['top3']['accuracy'],
                    'round': new_metrics['round']['accuracy']
                },
                'current_metrics': {'q3': 0.0, 'top3': 0.0, 'round': 0.0},
                'new_metrics': {k: v['accuracy'] for k, v in new_metrics.items()}
            }
        
        # Load existing metrics
        with open(metadata_file, 'r') as f:
            current_metadata = json.load(f)
        
        current_metrics = {
            'q3': current_metadata['models']['q3_binary']['accuracy'],
            'top3': current_metadata['models']['top3_binary']['accuracy'],
            'round': current_metadata['models']['q2_multiclass']['accuracy']
        }
        
        # Calculate improvements
        improvements = {
            'q3': new_metrics['q3']['accuracy'] - current_metrics['q3'],
            'top3': new_metrics['top3']['accuracy'] - current_metrics['top3'],
            'round': new_metrics['round']['accuracy'] - current_metrics['round']
        }
        
        # Average improvement
        avg_improvement = np.mean(list(improvements.values()))
        
        # Decision
        threshold = self.config['improvement_threshold']
        should_deploy = avg_improvement >= threshold or all(imp >= 0 for imp in improvements.values())
        
        result = {
            'should_deploy': should_deploy,
            'avg_improvement': avg_improvement,
            'threshold': threshold,
            'improvements': improvements,
            'current_metrics': current_metrics,
            'new_metrics': {k: v['accuracy'] for k, v in new_metrics.items()}
        }
        
        if should_deploy:
            logger.info(f"✅ New models better: avg improvement {avg_improvement:+.1%}")
        else:
            logger.info(f"⏭️  New models worse: avg improvement {avg_improvement:+.1%}")
            logger.info(f"   Keeping current models")
        
        return result
    
    def deploy_models(self, models: dict, metadata: dict):
        """
        Deploy new model version.
        
        Args:
            models: Dictionary of trained models
            metadata: Deployment metadata
        """
        # Create version
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = self.models_dir / f"v{version}"
        version_dir.mkdir(exist_ok=True, parents=True)
        
        # Save models to version directory
        for name, model_data in models.items():
            model_file = version_dir / f"{name}_classifier.pkl"
            joblib.dump(model_data['model'], model_file)
        
        # Save version metadata
        version_metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'accuracies': {k: v['accuracy'] for k, v in models.items()},
            **metadata
        }
        
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(version_metadata, f, indent=2)
        
        # Copy to main models directory (active models)
        for name, model_data in models.items():
            active_file = self.models_dir / f"{name}_classifier.pkl"
            joblib.dump(model_data['model'], active_file)
        
        # Update main metadata
        main_metadata = {
            'timestamp': datetime.now().isoformat(),
            'active_version': version,
            'features': metadata.get('features', []),
            'models': {
                'q3_binary': {
                    'accuracy': models['q3']['accuracy'],
                    'baseline': 0.50
                },
                'top3_binary': {
                    'accuracy': models['top3']['accuracy'],
                    'baseline': 0.15
                },
                'q2_multiclass': {
                    'accuracy': models['round']['accuracy'],
                    'baseline': 0.33
                }
            }
        }
        
        with open(self.models_dir / "classification_metadata.json", 'w') as f:
            json.dump(main_metadata, f, indent=2)
        
        # Update active version marker
        with open(self.models_dir / "active_version.txt", 'w') as f:
            f.write(version)
        
        # Update training history
        self._update_history(version, models, metadata)
        
        logger.info(f"🚀 Deployed new model version: v{version}")
    
    def _update_history(self, version: str, models: dict, metadata: dict):
        """Update training history file."""
        history_file = self.models_dir / "training_history.json"
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = {'versions': [], 'metrics': []}
        
        # Add new entry
        history['versions'].append(version)
        history['metrics'].append({
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'accuracies': {k: v['accuracy'] for k, v in models.items()},
            'improvements': metadata.get('improvements', {})
        })
        
        # Track last training data
        history['last_training_data'] = {
            'timestamp': datetime.now().isoformat(),
            'race_count': metadata.get('race_count', 0)
        }
        
        # Keep only last 20 versions
        if len(history['versions']) > 20:
            history['versions'] = history['versions'][-20:]
            history['metrics'] = history['metrics'][-20:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def run_auto_retrain(self, features_file: str) -> dict:
        """
        Main auto-retraining workflow.
        
        Called by main.py after feature engineering.
        
        Args:
            features_file: Path to features parquet file
            
        Returns:
            dict with status and results
        """
        features_path = Path(features_file)
        
        if not features_path.exists():
            logger.error(f"❌ Features file not found: {features_file}")
            return {'status': 'error', 'message': 'Features file not found'}
        
        # Step 1: Detect new data
        new_data_info = self.detect_new_data(features_path)
        
        # Step 2: Decide if should retrain
        if not self.should_retrain(new_data_info):
            return {
                'status': 'skipped',
                'reason': 'No retraining needed',
                'new_data_info': new_data_info
            }
        
        # Step 3: Load data
        logger.info("📂 Loading features for retraining...")
        features_df = pd.read_parquet(features_path)
        
        # Step 3.5: Determine feature list
        metadata_file = self.models_dir / "classification_metadata.json"
        
        if metadata_file.exists():
            # Load from existing metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            features_list = metadata['features']
            logger.info(f"   Using {len(features_list)} features from existing metadata")
        else:
            # First time training - infer features from dataframe
            logger.info("🆕 First time training - inferring features from data")
            
            # Exclude target and identifier columns
            exclude_cols = [
                'qualifying_position', 'race_position', 'sprint_position',
                'driver', 'event', 'year', 'session', 'team',
                'is_q3', 'is_top3', 'qualifying_round'
            ]
            
            features_list = [
                col for col in features_df.columns 
                if col not in exclude_cols and features_df[col].dtype in ['int64', 'float64']
            ]
            
            logger.info(f"   Inferred {len(features_list)} features from data")
            logger.info(f"   Features: {features_list[:10]}..." if len(features_list) > 10 else f"   Features: {features_list}")
        
        # Validate features exist in dataframe
        missing_features = [f for f in features_list if f not in features_df.columns]
        if missing_features:
            logger.error(f"❌ Missing features in data: {missing_features}")
            return {
                'status': 'error', 
                'message': f'Missing features: {missing_features}'
            }
        
        # Step 4: Retrain models
        new_models = self.retrain_models(features_df, features_list)
        
        # Step 5: Compare with existing
        comparison = self.compare_with_existing(new_models)
        
        # Step 6: Deploy if better
        if comparison['should_deploy'] and self.config['auto_deploy']:
            metadata_info = {
                'improvements': comparison['improvements'],
                'race_count': new_data_info['current_race_count'],
                'features': features_list
            }
            self.deploy_models(new_models, metadata_info)
            
            return {
                'status': 'deployed',
                'version': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'comparison': comparison,
                'new_data_info': new_data_info
            }
        else:
            return {
                'status': 'not_deployed',
                'reason': 'New models not better than existing',
                'comparison': comparison
            }


# Integration point for main.py
def auto_retrain_if_needed(features_file: str = "data/features/ml_features.parquet"):
    """
    Call this at the end of main.py to trigger auto-retraining.
    
    Args:
        features_file: Path to features file
    """
    retrainer = AutoRetrainer()
    result = retrainer.run_auto_retrain(features_file)
    
    if result['status'] == 'deployed':
        logger.info("🎉 Auto-retraining complete! New models deployed.")
    elif result['status'] == 'skipped':
        logger.info("⏭️  Auto-retraining skipped: " + result['reason'])
    elif result['status'] == 'not_deployed':
        logger.info("⏭️  Models trained but not deployed: " + result['reason'])
    
    return result


if __name__ == "__main__":
    # Test run
    result = auto_retrain_if_needed()
    print(json.dumps(result, indent=2))