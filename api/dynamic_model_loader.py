"""
Dynamic Model Loader for API

Ensures API always uses the latest trained model version.

REPLACE the static model loading in api/main.py with this.
"""

import joblib
import json
from pathlib import Path
from datetime import datetime
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class DynamicModelLoader:
    """
    Loads the latest model version dynamically.
    
    Features:
    - Auto-detects latest version
    - Caches models for performance
    - Reloads if new version deployed
    - Thread-safe
    """
    
    def __init__(self, models_dir: str = "../models"):
        self.models_dir = Path(models_dir)
        self._cached_version = None
        self._cached_models = {}
        self._cached_metadata = None
        self._last_check = None
        self._check_interval = 60  # Check for new version every 60s
        
    def get_active_version(self) -> str:
        """Get currently active model version."""
        version_file = self.models_dir / "active_version.txt"
        
        if version_file.exists():
            return version_file.read_text().strip()
        
        # Fallback: use latest version directory
        version_dirs = list(self.models_dir.glob("v*"))
        if version_dirs:
            latest = sorted(version_dirs)[-1]
            return latest.name
        
        return "v1"  # Default
    
    def _should_reload(self) -> bool:
        """Check if models should be reloaded."""
        # Check every N seconds
        now = datetime.now()
        if self._last_check:
            elapsed = (now - self._last_check).total_seconds()
            if elapsed < self._check_interval:
                return False
        
        self._last_check = now
        
        # Check if version changed
        current_version = self.get_active_version()
        if current_version != self._cached_version:
            logger.info(f"🔄 New model version detected: {self._cached_version} → {current_version}")
            return True
        
        return False
    
    def load_models(self, force_reload: bool = False) -> dict:
        """
        Load models (uses cache if available).
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            dict with all models
        """
        # Check if reload needed
        if not force_reload and not self._should_reload() and self._cached_models:
            return self._cached_models
        
        # Get active version
        version = self.get_active_version()
        
        logger.info(f"📦 Loading models: {version}")
        
        # Load models
        models = {
            'q3': joblib.load(self.models_dir / "q3_classifier.pkl"),
            'top3': joblib.load(self.models_dir / "top3_classifier.pkl"),
            'round': joblib.load(self.models_dir / "round_classifier.pkl")  # Changed from q2
        }
        
        # Load metadata
        metadata_file = self.models_dir / "classification_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Cache
        self._cached_version = version
        self._cached_models = models
        self._cached_metadata = metadata
        
        logger.info(f"✅ Loaded models version: {version}")
        
        return models
    
    def get_metadata(self) -> dict:
        """Get model metadata."""
        if not self._cached_metadata:
            self.load_models()
        return self._cached_metadata
    
    def get_features(self) -> list:
        """Get feature list from metadata."""
        metadata = self.get_metadata()
        return metadata.get('features', [])


# Global instance (singleton)
_model_loader = None

def get_model_loader() -> DynamicModelLoader:
    """Get or create model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = DynamicModelLoader()
    return _model_loader


# Convenience functions for API
def load_latest_models() -> dict:
    """Load latest model version."""
    loader = get_model_loader()
    return loader.load_models()


def get_model_version() -> str:
    """Get current model version."""
    loader = get_model_loader()
    return loader.get_active_version()


def get_model_metadata() -> dict:
    """Get model metadata."""
    loader = get_model_loader()
    return loader.get_metadata()


# Example usage in API
if __name__ == "__main__":
    # Test dynamic loading
    print("Testing dynamic model loader...")
    
    models = load_latest_models()
    print(f"✅ Loaded {len(models)} models")
    
    version = get_model_version()
    print(f"📦 Active version: {version}")
    
    metadata = get_model_metadata()
    print(f"📊 Model accuracies:")
    for model_name, model_info in metadata.get('models', {}).items():
        print(f"   {model_name}: {model_info.get('accuracy', 0):.1%}")