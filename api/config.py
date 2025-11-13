"""
API configuration.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
DATA_PATH = BASE_DIR / "data" / "features" / "ml_features_2022_2025.parquet"

# API settings
API_TITLE = "F1 Qualifying Predictor API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
🏎️ **F1 Qualifying Position Prediction API**

Predict qualifying positions using historical data, telemetry, and weather features.

**Features:**
- Real-time qualifying predictions
- Confidence intervals
- Historical feature lookups
- Model explanation
"""