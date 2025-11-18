"""
API configuration for F1 Classification Models.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_Q3_PATH = BASE_DIR / "models" / "q3_classifier.pkl"
MODEL_TOP3_PATH = BASE_DIR / "models" / "top3_classifier.pkl"
MODEL_Q2_PATH = BASE_DIR / "models" / "q2_classifier.pkl"
METADATA_PATH = BASE_DIR / "models" / "classification_metadata.json"
DATA_PATH = BASE_DIR / "data" / "features" / "ml_features_2022_2025.parquet"

# API settings
API_TITLE = "F1 Qualifying Classifier API"
API_VERSION = "2.0.0"
API_DESCRIPTION = """
🏎️ **F1 Qualifying Classification API**

Predict qualifying outcomes using historical data, telemetry, and weather features.

**Models Available:**
- **Q3 Qualification**: Will driver make top 10? (78.8% accuracy vs 50% baseline)
- **Top 3 Finish**: Will driver podium in quali? (89.1% accuracy vs 15% baseline)
- **Qualifying Round**: Which round will they reach? Q1/Q2/Q3 (70% accuracy vs 33% baseline)

**Features:**
- Binary classification with probability scores
- Multi-class classification for qualifying rounds
- Confidence intervals based on model performance
- Historical feature lookups
- Model explanation and feature importance
"""
