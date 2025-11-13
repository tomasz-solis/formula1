# 🏎️ Formula 1 Qualifying Position Prediction

A machine learning project to predict F1 qualifying results using practice session telemetry, circuit characteristics, and historical performance data.

**Current Status:** ✅ Feature engineering complete, ready for model training

---

## 🎯 Project Goal

Predict qualifying positions (P1-P20) based on:
- Practice session performance (FP1/FP2/FP3)
- Circuit characteristics (corners, altitude, track layout)
- Weather conditions
- Sprint weekend handling
- Historical driver/team performance (coming soon)

**Why this matters:** Demonstrates end-to-end ML workflow from raw telemetry to predictive models.

---

## 📊 Dataset

### Current Dataset Stats
- **1,353 driver-race combinations** (2022-2024 seasons)
- **36 features** extracted from telemetry and circuits
- **Target variable:** Qualifying position (1-20)
- **Zero missing values** after imputation

### Data Sources
- **Driver telemetry:** FastF1 API (throttle, braking, DRS, tire degradation)
- **Circuit profiles:** Track layout, corners, speed characteristics
- **Qualifying results:** Official FIA classification data

### Feature Categories

**Driver Performance (13 features):**
- `best_throttle_ratio` - Peak throttle usage across practice
- `avg_throttle_ratio` - Average throttle consistency
- `best_brake_max_g` - Peak braking force
- `fp3_throttle_ratio` - FP3 performance (most predictive)
- `sprint_quali_throttle` - Sprint qualifying data (sprint weekends only)
- Tire degradation, DRS activations, braking intensity

**Circuit Characteristics (10 features):**
- `slow_corner_pct` - Percentage of slow-speed corners
- `medium_corner_pct` - Percentage of medium-speed corners
- `fast_corner_pct` - Percentage of high-speed corners
- `total_corners` - Total corner count
- `chicanes` - Chicane count
- `avg_speed_circuit` - Average track speed
- `top_speed_circuit` - Maximum achievable speed

**Weather & Conditions (3 features):**
- `rain_in_practice` - Rainfall detected in any practice session
- `avg_track_temp` - Mean track temperature
- `track_temp_std` - Temperature variation (track evolution)

**Weekend Format (3 features):**
- `is_sprint_weekend` - Sprint vs normal weekend flag
- `has_sprint_quali_data` - Sprint qualifying data available
- `sessions_available` - Which practice sessions occurred

---

## 🏗️ Project Structure
```
formula1/
├── main.py                           # Data pipeline orchestration
├── helpers/
│   ├── feature_engineering.py        # ML feature extraction
│   ├── historical_features.py        # Historical performance
│   ├── general_utils.py              # Session loading, caching
│   ├── driver_utils.py               # Driver telemetry features
│   ├── circuit_utils.py              # Circuit profile extraction
│   └── prediction.py                 # SSOT classification exports
├── data/
│   ├── driver/                       # Driver session profiles (CSV)
│   ├── circuit/                      # Circuit profiles (CSV)
│   ├── driver_timing/                # Detailed lap telemetry (Parquet)
│   ├── predictions/ssot/             # Official qualifying results (CSV)
│   └── processed/                    # ML-ready features (CSV)
├── features/
│   └── ml_features_2022_2025.parquet # Historical performance features
├── models/
│   ├── best_model.pkl                # Trained Random Forest (MAE 3.168)
│   └── model_metadata.json           # Feature list and metrics
├── EDA/
│   ├── 00_wip.ipynb                  # Experimentation notebook
│   ├── 01_general.ipynb              # Track clustering analysis
│   ├── 02_feature_exploration.ipynb  # EDA with Plotly visualizations
│   ├── 02_feature_importance.ipynb   # Multi-method feature ranking (F-stat, MI, RF, SHAP)
│   └── 04_baseline_ml_model.ipynb    # Baseline models (RF/XGBoost/LightGBM)
└── api/
    ├── __init__.py             # Package initialization
    ├── config.py               # API configuration and paths
    ├── main.py                 # FastAPI application with endpoints
    ├── models.py               # Pydantic request/response schemas
    └── predictor.py            # Prediction logic and feature engineering
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/tomasz-solis/formula1.git
cd formula1
python -m venv f1env
source f1env/bin/activate  # Windows: f1env\Scripts\activate
pip install -r requirements.txt
```

### 2. Build Raw Data (if needed)
```bash
# Extract telemetry and results for 2022-2024
python main.py --from 2022 --to 2024
```

This creates:
- `data/driver/20XX_driver_profiles.csv` - Driver telemetry per session
- `data/circuit/20XX_circuit_profiles.csv` - Circuit characteristics
- `data/predictions/ssot/20XX_qualifying.csv` - Qualifying results

### 3. Generate ML Features
```bash
# Build feature matrix for modeling
python helpers/feature_engineering.py
```

Output: `data/processed/qualifying_features.csv` (1,353 rows × 36 features)

---

## 4. Feature Engineering Pipeline

The `feature_engineering.py` module transforms raw telemetry into ML-ready features:

### Pipeline Steps

1. **Load raw data** - Driver profiles, circuit data, qualifying results (3 years)
2. **Filter to practice sessions** - Only FP1/FP2/FP3/SQ (not qualifying itself!)
3. **Merge driver + circuit** - Combine telemetry with track characteristics
4. **Aggregate sessions** - One row per driver-race (handle sprint weekends)
5. **Add target variable** - Merge with qualifying positions
6. **Fix missing data** - Impute missing circuit features from other years

### Sprint Weekend Handling

**Challenge:** Sprint weekends have different practice structure
- **Normal weekend:** FP1 → FP2 → FP3 → Qualifying
- **Sprint weekend:** FP1 → Sprint Qualifying → Sprint Race → Qualifying

**Solution:** 
- Use `sprint_quali_throttle` feature for sprint weekends
- Flag `is_sprint_weekend` so model learns different patterns
- Impute missing FP3 data with FP1 for sprint races

### Key Design Decisions

**Why "best" features?**
- Teams often sandbag in practice (hide true pace)
- `best_throttle_ratio` captures peak performance across all sessions
- More predictive than single-session averages

**Why drop altitude?**
- Only 2/24 tracks have significant altitude (Mexico, Brazil)
- Low variance feature → minimal predictive value
- Simplified model, no loss in performance

---

## 5. Historical Features for ML

The pipeline now computes historical performance features for machine learning:

### Quick Start
````bash
# Compute features for multiple seasons
python main.py --from 2022 --to 2025

# Skip feature computation for faster runs
python main.py --from 2024 --to 2024 --no-features
````

### Features Computed

**Circuit History** (3-year lookback)
- Average qualifying position at each circuit
- Best position at circuit
- Consistency (standard deviation)

**Recent Form** (5-race rolling window)
- Rolling average position
- Best recent position
- Momentum trend (negative = improving)

**Weather Performance**
- Average position in dry conditions
- Average position in wet conditions  
- Wet-dry delta (negative = better in rain)

**Team Performance**
- Team average at each circuit
- Team development momentum

### Output
````
data/features/ml_features_2022_2025.parquet
````

**Shape:** ~4,200 driver-sessions × 42 features

**Columns:**
- Base: year, event, session, driver, team
- Telemetry: throttle, braking, DRS, tire degradation
- Weather: rainfall, track temp, air temp
- **Historical**: circuit history, recent form, momentum
- **Targets**: qualifying_position, race_position

### Feature Quality
````
Missing data by feature:
  circuit_avg_position    : 27.5% (expected - new circuits/drivers)
  recent_avg_position     :  0.0% ✅
  form_trend              :  0.0% ✅
  wet_dry_delta           :  1.6% ✅
  team_circuit_avg_pos    : 10.1%
  team_momentum           :  1.2% ✅
````

### Usage in ML
````python
import pandas as pd

# Load features
df = pd.read_parquet('data/features/ml_features_2022_2025.parquet')

# Filter to qualifying sessions
df_qual = df[df['qualifying_position'].notna()]

# Features for modeling
features = [
    'circuit_avg_position',
    'recent_avg_position', 
    'form_trend',
    'wet_dry_delta',
    'team_circuit_avg_position',
    'max_throttle_ratio',
    'avg_rainfall'
]

X = df_qual[features]
y = df_qual['qualifying_position']
````

### Configuration

Customize in `main.py` → `compute_historical_features()`:
````python
features_df = compute_historical_features(
    driver_profiles=df_driver,
    circuit_profiles=df_circuit,
    lookback_years=3,      # Years of circuit history
    form_window=5,         # Races for recent form
    rain_threshold=0.1,    # mm/h for "wet" classification
    start_year=2022,
    end_year=2025
)
````

---

## 6. ML Analysis Notebooks

Interactive Jupyter notebooks for complete machine learning workflow.

### Notebooks Overview

#### **`EDA/02_feature_exploration.ipynb`** - Exploratory Data Analysis
Comprehensive analysis of ML-ready features:
- Dataset overview (810K+ driver-sessions across 4 seasons)
- Feature distributions by category (telemetry, historical, weather)
- Missing data analysis and validation
- Correlation analysis with target variables
- Historical feature effectiveness validation
- Weather impact analysis (rain specialists identification)
- Team performance trends over time

**Visualizations:** 100% Plotly for interactivity (hover, zoom, pan)

---

#### **`EDA/03_feature_importance.ipynb`** - Feature Importance Analysis
Multi-method feature ranking to identify most predictive features:

**Methods:**
1. **F-statistic** - Linear relationships
2. **Mutual Information** - Non-linear relationships  
3. **Random Forest** - Tree-based importance
4. **Permutation Importance** - Model-agnostic impact
5. **SHAP Values** - Interpretable ML explanations

**Output:** Consolidated ranking averaging 4 different importance methods for robust feature selection.

---

#### **`EDA/04_baseline_ml_model.ipynb`** - Baseline Model Training
Train and evaluate baseline ML models:

**Models Trained:**
- **Random Forest** (n_estimators=200, max_depth=15)
- **XGBoost** (optional, if available)
- **LightGBM** (optional, if available)

**Train/Test Split:**
- Train: 2022-2024 seasons
- Test: 2025 season (time-based split, no data leakage)

**Baseline Performance:**
```
Model: Random Forest
MAE:  3.168 positions
EMSE: 3.934 positions
R²:   0.525
```

**Analysis:**
- Actual vs predicted scatter plots
- Error distribution analysis
- Best/worst predictions identification
- Error by driver (top teams easier to predict)
- Error by position (midfield hardest to predict)

**Output:** 
- Saved model: `models/best_model.pkl`
- Metadata: `models/model_metadata.json`

---

### Running the Notebooks
```bash
# Start Jupyter
jupyter notebook EDA/

# Or run specific notebook
jupyter notebook EDA/02_feature_exploration.ipynb
```

**Run in order:**
1. `02_feature_exploration.ipynb` - Understand the data
2. `03_feature_importance.ipynb` - Select best features
3. `04_baseline_ml_model.ipynb` - Train and evaluate models

**Prerequisites:**
```bash
# Ensure ML features exist
python main.py --from 2022 --to 2025

# Verify file
ls data/features/ml_features_2022_2025.parquet
```

**Note:** All visualizations use Plotly - no matplotlib required!

---

### Model Deployment Ready

The trained model from `04_baseline_ml_model.ipynb` is ready for deployment:
```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/best_model.pkl')

# Load feature metadata
import json
with open('models/model_metadata.json') as f:
    metadata = json.load(f)

features = metadata['features']  # List of required features

# Make predictions
X_new = pd.DataFrame([{...}])  # New driver data
prediction = model.predict(X_new[features])
print(f"Predicted position: P{int(round(prediction[0]))}")
```

---

## 7. Prediction API

A REST API for predicting qualifying positions in real-time.

### What it does

The API takes basic information (driver, circuit, weather) and returns a predicted qualifying position using the trained ML model. It automatically looks up historical performance data and combines it with current conditions.

### Quick start

Start the API server:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser to `http://localhost:8000/docs` for interactive documentation.

### Making a prediction

**Using the interactive docs:**

1. Go to `http://localhost:8000/docs`
2. Click on `POST /predict`
3. Click "Try it out"
4. Fill in the form:
```json
   {
     "driver": "VER",
     "circuit": "Monza",
     "year": 2025
   }
```
5. Click "Execute"

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "driver": "VER",
    "circuit": "Monza",
    "year": 2025,
    "avg_rainfall": 0.0,
    "avg_track_temp": 35.0
  }'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "driver": "VER",
        "circuit": "Monza",
        "year": 2025,
        "avg_rainfall": 0.0,
        "avg_track_temp": 35.0
    }
)

result = response.json()
print(f"Predicted position: P{result['predicted_position_rounded']}")
print(f"Confidence: P{result['confidence_interval_lower']:.1f} to P{result['confidence_interval_upper']:.1f}")
```

### What you get back

The API returns:
- **Predicted position**: The most likely qualifying result (e.g., P2)
- **Confidence interval**: Range of likely positions (e.g., P1 to P3)
- **Features used**: What historical data influenced the prediction
- **Model accuracy**: How reliable the prediction is (MAE: 3.17 positions)

Example response:
```json
{
  "driver": "VER",
  "circuit": "Monza",
  "year": 2025,
  "predicted_position": 1.8,
  "predicted_position_rounded": 2,
  "confidence_interval_lower": 1.2,
  "confidence_interval_upper": 3.4,
  "model_name": "Random Forest",
  "model_mae": 3.168,
  "features_used": {
    "circuit_avg_position": 1.5,
    "recent_avg_position": 2.1,
    "team_circuit_avg_position": 1.8,
    "avg_rainfall": 0.0,
    "avg_track_temp": 35.0
  }
}
```

### Available endpoints

**Health check:**
```bash
curl http://localhost:8000/health
```
Returns whether the model is loaded and ready.

**Model information:**
```bash
curl http://localhost:8000/model/info
```
Shows which features the model uses and its accuracy metrics.

**List drivers:**
```bash
curl http://localhost:8000/drivers
```
Returns all drivers available in the historical data.

**List circuits:**
```bash
curl http://localhost:8000/circuits
```
Returns all circuits available in the historical data.

### How it works

1. You provide basic information (driver, circuit, optional weather)
2. API looks up driver's historical performance at that circuit
3. API looks up driver's recent form (last 5 races)
4. API looks up team performance at that circuit
5. Model combines all features to make prediction
6. Returns predicted position with confidence interval

### Optional weather overrides

If you know specific conditions, you can override the defaults:
```json
{
  "driver": "HAM",
  "circuit": "Silverstone",
  "year": 2025,
  "avg_rainfall": 2.5,        // Wet session
  "avg_track_temp": 18.0,
  "avg_air_temp": 15.0,
  "tyre_age": 0,              // Fresh tires
  "is_fresh_tyre": true
}
```

This is useful for:
- Predicting how conditions affect performance
- What-if scenarios (what if it rains?)
- Comparing driver performance in different conditions

### Missing data handling

If the API doesn't have historical data for a driver/circuit combination:
- Falls back to median values from all drivers
- Still makes a prediction but with wider confidence intervals
- Warns you in the response that historical data is limited

### Requirements

The API needs these files to work:
- `models/best_model.pkl` - Trained model (generated from notebook 04)
- `models/model_metadata.json` - Feature list and metrics
- `data/features/ml_features_2022_2025.parquet` - Historical data for lookups

### Performance

- First request: ~500ms (loading historical data)
- Subsequent requests: ~50ms (data cached in memory)
- Memory usage: ~2GB (historical data)
- Concurrent requests: Supported (async/await)

### Error handling

The API handles common issues gracefully:
- Unknown driver: Returns list of available drivers
- Unknown circuit: Returns list of available circuits
- Invalid year: Suggests valid range
- Missing historical data: Uses fallback values with warning
- Model not loaded: Returns 503 Service Unavailable

### Interactive documentation

FastAPI automatically generates interactive docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

These let you:
- See all available endpoints
- Try requests directly in your browser
- View request/response schemas
- Download OpenAPI specification

No need for Postman or curl - just click and test!

---

## 8. Development Status

### Completed

**Data Pipeline**
- Raw telemetry extraction from FastF1 API
- Circuit profile generation
- Classification exports (qualifying/race results)

**Feature Engineering**
- 42 features from telemetry, weather, and historical performance
- Circuit-specific history (3-year lookback)
- Recent form tracking (5-race rolling window)
- Weather-adjusted performance metrics
- Team momentum indicators

**Analysis & Modeling**
- Exploratory data analysis with 810K+ driver-sessions
- Multi-method feature importance analysis
- Baseline models trained (Random Forest, XGBoost, LightGBM)
- Model evaluation and error analysis
- Best model: Random Forest with MAE 3.17 positions, R² 0.525

**Deployment**
- REST API with FastAPI
- Automatic feature lookup from historical data
- Interactive API documentation (Swagger)
- Prediction endpoints with confidence intervals

### In Progress

**Model Improvements**
- Hyperparameter tuning with Optuna
- Ensemble methods (stacking multiple models)
- Cross-validation for more robust metrics

**API Enhancements**
- Docker containerization for easy deployment
- Rate limiting and authentication
- Batch prediction endpoint
- WebSocket for live race weekend predictions

### Planned

**Advanced Modeling**
- Deep learning models (LSTM for sequential patterns)
- Race result prediction (beyond qualifying)
- Pit stop strategy optimization
- Tire degradation forecasting

**Data Sources**
- Preseason testing data parser (2026 regulation changes)
- Real-time weather API integration
- Team radio sentiment analysis

**Deployment**
- Deploy to Railway/Render/Fly.io
- CI/CD pipeline with GitHub Actions
- Monitoring and logging with Sentry
- Performance metrics dashboard

**User Interface**
- Interactive Streamlit dashboard
- Race weekend live predictions
- Historical performance comparison tool
- Driver/team analytics visualization

---

## 9. Technical Details

### Data Quality

**Training Data (2022-2024)**
- 1,353 qualifying sessions
- Zero missing values after imputation
- Time-based train/test split (no data leakage)

**Feature Dataset (2022-2025)**
- 810,305 driver-sessions
- 42 engineered features
- 4,217 sessions with position data

**Missing Data Handling**
- Circuit features: 27.5% missing (new circuits/drivers)
- Recent form: 0% missing (always computable)
- Weather features: 1.6% missing (interpolated)

### Model Performance

**Baseline Random Forest**
- Mean Absolute Error: 3.17 positions
- Root Mean Squared Error: 3.93 positions
- R-squared: 0.525
- 42% of predictions within ±2 positions
- 67% of predictions within ±3 positions

**What this means:**
- On average, predictions are off by about 3 positions
- Stronger performance for top teams (more consistent)
- Midfield hardest to predict (close competition)
- Weather conditions increase prediction error

### Reproducibility

**Fixed Elements**
- Random seed: 42 for all models
- FastF1 cache: `data/.fastf1_cache/`
- Deterministic pipeline (same input = same output)

**Version Control**
- All notebooks include requirements
- Model versioning in metadata
- Training date recorded

### Known Limitations

**Current Model**
- Doesn't account for car development during season
- Limited wet weather training data
- No tire strategy modeling
- Sprint weekend format not fully validated

**Data Sources**
- FastF1 API has occasional missing sessions
- Weather data limited to basic metrics
- No team radio or strategy information

**Predictions**
- Historical performance assumes consistent regulations
- 2026 regulation changes will require new features
- No real-time track condition updates
- Doesn't predict mechanical failures or penalties

---

## 10. Installation & Setup

### System Requirements
- Python 3.11+
- 8GB RAM minimum (16GB recommended for full pipeline)
- 10GB free disk space (for FastF1 cache)
- Internet connection (for FastF1 API)

### Installation

Clone and setup:
```bash
git clone https://github.com/tomasz-solis/formula1.git
cd formula1
python -m venv f1env
source f1env/bin/activate  # Windows: f1env\Scripts\activate
pip install -r requirements.txt
```

### Configuration

The pipeline uses sensible defaults, but you can customize:

**Historical features** (in `main.py`):
```python
compute_historical_features(
    lookback_years=3,      # Years of circuit history
    form_window=5,         # Races for recent form
    rain_threshold=0.1     # mm/h for "wet" classification
)
```

**Model training** (in notebook 04):
```python
RandomForestRegressor(
    n_estimators=200,      # Number of trees
    max_depth=15,          # Tree depth
    min_samples_split=10   # Min samples to split node
)
```

### Running the Pipeline

**Full pipeline** (takes ~15 minutes):
```bash
python main.py --from 2022 --to 2025
```

**Quick profile update** (skip features):
```bash
python main.py --from 2025 --to 2025 --no-features
```

**Single circuit** (for testing):
```bash
python main.py --from 2024 --to 2024 --gp "Monaco" --no-features
```

### Troubleshooting

**XGBoost won't load on macOS:**
```bash
brew install libomp
pip uninstall xgboost
pip install xgboost --no-cache-dir
```

**Out of memory during feature computation:**
- Close other applications
- Use `--no-features` flag
- Process fewer years at once

**FastF1 API timeouts:**
- Check internet connection
- Retry (pipeline is idempotent)
- Clear cache if corrupted: `rm -rf data/.fastf1_cache`

**Notebook file too large for Git:**
```bash
# Clear outputs before committing
jupyter nbconvert --clear-output --inplace EDA/*.ipynb
git add EDA/*.ipynb
```

---

## 11. Testing

Run unit tests:
```bash
# All tests
pytest tests/test_feature_engineering.py -v

# With coverage
pytest tests/ --cov=helpers --cov-report=html
```

### Get Help
```bash
python helpers/feature_engineering.py --help
```

---

## 12. Resources

- **FastF1 Documentation:** https://docs.fastf1.dev/
- **F1 Technical Regulations:** https://www.fia.com/regulation/category/110
- **Ergast API (historical data):** http://ergast.com/mrd/

---

## 13. Contributing

This is a learning project, but suggestions welcome! Areas for improvement:
- Better weather feature engineering
- Tire strategy modeling
- Real-time prediction during race weekends
- Interactive Streamlit dashboard

---

## 14. License

MIT License - feel free to learn from and build upon this work.

---

## 15. Acknowledgements

- [Mirco Bartolozzi](https://www.linkedin.com/in/mirco-bartolozzi/) — Formula Data Analysis inspiration
- **FastF1** — telemetry and timing data
- **OpenF1** — alternative data source
- The broader F1 data and fan community ❤️

---

## 16. Contact

For help customizing or extending this project:

- [tomasz.solis@gmail.com](mailto\:tomasz.solis@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/tomaszsolis/)

---

**Last updated:** November 13, 2025  
**Status:** REST API complete and ready for deployment  
**Current model:** Random Forest (MAE: 3.17 positions, R²: 0.525)