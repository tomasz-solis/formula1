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
├── main.py                     # Data pipeline orchestration
├── helpers/
│   ├── feature_engineering.py  # ML feature extraction
│   ├── historical_features.py  # Historical performance
│   ├── general_utils.py        # Session loading, caching
│   ├── driver_utils.py         # Driver telemetry features
│   ├── circuit_utils.py        # Circuit profile extraction
│   └── prediction.py           # SSOT classification exports
├── data/
│   ├── driver/                 # Driver session profiles (CSV)
│   ├── circuit/                # Circuit profiles (CSV)
│   ├── driver_timing/          # Detailed lap telemetry (Parquet)
│   ├── predictions/ssot/       # Official qualifying results (CSV)
│   └── processed/              # ML-ready features (CSV)
├── features/
│   └── ml_features_2022_2025.parquet # Historical performance features
└── EDA/
    ├── 01_general.ipynb           # Track clustering analysis
    └── 00_wip.ipynb               # Experimentation notebook
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

## 7. Development Status

### ✅ Completed
- [x] Data pipeline (driver/circuit/timing profiles)
- [x] Historical feature engineering
- [x] Exploratory data analysis
- [x] Feature importance analysis
- [x] Baseline ML models (Random Forest, XGBoost, LightGBM)
- [x] Model evaluation and error analysis

### 🔄 In Progress
- [ ] Hyperparameter tuning
- [ ] Model ensembling
- [ ] REST API for predictions
- [ ] Testing data parser (2026 preseason analysis)

### 📋 Planned
- [ ] Deep learning models
- [ ] Race result prediction (extends beyond qualifying)
- [ ] Real-time prediction during race weekends
- [ ] Interactive dashboard
- [ ] Docker deployment

---

## 8. Testing

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

## 9. Technical Notes

### Data Quality
- **Missing circuit data:** 119 rows (2023 Abu Dhabi) imputed from 2022/2024
- **Missing altitude:** Dropped feature (low variance)
- **Dropped rows:** 68 drivers without qualifying result (DNS/DSQ)

### Reproducibility
- FastF1 cache: `data/.fastf1_cache/`
- Deterministic feature engineering (no randomness)
- Idempotent pipeline (safe to re-run)

### Known Limitations
- No historical features yet (coming in Phase 2)
- Sprint qualifying impact not fully validated
- Weather features basic (only rain detection)
- No tire strategy modeling

---

## 10. Resources

- **FastF1 Documentation:** https://docs.fastf1.dev/
- **F1 Technical Regulations:** https://www.fia.com/regulation/category/110
- **Ergast API (historical data):** http://ergast.com/mrd/

---

## 11. Contributing

This is a learning project, but suggestions welcome! Areas for improvement:
- Better weather feature engineering
- Tire strategy modeling
- Real-time prediction during race weekends
- Interactive Streamlit dashboard

---

## 12. License

MIT License - feel free to learn from and build upon this work.

---

## 13. Acknowledgements

- [Mirco Bartolozzi](https://www.linkedin.com/in/mirco-bartolozzi/) — Formula Data Analysis inspiration
- **FastF1** — telemetry and timing data
- **OpenF1** — alternative data source
- The broader F1 data and fan community ❤️

---

## 14. Contact

For help customizing or extending this project:

- [tomasz.solis@gmail.com](mailto\:tomasz.solis@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/tomaszsolis/)

---

**Last updated: November 12, 2025**
**Status:** Baseline ML models complete.