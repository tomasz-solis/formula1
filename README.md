# 🏎️ Formula 1 Qualifying Outcome Prediction

A machine learning project to predict F1 qualifying outcomes using practice session telemetry, circuit characteristics, and historical performance data.

**Current Status:** ✅ Classification models deployed - Q3: 78.8% accuracy, Top 3: 89.1% accuracy, Round: 70% accuracy

---

## 🎯 Project Goal

Predict qualifying outcomes based on:
- Practice session performance (FP1/FP2/FP3)
- Circuit characteristics (corners, altitude, track layout)
- Weather conditions
- Sprint weekend handling
- Historical driver/team performance

**Why classification over regression:** After discovering that exact position prediction (regression) couldn't beat a naive baseline (MAE 3.78 vs 3.60 baseline), I pivoted to classification tasks that are more tractable and practically useful.

---

## 📊 Key Results

### Classification Models (Production Ready ✅)

| Model | Task | Baseline | Achieved | Improvement | Status |
|-------|------|----------|----------|-------------|--------|
| **Q3 Qualification** | Will driver make top 10? | 50.0% | **78.8%** | **+28.8 pts** | ✅ Production |
| **Top 3 Finish** | Will driver podium in quali? | 15.0% | **89.1%** | **+74.1 pts** | ✅ Production |
| **Qualifying Round** | Which round will they reach? | 33.3% | **70.0%** | **+37 pts** | ✅ Production |

### Why Classification Worked

**Problem with Regression:**
- Predicting exact positions (P1-P20) achieved MAE 3.78
- Worse than naive baseline (MAE 3.60)
- Single feature (`recent_avg_position`) dominated with 91% importance
- Model essentially copied "past = future" without learning patterns

**Success with Classification:**
- Binary targets (Q3 or not) reduce variance
- Multi-feature learning: weather-adjusted metrics now 42% importance
- More actionable predictions ("Will Verstappen make Q3?" vs "Position 3.2")
- Clear evaluation metrics (accuracy vs ambiguous MAE)

### Model Performance Details

**Q3 Qualification (Top 10)**
- Accuracy: 78.8% | Precision: 81.3% | Recall: 75.2% | ROC AUC: 87.4%
- Confusion Matrix (Test): TN=369, FP=78, FN=112, TP=339
- Top Features: `dry_avg_position` (24.5%), `wet_avg_position` (18.0%), `team_recent_avg` (17.2%)

**Top 3 Finish**
- Accuracy: 89.1% | Precision: 72.3% | Recall: 44.4% | ROC AUC: 92.1%
- Very conservative: only predicts "Top 3" when highly confident
- Confusion Matrix (Test): TN=740, FP=23, FN=75, TP=60

**Qualifying Round (Multi-Class)**
- Overall Accuracy: 70.0%
- Predicts which round driver reaches (Q1 eliminated / Q2 eliminated / Q3 made)
- Demonstrates multi-class classification capability

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
│   └── features/                     # ML-ready features (Parquet)
├── models/
│   ├── q3_classifier.pkl             # Q3 binary classifier (78.8%)
│   ├── top3_classifier.pkl           # Top 3 binary classifier (89.1%)
│   ├── q2_classifier.pkl             # Round multi-class classifier (70%)
│   ├── classification_metadata.json  # Model metrics and features
│   └── feature_importance_*.csv      # Feature rankings
├── EDA/
│   ├── 02_feature_exploration.ipynb  # EDA with visualizations
│   ├── 03_feature_importance.ipynb   # Feature analysis
│   ├── 04_quali_prediction_model.ipynb # Regression baseline
│   └── 06_classification_models.ipynb  # ✨ Classification training
└── api/
    ├── __init__.py                   # Package initialization
    ├── config.py                     # API configuration (v2.0)
    ├── main.py                       # FastAPI app with 4 endpoints
    ├── models.py                     # Classification request/response schemas
    └── predictor.py                  # Prediction logic for 3 models
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

### 2. Build Raw Data (Optional)
```bash
# Extract telemetry and results for 2022-2025
python main.py --from 2022 --to 2025
```

### 3. Run Classification Notebook
```bash
cd EDA
jupyter notebook 06_classification_models.ipynb
# Run all cells to train models and generate visualizations
```

### 4. Start API
```bash
cd api
uvicorn main:app --reload
# Visit: http://127.0.0.1:8000/docs
```

---

## 📊 Dataset

### Current Dataset Stats
- **1,778 qualifying sessions** (2022-2025 seasons)
- **47 features** extracted from telemetry, circuits, and historical performance
- **Target variables:** Q3 qualification (binary), Top 3 finish (binary), Qualifying round (multi-class)
- **Train/Test split:** Temporal (2022-2023 train, 2024-2025 test)

### Data Sources
- **Driver telemetry:** FastF1 API (throttle, braking, DRS, tire degradation)
- **Circuit profiles:** Track layout, corners, speed characteristics
- **Qualifying results:** Official FIA classification data
- **Historical features:** Circuit performance, recent form, weather-adjusted metrics

### Feature Categories

**Driver Performance (13 features):**
- `best_throttle_ratio` - Peak throttle usage across practice
- `avg_throttle_ratio` - Average throttle consistency
- `best_brake_max_g` - Peak braking force
- `fp3_throttle_ratio` - FP3 performance
- `sprint_quali_throttle` - Sprint qualifying data
- Tire degradation, DRS activations, braking intensity

**Historical Performance (12 features):**
- `circuit_avg_position` - Average qualifying position at circuit (3-year lookback)
- `circuit_best_position` - Best finish at circuit
- `recent_avg_position` - Rolling 5-race average
- `form_trend` - Performance momentum (negative = improving)
- `wet_avg_position` - Average position in wet conditions
- `dry_avg_position` - Average position in dry conditions
- `wet_dry_delta` - Wet vs dry performance gap
- `team_circuit_avg_position` - Team historical performance
- `team_momentum` - Team development trajectory
- `team_recent_avg` - Team's recent form

**Circuit Characteristics (10 features):**
- `slow_corner_pct` - Percentage of slow-speed corners
- `medium_corner_pct` - Percentage of medium-speed corners
- `fast_corner_pct` - Percentage of high-speed corners
- `total_corners` - Total corner count
- `chicanes` - Chicane count
- `avg_speed_circuit` - Average track speed
- `top_speed_circuit` - Maximum achievable speed

**Weather & Conditions (3 features):**
- `rain_in_practice` - Rainfall detected
- `avg_track_temp` - Mean track temperature
- `track_temp_std` - Temperature variation

**Weekend Format (3 features):**
- `is_sprint_weekend` - Sprint vs normal weekend flag
- `has_sprint_quali_data` - Sprint qualifying data available
- `sessions_available` - Which practice sessions occurred

---

## 🤖 Machine Learning Models

### Classification Approach

**Binary Classification (2 models):**
1. **Q3 Qualification:** Predicts if driver makes top 10 (Q3 shootout)
2. **Top 3 Finish:** Predicts if driver qualifies in podium positions

**Multi-Class Classification (1 model):**
3. **Qualifying Round:** Predicts which round driver reaches (Q1/Q2/Q3)

### Model Architecture

**Algorithm:** Random Forest Classifier
```python
RandomForestClassifier(
    n_estimators=200,      # 200 trees for stable predictions
    max_depth=10,          # Prevents overfitting
    min_samples_split=20,  # Requires 20 samples to split
    min_samples_leaf=10,   # Minimum leaf size
    random_state=42        # Reproducibility
)
```

**Why Random Forest:**
- Handles non-linear relationships
- Provides feature importance rankings
- Robust to missing data (~40% in historical features)
- Probability calibration for confidence scores

### Feature Importance

**Q3 Model Top Features:**
```
1. dry_avg_position      24.5%  ← Driver's average in dry conditions
2. wet_avg_position      18.0%  ← Driver's average in wet conditions
3. team_recent_avg       17.2%  ← Team's recent form
4. recent_avg_position    3.7%  ← Driver's recent form
5. team_momentum          3.1%  ← Team development trajectory
```

**Key Insight:** Weather-adjusted metrics (`wet_avg` + `dry_avg`) combine for 42% of model importance. This is a dramatic improvement from regression, where these same features had 0.0008 importance and were effectively ignored.

### Training & Validation

**Data Split:**
- **Train:** 2022-2023 seasons (880 qualifying sessions)
- **Test:** 2024-2025 seasons (898 qualifying sessions)
- **Temporal split:** No data leakage (strict chronological order)

**Handling Imbalance:**
- Q3: 50/50 split (balanced)
- Top 3: 15% positive (imbalanced, but acceptable for binary)
- Round: 25%/25%/50% split (Q1/Q2/Q3)

**Missing Data:**
- 40% missing in historical features (new drivers, new circuits)
- Median imputation strategy
- Model robust to missing data due to ensemble approach

---

## 🌐 REST API

### API v2.0 - Classification Endpoints

**Base URL:** `http://localhost:8000`

### Endpoints

#### 1. Q3 Prediction
```bash
POST /predict/q3
```

**Request:**
```json
{
  "driver": "VER",
  "circuit": "Monza",
  "year": 2025,
  "avg_rainfall": 0.0,
  "avg_track_temp": 35.0
}
```

**Response:**
```json
{
  "will_make_q3": true,
  "probability": 0.94,
  "confidence": "high",
  "model_accuracy": 0.788,
  "features_used": {
    "dry_avg_position": 1.8,
    "wet_avg_position": 2.1,
    "team_recent_avg": 1.5
  }
}
```

#### 2. Top 3 Prediction
```bash
POST /predict/top3
```

**Response:**
```json
{
  "will_make_top3": true,
  "probability": 0.78,
  "confidence": "high",
  "model_accuracy": 0.891
}
```

#### 3. Round Prediction
```bash
POST /predict/round
```

**Response:**
```json
{
  "predicted_round": "Q3",
  "probabilities": {
    "Q1": 0.02,
    "Q2": 0.08,
    "Q3": 0.90
  },
  "confidence": "high",
  "model_accuracy": 0.70
}
```

#### 4. Combined Predictions
```bash
POST /predict/all
```

Returns all three predictions in one call.

### Other Endpoints

- `GET /` - API information
- `GET /health` - Health check (model status)
- `GET /model/info` - Model metadata and metrics
- `GET /drivers` - List available drivers
- `GET /circuits` - List available circuits
- `GET /docs` - Interactive API documentation (Swagger)

### Running the API

```bash
cd api
uvicorn main:app --reload

# Visit interactive docs:
# http://127.0.0.1:8000/docs
```

### API Features

- ✅ **Probability Scores:** Every prediction includes confidence probability
- ✅ **Confidence Levels:** Automatic high/medium/low classification
- ✅ **Historical Feature Lookup:** Automatic retrieval from 2022-2025 data
- ✅ **Manual Overrides:** Provide weather/temperature if known
- ✅ **Fallback Strategy:** Uses median values for new drivers
- ✅ **Model Metadata:** Exposes accuracy and feature importance

---

## 📈 Visualizations

The classification notebook generates publication-quality visualizations:

### 1. Model Performance Summary
- Baseline vs achieved accuracy comparison
- Improvement over baseline for each model
- Top 5 features for Q3 model

### 2. Confusion Matrices
- Q3 classification: 369 TN, 78 FP, 112 FN, 339 TP
- Top 3 classification: Very conservative (23 FP vs 740 TN)
- Round classification: Shows Q1/Q2/Q3 prediction patterns

### 3. Feature Importance
- Q3 model: Weather-adjusted metrics dominate
- Top 3 model: Similar pattern but higher importance on `dry_avg`
- Comparison shows consistent patterns across models

### 4. ROC Curves
- Q3: AUC 0.874 (excellent discrimination)
- Top 3: AUC 0.921 (outstanding discrimination)
- Both significantly above random classifier

### 5. Class Balance
- Q3: 50/50 (perfectly balanced)
- Top 3: 85/15 (imbalanced but manageable)
- Round: 25/25/50 (Q1/Q2/Q3)

---

## 🔄 Model Evolution

### Regression → Classification Pivot

**Initial Approach (Failed):**
- **Task:** Predict exact qualifying positions (1-20)
- **Model:** Random Forest Regressor
- **Result:** MAE 3.78 (worse than baseline 3.60)
- **Problem:** Single feature dominated (91% importance on `recent_avg_position`)
- **Conclusion:** Model just learned "past = future" without patterns

**Pivot to Classification (Success):**
- **Task:** Predict binary/multi-class outcomes
- **Models:** Random Forest Classifiers (Q3, Top 3, Round)
- **Result:** 70-89% accuracy (29-74 points above baseline)
- **Benefit:** Multi-feature learning, weather metrics emerged (42% importance)

### Key Learnings

1. **Problem Formulation Matters More Than Model Complexity**
   - Same features, different target → dramatically different results
   - Classification enabled better feature utilization

2. **Baselines Are Critical**
   - Without naive baseline (MAE 3.60), would have thought MAE 3.78 was good
   - Baseline comparison revealed regression was useless

3. **Feature Importance Changes With Target**
   - Regression: `recent_avg` = 91% importance
   - Classification: Weather metrics = 42% importance
   - Same features, different patterns learned

4. **Missing Data Isn't Always a Dealbreaker**
   - 40% missing in historical features
   - Still achieved 79% accuracy with median imputation
   - Clear target helps model handle sparse data

---

## 📊 Project Status

### ✅ Completed

**Data Pipeline**
- FastF1 telemetry extraction for 2022-2025 seasons
- Circuit profile database (24 tracks)
- Historical feature computation (3-year lookback)
- Data leakage eliminated (temporal split validation)
- Qualifying results SSOT (single source of truth)

**Modeling**
- 3 classification models trained and validated
- Feature importance analysis complete
- Confusion matrices and ROC curves generated
- Model artifacts saved with metadata
- Production-ready predictions with confidence scores

**Deployment**
- REST API with FastAPI (v2.0)
- 4 prediction endpoints (Q3, Top 3, Round, Combined)
- Interactive API documentation (Swagger)
- Automatic historical feature lookup
- Health monitoring and model info endpoints

### 🚧 In Progress

**Model Improvements**
- Sprint weekend validation (limited data currently)
- Circuit-specific model fine-tuning
- Teammate comparison features (driver skill isolation)
- Ensemble methods (stacking classifiers)

**API Enhancements**
- Docker containerization
- Batch prediction endpoint
- Real-time prediction during race weekends

### 📋 Planned

**Advanced Modeling**
- Quali position within Q3 (P1-P10 among qualifiers)
- Race outcome classification (podium, points, DNF)
- Strategy optimization (tire choice, fuel load)
- Driver skill metrics (overtaking, defending, consistency)

**Data Sources**
- Preseason testing data (2026 regulations)
- Real-time weather API integration
- Team radio sentiment analysis
- Tire compound performance database

**User Interface**
- Streamlit dashboard for race weekend predictions
- Historical comparison tool (driver vs driver)
- Team performance analytics
- Interactive visualizations

---

## 🧪 Data Quality & Integrity

### Data Leakage Elimination (Critical Fix)

**Problem Identified (Nov 2025):**
- Historical features contained test set data
- `circuit_avg_position` showed perfect 1.000 correlation with actual positions
- Model achieved unrealistic MAE < 1.0 by essentially copying answers

**Detection:**
- Diagnostic script `check_leakage.py` revealed 95-99% contamination
- Circuit history included current year when predicting current year
- Recent form used earlier races from same test year

**Resolution:**
- Strict temporal split: train on 2022-2023, test on 2024-2025
- Circuit history excludes current year
- Recent form computed only from previous years
- Post-fix correlation: `circuit_avg_position` = 0.552 (realistic)

**Impact:**
- Honest classification accuracies: 70-89% (from fraudulent near-perfect)
- Test predictions now genuinely predictive
- All features validated for temporal separation

### Missing Data Analysis

**Historical Features (Expected Missingness):**
```
circuit_avg_position:    27.5%  ← New circuits, new drivers
circuit_best_position:   27.5%  ← Same as above
recent_avg_position:      0.0%  ← Always computable
form_trend:               0.0%  ← Always computable
wet_dry_delta:            1.6%  ← Limited wet sessions
team_circuit_avg:        10.1%  ← New teams at circuits
team_momentum:            1.2%  ← Mostly complete
```

**Handling Strategy:**
- Median imputation for missing numerical features
- Preserves feature distribution
- Random Forest robust to imputation artifacts
- New drivers fallback to population medians

### Train/Test Split

**Temporal Split (No Leakage):**
- **Train:** 2022-2023 seasons (880 qualifying sessions)
- **Test:** 2024-2025 seasons (898 qualifying sessions)
- **Validation:** Chronological ordering enforced
- **No shuffling:** Maintains temporal structure

**Why Temporal:**
- Simulates real prediction scenario (predict future from past)
- Prevents data leakage (can't use future to predict past)
- Tests generalization to new season dynamics

---

## 🛠️ Technical Implementation

### Classification Pipeline

1. **Load Data:** Historical features + qualifying results (2022-2025)
2. **Filter:** Keep only rows with qualifying position data
3. **Create Targets:**
   - Top 3: Binary (position ≤ 3)
   - Q3: Binary (position ≤ 10)
   - Round: Multi-class (Q1: 16-20, Q2: 11-15, Q3: 1-10) - called Q2 classifier
4. **Feature Selection:** Remove target-related columns
5. **Train/Test Split:** Temporal (2022-2023 / 2024-2025)
6. **Imputation:** Median strategy for missing values
7. **Train Models:** Random Forest Classifiers (3 models)
8. **Evaluate:** Accuracy, precision, recall, F1, ROC AUC
9. **Save:** Model artifacts, feature importance, metadata

### Code Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Create Q3 target
df['made_q3'] = (df['qualifying_position'] <= 10).astype(int)

# Temporal split
train = df[df['year'] <= 2023]
test = df[df['year'] >= 2024]

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(train[features])
X_test = imputer.transform(test[features])

# Train classifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    random_state=42
)
model.fit(X_train, train['made_q3'])

# Predict with probabilities
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
```

### API Implementation

**Prediction Flow:**
1. Receive request: driver, circuit, year, weather (optional)
2. Lookup historical features from database
3. Override with manual weather if provided
4. Impute missing features with medians
5. Generate feature vector (47 features)
6. Predict with all 3 models
7. Return probabilities + confidence levels

**Confidence Levels:**
```python
def get_confidence(probability):
    if probability >= 0.75:
        return "high"
    elif probability >= 0.55:
        return "medium"
    else:
        return "low"
```

---

## 📚 Resources & References

### Documentation
- **FastF1:** https://docs.fastf1.dev/
- **F1 Regulations:** https://www.fia.com/regulation/category/110
- **Ergast API:** http://ergast.com/mrd/

### Learning Resources
- **"Designing Machine Learning Systems"** by Chip Huyen (referenced for data leakage detection)
- **scikit-learn Docs:** https://scikit-learn.org/

### Inspiration
- [Mirco Bartolozzi](https://www.linkedin.com/in/mirco-bartolozzi/) - Formula Data Analysis
- F1 data science community on Reddit, Kaggle

---

## 🤝 Contributing

This is a learning project demonstrating end-to-end ML workflow. Suggestions welcome!

**Areas for Improvement:**
- Sprint weekend model validation (need more data)
- Circuit-specific features (overtaking difficulty, tire wear)
- Driver skill metrics (qualifying vs race pace)
- Real-time predictions during race weekends

---

## 📄 License

MIT License - feel free to learn from and build upon this work.

---

## 🙏 Acknowledgements

- [Mirco Bartolozzi](https://www.linkedin.com/in/mirco-bartolozzi/) - Formula Data Analysis inspiration
- **FastF1** - Telemetry and timing data
- **F1 Community** - Inspiration and data science discussions

---

## 📧 Contact

**Tomasz Solis**
- Email: [tomasz.solis@gmail.com](mailto:tomasz.solis@gmail.com)
- LinkedIn: [linkedin.com/in/tomaszsolis](https://www.linkedin.com/in/tomaszsolis/)
- GitHub: [github.com/tomasz-solis](https://github.com/tomasz-solis)

---

**Last Updated:** November 18, 2025  
**Status:** Classification models production-ready  
**Current Models:**
- Top 3 Finish: 89.1% accuracy (vs 15% baseline)
- Q3 Qualification: 78.8% accuracy (vs 50% baseline)
- Qualifying Round: 70.0% accuracy (vs 33% baseline)
