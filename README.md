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

## 5. Next Steps

### Phase 3: Baseline Models
Establish performance targets:
- Baseline 1: Always predict median (10.5)
- Baseline 2: Predict driver's last 3 race average
- **Target to beat:** MAE < 3.5 positions

### Phase 4: Machine Learning
Train predictive models:
- Linear Regression (interpretable)
- Random Forest (handles non-linearity)
- Ridge Regression (regularization)
- **Goal:** MAE < 2.8 positions (20% improvement over baseline)

### Phase 5: Model Evaluation
- Error analysis by track type, driver, team
- Feature importance visualization
- Failure mode investigation

---

## 6. Sample Data

### Verstappen's 2024 Performance
```python
import pandas as pd
df = pd.read_csv('data/processed/qualifying_features.csv')
ver_2024 = df[(df['driver'] == 'VER') & (df['year'] == 2024)]

print(ver_2024[['event', 'qualifying_position', 'best_throttle_ratio', 'is_sprint_weekend']])
```

**Best qualifying:** Australia (P1, throttle 0.787)  
**Worst qualifying:** São Paulo (P12, sprint weekend)

## 7. Feature Engineering - Command-Line Usage

### Basic Usage
```bash
# Generate features for 2022-2024 (default)
python helpers/feature_engineering.py
```

### Advanced Options
```bash
# Custom years
python helpers/feature_engineering.py --years 2023 2024

# Custom output path
python helpers/feature_engineering.py --output data/custom/features.csv

# Verbose logging
python helpers/feature_engineering.py --verbose

# Show detailed summary
python helpers/feature_engineering.py --show-summary

# Quiet mode (only errors)
python helpers/feature_engineering.py --quiet
```

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

**Last updated: November 11, 2025**
**Status:** Feature engineering complete, historical performance added, baseline modeling next
