# Formula 1 Qualifying Predictor

## 2026 season compatibility

This project is built for F1 seasons 2022 to 2025, on the technical and sporting regulations from that era.

For 2026, major regulation changes (new power units, aerodynamics and more) need real model retraining and feature work. This codebase stays as a reference for 2022 to 2025 analysis and as a baseline for 2027 or later, once car development stabilizes. For 2026-onward compatibility, see [F1-2026-predictor](https://github.com/tomasz-solis/f1-2026-predictions), which accounts for the new regulations.

This project is fully functional for historical F1 analysis and works as a complete example of production ML for motorsport analytics under stable technical regulations.

Predicting F1 qualifying outcomes with machine learning. This started as a way to learn ML properly. Predicting exact grid positions turned out to be basically impossible (chaos theory is real), but predicting who makes Q3 or the podium works well.

Current models: Q3 74% accurate, Top 3 89% accurate, Round 75% accurate.

The system learns from each race weekend automatically, with no manual retraining needed. It handles rookies properly now (Antonelli at Mercedes isn't the same as a random rookie at Williams) and has robust fallbacks for missing data.

## Why this project

Instead of Kaggle competitions, I wanted to build something real: messy data (missing telemetry, DNS, wet and dry chaos), production concerns (APIs, versioning, monitoring), and real failures and pivots (regression first, which failed badly).

F1 fits this well. It has many variables (weather, tires, track evolution, driver form), genuine unpredictability (you can't just memorize "Verstappen always wins"), a small dataset (about 20 races a year forces careful feature choices), and a clear evaluation question: did they make Q3, yes or no.

## From failure to working models

### What didn't work: regression

The first attempt predicted exact qualifying positions (P1, P2, P3 and so on). It was a disaster: the model's MAE was 3.78 positions, against a naive baseline (just use practice times) of 3.60 positions. The fancy ML model was worse than doing nothing.

Qualifying is chaos. Verstappen crashes in Q1 and goes from P1 to P20. A red flag at the wrong time puts a random driver in P3. Rain in Q3 shuffles everything. The model learned "past position predicts future position" and gave up trying to find real patterns.

### What works: classification

Switching to yes/no questions changed that: will this driver make Q3 (top 10), will they get pole, P2 or P3 (podium), and which round will they reach (Q1, Q2 or Q3).

Results improved immediately: Q3 prediction is 74% accurate (vs 50% by guessing), and top 3 prediction is 89% accurate (vs a 15% baseline). Predicting patterns instead of the unpredictable was the fix.

## What makes it learn

### Team-based rookie handling

Early versions treated all rookies the same, filling missing data with grid averages. That's wrong: Antonelli at Mercedes has far more potential than a pay driver at a backmarker team.

Now each team gets a performance baseline (Red Bull around P2, Williams around P17, and so on). Rookies inherit their team's baseline plus an uncertainty penalty, and as they race more, the prediction shifts gradually toward their own data.

Example progression for Antonelli at Mercedes (2025):

| Point | Basis | Prediction |
|---|---|---|
| Race 1 | Team baseline (P6) plus rookie penalty (+1.5) | P8 |
| Race 5 | 60% team baseline, 40% his own data | P7 |
| Race 15 | Fully his own data | Wherever he's actually been |

2025 has 4 rookies, about 20% of the grid. The model used to treat "no historical data" as "probably midfield". Now it knows Antonelli (Mercedes) isn't the same as Bortoleto (Sauber). The system tracks team performance separately, regenerates baselines after each race, and fills missing features using team context.

### Robust missing data handling

The November 28, 2025 update fixed production bugs around missing data: predicting 2025 races before they happen (no historical data yet), rookies with zero F1 experience, drivers at new circuits (Las Vegas 2023), and an API that crashed with "index out of bounds" errors.

The fix is a 5-level feature fallback:

1. Exact match: driver at this circuit this year.
2. Circuit history: driver at this circuit, any recent year.
3. Same-year average: driver's average at other circuits this year.
4. Previous year: driver's average from last season.
5. Population median: last resort, for rookies without team data.

The system no longer crashes on missing data, predictions use the most relevant data available, and quality degrades gracefully from best to acceptable rather than failing outright.

Example, predicting Verstappen at Las Vegas 2025 before the race: VER + Las Vegas + 2025 isn't found, VER + Las Vegas + 2024 is found, so the prediction uses 2024 Las Vegas performance.

Example, predicting rookie Antonelli at Monaco 2025: ANT + Monaco + 2025 isn't found (rookie), ANT + Monaco + any year isn't found (rookie), ANT + other 2025 circuits isn't found (hasn't raced yet), ANT + 2024 data isn't found (rookie), so it falls back to the Mercedes team baseline plus a rookie adjustment, giving a prediction around P7 (good team, rookie uncertainty).

### Recent form tracking

Looks at the last 5 races to catch momentum: if Lawson suddenly performs well at RB the model adjusts up, if Verstappen has 3 bad weekends the model accounts for it, and outliers (pit lane starts, DNQs) are filtered out to avoid noise.

### Team change detection

Drivers who change teams (Hamilton to Ferrari, Sainz to Williams) are tricky. The model uses 70% of the driver's historical skill and blends in 30% of the new team's baseline, then caps predictions at realistic levels for the new team.

Example: Sainz at Williams in 2025. His Ferrari history says P6 to P8 (fast driver, fast car). The Williams baseline says P14 to P16 (slower car). The blended prediction is P10 to P12 (fast driver, slower car), since predicting P6 at Williams would be unrealistic.

### Smart missing data handling

Real F1 data is messy: team names change (AlphaTauri to RB, Alfa Romeo to Sauber), circuits get added mid-season (Las Vegas 2023), rookies have zero history, and returnees exist (Colapinto back to reserve, then back to a race seat). The system canonicalizes team names, fills missing values using context, and tracks data availability per feature, rather than treating a missing value as a random guess.

## The self-learning part (MLOps)

This was the main learning goal: a production ML system that improves automatically.

After each race weekend: run `python main.py --from 2022 --to 2025` to extract new data. The system detects the new race, regenerates team baselines with the new data, retrains all 3 models with updated features, compares the new models against the current ones, deploys the new version automatically if it's better, and the API picks up the new models within 60 seconds, with no restart.

What gets tracked:

```text
models/
├── v20251122_125637/  Version from Nov 22
│   ├── q3_classifier.pkl
│   ├── metadata.json (accuracies, training date)
├── v20251128_161325/  Version from Nov 28 (critical fixes)
│   ├── q3_classifier.pkl
│   ├── metadata.json
├── team_baselines.json      Regenerates after each race
├── training_history.json    Shows improvement over time
└── active_version.txt       API uses this
```

Most ML tutorial projects train a model once, save it to disk, and never update it, so accuracy slowly degrades. This one does continuous learning from new data, automatic deployment of improvements, version control with rollback, performance monitoring and dynamic feature generation as team baselines update. It's not "I trained a model", it's "I built a self-improving system".

## Current performance

| Model | Predicts | Baseline | Actual | Status |
|-------|-----------------|----------|--------|--------|
| Q3 | Will the driver make top 10? | 50% | 74% | Production |
| Top 3 | Will the driver podium? | 15% | 89% | Production |
| Round | Which round (Q1/Q2/Q3)? | 33% | 75% | Production |

Recent improvements (November 28, 2025): fixed critical missing-data crashes (API 500 errors eliminated), added the 5-level feature fallback, 2025 predictions now work including future races, rookie predictions improved with team context, and team-change handling is more realistic with capped blending.

### What "production" means here

A REST API (FastAPI with Swagger docs at `/docs`), dynamic model loading (the API reloads models when a new version deploys), health checks (`/health` shows model status), version tracking (know which model version served which prediction), graceful degradation (a missing feature gets an intelligent fallback instead of a crash), proper error handling instead of bare 500 errors, and robust predictions that handle rookies, team changes, future races and missing data.

## Project structure

```text
formula1/
├── main.py                    # Data pipeline (extracts from FastF1)
├── race_prediction.py         # Production prediction script
├── helpers/
│   ├── auto_retrain.py       # Auto-retraining after new races
│   ├── feature_engineering.py # Feature generation
│   ├── historical_features.py # Time-series features
│   ├── team_priors.py        # Team baseline computation
│   ├── team_name_mapping.py  # Handle team rebranding
│   └── validation.py         # Data quality checks
├── data/
│   ├── features/
│   │   └── ml_features.parquet  # All training data
│   └── predictions/ssot/        # Official results
├── models/
│   ├── q3_classifier.pkl        # Active models
│   ├── top3_classifier.pkl
│   ├── round_classifier.pkl
│   ├── team_baselines.json      # Team performance baselines
│   ├── v20251128_161325/        # Latest version (Nov 28)
│   └── training_history.json
└── api/
    ├── main.py                  # FastAPI server
    ├── predictor.py             # Prediction logic (robust fallback)
    └── dynamic_model_loader.py  # Auto-reload models
```

## Quick start

Install:

```bash
git clone https://github.com/tomasz-solis/formula1.git
cd formula1
python -m venv f1env
source f1env/bin/activate
pip install -r requirements.txt
```

Extract data and train models:

```bash
# Get all F1 data from 2022-2025 and train initial models
python main.py --from 2022 --to 2025

# What happens:
# 1. Downloads telemetry from FastF1
# 2. Engineers 48 features
# 3. Computes team performance baselines
# 4. Trains 3 classification models
# 5. Saves to models/ directory
```

Start the API:

```bash
cd api
uvicorn main:app --reload

# Visit http://127.0.0.1:8000 for the landing page
# Or http://127.0.0.1:8000/docs for Swagger UI
```

Make a prediction through the API:

```bash
# Will Verstappen make Q3 at Monza?
curl -X POST http://localhost:8000/predict/q3 \
  -H "Content-Type: application/json" \
  -d '{
    "driver": "VER",
    "circuit": "Monza",
    "year": 2025
  }'

# Response:
# {
#   "will_make_q3": true,
#   "probability": 0.95,
#   "confidence": "high"
# }
```

Make a prediction with the script: edit `race_prediction.py` to update `RACE_CONFIG` at the top (circuit, weather, sprint weekend status), then run `python race_prediction.py`. It outputs full grid predictions with probabilities, recent form adjustments, team change impacts and practice session integration where available, and saves a CSV like `Qatar_GP_2025_predictions.csv`.

Race configuration example:

```python
RACE_CONFIG = {
    "circuit": "Qatar Grand Prix",
    "year": 2025,
    "is_sprint_weekend": True,  # Sprint or normal

    "weather": {
        "avg_rainfall": 0.0,
        "avg_track_temp": 32.0,
        "avg_air_temp": 28.0
    }
}
```

To predict a new race, update these 5 values. The script handles sprint weekends (FP1 plus Sprint Quali), normal weekends (FP1, FP2, FP3), rookies (team baseline), team changes (blended predictions) and missing data (intelligent fallback).

## Features that actually matter

After trying more than 50 features, these are what the models use for Q3 prediction:

| Feature | Importance | Meaning |
|---|---:|---|
| `dry_avg_position` | 24% | How they normally qualify in the dry |
| `wet_avg_position` | 18% | How they qualify in the wet |
| `team_recent_avg` | 17% | The team's current form |
| `team_baseline_quali` | 8% | Team performance baseline |
| `recent_avg_position` | 4% | The driver's last 5 races |
| `circuit_avg_position` | 3% | History at this track |

Weather-adjusted features matter. The model learned that some drivers (Sainz, Alonso, Norris) are rain specialists who punch above their weight when it's wet, and that team context is crucial for drivers with limited history.

## Things I learned building this

### Data leakage is sneaky

The first version had 99% accuracy, which was suspicious. It turned out test-set data had leaked into the training features: "circuit average" included races from 2024 when predicting 2024, essentially giving the model the answers. Fixing it dropped accuracy to a realistic and honest 74%. Lesson: always check your train/test split isn't contaminated.

### Missing data needs context

The first approach filled missing data with the median. Simple, and wrong. In November 2025 the API crashed predicting 2025 races (future data doesn't exist yet), rookies got median-filled features (treating "no data" like "bad data"), and feature counts sometimes mismatched (2 features returned when the model expected 48).

The fix: the 5-level fallback hierarchy (exact match, circuit history, year average, previous year, median), context-aware imputation (rookies get their team's baseline, not the population median), and graceful degradation that always returns a valid feature vector. The result: zero crashes on missing data, working predictions for future races, and sensible rookie predictions (a Mercedes rookie isn't a Sauber rookie).

Lesson: understand why data is missing before deciding how to fill it. Different kinds of missing data need different strategies.

### Small datasets force you to be smart

F1 has about 20 races a year and 20 drivers, so about 400 data points a year. You can't just throw data at the problem. That meant careful feature engineering, blending historical baselines for new drivers, and handling missing data with judgment (a rookie at a new circuit needs a real answer, not a shrug). Lesson: feature engineering beats more data when you can't get more data.

### Classification beats regression for chaos

Predicting "P1 vs P2 vs P3" is too granular when strategy varies (fuel loads, tire choice), red flags shuffle everything, and one mistake in Q3 turns P1 into P10. "Will they make Q3?" averages out that noise. Lesson: match your problem's granularity to how predictable the data actually is.

### Team context matters more than expected

Predicting rookies from driver-level features alone failed. Antonelli at Mercedes was predicted at P15 (midfield) because "no data" defaulted to "assume average", but Mercedes isn't average. Adding team baselines improved rookie predictions significantly: the model learned an unknown Mercedes driver is very different from an unknown Williams driver. Lesson: domain knowledge beats generic ML tricks. F1 teams vary wildly in performance.

### Production means handling edge cases

A model that works on clean training data is easy to build. One that doesn't crash in production is hard. Edge cases fixed: future races (data doesn't exist yet), rookies mid-season (partial data), team changes (conflicting historical data), new circuits (no circuit history), missing practice data (weather, crashes), and numpy type serialization (JSON doesn't understand `numpy.bool_`). Lesson: production ML is 20% modeling and 80% handling edge cases gracefully.

## What's next

Short term: practice session predictions (FP1 to FP2 to FP3 progression), trying gradient boosting (XGBoost) against Random Forest, and feature engineering for tire compound effects and track evolution.

Medium term: Docker deployment, a monitoring dashboard, A/B testing new model versions, automated data quality alerts, and race predictions beyond qualifying.

Longer term, maybe: strategy optimization ("should Hamilton pit now?"), real-time predictions during practice sessions, and a web UI for non-technical users.

## Technical details

Stack: Python 3.13, the FastF1 API for telemetry, scikit-learn for models, FastAPI for serving, and Parquet for data storage. FastF1 is the only good F1 data source, scikit-learn keeps models simple and debuggable, FastAPI is fast with automatic docs, and Parquet is columnar storage, much faster than CSV.

Recent additions (November 28, 2025): the 5-level feature fallback hierarchy, numpy type conversion (fixes JSON serialization errors), the production race prediction script (handles sprint weekends), and improved team-change blending with realistic capping.

## Contact

Tomasz Solis · tomasz.solis@gmail.com · [LinkedIn](https://www.linkedin.com/in/tomaszsolis/) · [GitHub](https://github.com/tomasz-solis)

Last updated: November 28, 2025. Status: production-ready with robust missing data handling.

Current models: Top 3 finish 88.9% accuracy (vs 15% baseline), Q3 qualification 73.9% accuracy (vs 50% baseline), qualifying round 75.3% accuracy (vs 33% baseline), and a 5-level feature fallback that handles rookies, new circuits and team changes.
