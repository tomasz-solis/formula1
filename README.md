# Formula 1 Qualifying Predictor

Predicting F1 qualifying outcomes using machine learning. This started as a way to learn ML properly - turns out predicting exact grid positions is basically impossible (chaos theory is real!), but predicting who makes Q3 or podium? That works pretty well.

**Current Models:** Q3: 74% accurate | Top 3: 89% accurate | Round: 75% accurate

The system learns from each race weekend automatically - no manual retraining needed. Now with proper rookie handling (because Antonelli at Mercedes is not equal to random rookie at Williams) and robust missing data fallbacks.

---

## Why This Project Exists

Instead of just doing Kaggle competitions, I wanted to build something real:
- Real messy data (missing telemetry, DNS, wet/dry chaos)
- Real production concerns (APIs, versioning, monitoring)
- Real failures and pivots (tried regression first, failed spectacularly)

Turns out F1 is perfect for this because:
1. **Lots of variables**: weather, tires, track evolution, driver form
2. **Actual unpredictability**: You can't just memorize "Verstappen always wins"
3. **Small datasets**: Only ~20 races/year forces you to be smart about features
4. **Clear evaluation**: Did they make Q3? Yes or no - simple.


## The Journey: From Failure to Working Models

### What Didn't Work: Regression

First attempt was predicting exact qualifying positions (P1, P2, P3...). Complete disaster:
- **My model MAE: 3.78 positions**
- **Naive baseline (just use practice times): 3.60 positions**
- My fancy ML model was **worse than doing nothing**

Why? Because qualifying is chaos:
- Verstappen crashes in Q1 goes from P1 to P20
- Red flag at wrong time random driver in P3
- Rain in Q3 everything shuffles

The model basically learned "past position = future position" and gave up trying to predict actual patterns.

### What Works: Classification

Switched to yes/no questions instead:
- "Will this driver make Q3?" (top 10)
- "Will they get pole/P2/P3?" (podium)
- "Which round will they reach?" (Q1/Q2/Q3)

**Immediately better results:**
- Q3 prediction: 74% accurate (vs 50% if you just guess)
- Top 3 prediction: 89% accurate (vs 15% baseline)
- Stopped trying to predict the unpredictable, started predicting patterns


## What Makes It Actually Learn

The system has smart features that handle real F1 scenarios:

### 1. **Team-Based Rookie Handling**

This was trickier than expected. Early versions treated all rookies the same - filling missing data with grid averages. Problem: Antonelli at Mercedes has way more potential than a pay driver at a backmarker team.

**New approach:**
- Each team gets a performance baseline (Red Bull ~P2, Williams ~P17, etc.)
- Rookies inherit their team's baseline + uncertainty penalty
- As they get more races, gradually shift to their own performance data

Example progression:
```
Antonelli at Mercedes (2025):
Race 1: Team baseline (P6) + rookie penalty (+1.5) = Predict P8
Race 5: 60% team baseline + 40% his actual data = Predict P7
Race 15: Fully his own data = Predict wherever he's actually been
```

**Why this matters:**
- 2025 has 4 rookies (~20% of the grid)
- Model was treating "no historical data" as "probably midfield"
- Now it knows Antonelli (Mercedes) is not equal to Bortoleto (Sauber)

The system tracks team performance separately, regenerates baselines after each race, and intelligently fills missing features based on team context. Much better than just saying "idk, probably P10?"

### 2. **Robust Missing Data Handling**

The November 28 2025 update fixed critical production bugs around missing data:

**The Problem:**
- Predicting 2025 races before they happen (no historical data yet)
- Rookies with zero F1 experience
- Drivers at new circuits (Las Vegas 2023)
- API was crashing with "index out of bounds" errors

**The Solution - 5-Level Feature Fallback:**
1. **Exact match**: Driver at this circuit this year
2. **Circuit history**: Driver at this circuit (any recent year)
3. **Same year average**: Driver's average at other circuits this year
4. **Previous year**: Driver's average from last season
5. **Population median**: Last resort baseline (rookies without team data)

**Why this matters:**
- System never crashes on missing data anymore
- Predictions are context-aware (uses most relevant available data)
- Graceful degradation (best data → good data → acceptable baseline)

Example: Predicting Verstappen at Las Vegas 2025 (before race happens)
```
Try: VER + Las Vegas + 2025 → Not found
Try: VER + Las Vegas + 2024 → Found! Use this
Result: Prediction based on 2024 Las Vegas performance
```

Example: Predicting rookie Antonelli at Monaco 2025
```
Try: ANT + Monaco + 2025 → Not found (rookie)
Try: ANT + Monaco + any year → Not found (rookie)
Try: ANT + other 2025 circuits → Not found (hasn't raced yet)
Try: ANT + 2024 data → Not found (rookie)
Use: Mercedes team baseline + rookie adjustment
Result: Prediction ~P7 (good team, rookie uncertainty)
```

### 3. **Recent Form Tracking**

Looks at last 5 races to catch momentum:
- Lawson suddenly performing well at RB? Model adjusts up
- Verstappen has 3 bad weekends? Model accounts for it
- Filters out outliers (pit lane starts, DNQs) to avoid noise

### 4. **Team Change Detection**

Drivers moving teams (Hamilton to Ferrari, Sainz to Williams) are tricky:
- Use 70% driver historical skill
- Mix in 30% new team baseline performance
- Cap predictions at realistic levels for new team

Example: Sainz at Williams in 2025
- His Ferrari history says P6-8 (fast driver, fast car)
- Williams baseline says P14-16 (slower car)
- Blended prediction: P10-12 (fast driver, slower car)
- Reality check: Can't predict P6 at Williams (unrealistic)

### 5. **Smart Missing Data Handling**

Because real F1 data is messy:
- Team names change (AlphaTauri to RB, Alfa Romeo to Sauber)
- Circuits get added mid-season (Las Vegas 2023)
- Rookies have zero history
- Returnees (Colapinto back to reserve back to race seat)

The system canonicalizes team names, fills missing values intelligently based on context, and tracks data availability per feature. No more "NaN means guess randomly".


## The Self-Learning Part (MLOps)

This was the real learning goal - building a production ML system that improves automatically:

### How It Works

**After each race weekend:**
1. Run `python main.py --from 2022 --to 2025` to extract new data
2. System detects: "Hey, there's 1 new race!"
3. Automatically regenerates team baselines with new data
4. Retrains all 3 models with updated features
5. Compares new models vs current: Are they better?
6. If yes deploys new version automatically
7. API picks up new models within 60 seconds (no restart!)

**What gets tracked:**
```
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

### Why This Matters

Most ML projects I saw in tutorials:
- Train model once
- Save to disk
- Never update it
- Accuracy slowly degrades

Real production ML:
- Continuous learning from new data
- Automatic deployment of improvements
- Version control (can rollback if needed)
- Performance monitoring
- Dynamic feature generation (team baselines update)

This project does the real thing. It's not just "I trained a model", it's "I built a self-improving system".


## Current Performance

| Model | What It Predicts | Baseline | Actual | Status |
|-------|-----------------|----------|--------|--------|
| Q3 | Will driver make top 10? | 50% | **74%** | Production |
| Top 3 | Will driver podium? | 15% | **89%** | Production |
| Round | Which round (Q1/Q2/Q3)? | 33% | **75%** | Production |

**Recent improvements (Nov 28, 2025):**
- Fixed critical missing data crashes (API 500 errors eliminated)
- 5-level feature fallback hierarchy implemented
- 2025 predictions now work (including future races)
- Rookie predictions improved with team context
- Team change handling more realistic (capped blending)

### What "Production" Means Here

- **REST API**: FastAPI with Swagger docs at `/docs`
- **Dynamic loading**: API reloads models when new version deployed
- **Health checks**: `/health` endpoint shows model status
- **Version tracking**: Know exactly which model version served which prediction
- **Graceful degradation**: Missing features? Use intelligent fallback, don't crash
- **Error handling**: Proper validation, not just 500 errors
- **Robust predictions**: Handles rookies, team changes, future races, missing data

Basically, it's deployable to actual users (if anyone wanted F1 predictions from me).


## Project Structure

```
formula1/
├── main.py                    # Data pipeline (extracts from FastF1)
├── race_prediction.py         # NEW: Production prediction script
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
    ├── predictor.py             # Prediction logic (FIXED: robust fallback)
    └── dynamic_model_loader.py  # Auto-reload models
```


## Quick Start

### Installation
```bash
git clone https://github.com/tomasz-solis/formula1.git
cd formula1
python -m venv f1env
source f1env/bin/activate
pip install -r requirements.txt
```

### Extract Data & Train Models
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

### Start the API
```bash
cd api
uvicorn main:app --reload

# Visit http://127.0.0.1:8000 for nice landing page
# Or http://127.0.0.1:8000/docs for Swagger UI
```

### Make Predictions (API)
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

### Make Predictions (Script - NEW)
```bash
# Predict full grid for upcoming race
# 1. Edit race_prediction.py - update RACE_CONFIG at top
# 2. Set circuit, weather, sprint weekend status
# 3. Run:
python race_prediction.py

# Output:
# - Full grid predictions with probabilities
# - Recent form adjustments
# - Team change impacts
# - Practice session integration (if available)
# - Saves to CSV: Qatar_GP_2025_predictions.csv
```

**Race Configuration Example:**
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

To predict a new race, just update these 5 values. Script handles:
- Sprint weekends (FP1 + Sprint Quali)
- Normal weekends (FP1 + FP2 + FP3)
- Rookies (uses team baseline)
- Team changes (blended predictions)
- Missing data (intelligent fallback)


## Features That Actually Matter

After trying 50+ features, these are what the models actually use:

**Top Features for Q3 Prediction:**
1. `dry_avg_position` (24%) - How they normally qualify in dry
2. `wet_avg_position` (18%) - How they qualify in wet
3. `team_recent_avg` (17%) - Team's current form
4. `team_baseline_quali` (8%) - Team performance baseline
5. `recent_avg_position` (4%) - Driver's last 5 races
6. `circuit_avg_position` (3%) - History at this track

**Key insight**: Weather-adjusted features matter. The model learned that some drivers (Sainz, Alonso, Norris) are rain specialists - they punch above their weight when it's wet. Also learned that team context is crucial for drivers with limited history.


## Things I Learned Building This

### Data Leakage is Sneaky
First version had 99% accuracy. Suspicious.

Turned out I was accidentally including test set data in the training features:
- "Circuit average" included races from 2024... when predicting 2024
- Basically giving the model the answers
- Fixed it, accuracy dropped to realistic 74% (honest)

**Lesson**: Always validate your train/test split isn't contaminated

### Missing Data Requires Context (NEW)

First approach to missing data: Fill with median. Simple, wrong.

**Problem discovered (Nov 2025):**
- API crashed when predicting 2025 races (future data doesn't exist yet)
- Rookies got median-filled features (treating "no data" like "bad data")
- Feature count mismatches (2 features returned when model expects 48)

**Solution implemented:**
- 5-level fallback hierarchy (exact match → circuit history → year average → previous year → median)
- Context-aware imputation (rookies get team baseline, not population median)
- Graceful degradation (always return valid feature vector)

**Result:**
- Zero crashes on missing data
- Predictions work for future races (uses most recent data)
- Rookies predicted sensibly (Mercedes rookie not equal to Sauber rookie)

**Lesson**: Understand WHY data is missing before you fill it. Different types of missing data need different strategies.

### Small Datasets Force You to be Smart
F1 only has ~20 races per year times 20 drivers = 400 data points/year

Can't just throw data at the problem. Had to:
- Use careful feature engineering
- Blend historical baselines for new drivers
- Handle missing data intelligently (rookie at new circuit = what?)

**Lesson**: Feature engineering > more data (when you can't get more data)

### Classification > Regression for Chaos
Predicting "P1 vs P2 vs P3..." is too granular when:
- Strategy varies (fuel loads, tire choice)
- Red flags shuffle everything
- One mistake in Q3 = P1 to P10

But "Will they make Q3?" averages out the noise.

**Lesson**: Match your problem granularity to your data's predictability

### Team Context Matters More Than Expected
Tried to predict rookies using just driver-level features. Failed.

Antonelli at Mercedes was getting predicted at P15 (midfield) because "no data = assume average". But Mercedes isn't average.

Added team baselines rookie predictions improved significantly. The model learned "unknown Mercedes driver" is way different from "unknown Williams driver".

**Lesson**: Domain knowledge > generic ML tricks. F1 teams vary wildly in performance.

### Production Means Handling Edge Cases (NEW)

Building a model that works on clean training data is easy. Building one that doesn't crash in production is hard.

**Edge cases fixed:**
- Future races (data doesn't exist yet)
- Rookies mid-season (partial data)
- Team changes (conflicting historical data)
- New circuits (no circuit history)
- Missing practice data (weather issues, crashes)
- Numpy type serialization (JSON doesn't understand numpy.bool_)

**Lesson**: Production ML is 20% modeling, 80% handling edge cases gracefully.


## What's Next

**Short term (learning):**
- Add practice session predictions (FP1 → FP2 → FP3 progression)
- Try gradient boosting (XGBoost) vs Random Forest
- Feature engineering: tire compound effects, track evolution

**Medium term (production):**
- Docker deployment
- Monitoring dashboard (how's the model doing this season?)
- A/B testing new model versions
- Automated data quality alerts
- Race predictions (not just qualifying)

**Long term (maybe):**
- Strategy optimization ("should Hamilton pit now?")
- Real-time predictions during FP sessions
- Web UI for non-technical users


## Technical Bits

**Stack:**
- Python 3.13
- FastF1 API for telemetry
- scikit-learn for models
- FastAPI for serving
- Parquet for data storage

**Why these choices:**
- FastF1: Only good F1 data source
- scikit-learn: Simple, interpretable models (can debug)
- FastAPI: Fast, automatic docs, easy
- Parquet: Columnar storage, way faster than CSV

**Recent additions (Nov 28, 2025):**
- 5-level feature fallback hierarchy (robust missing data handling)
- Numpy type conversion (fixes JSON serialization errors)
- Production race prediction script (handles sprint weekends)
- Improved team change blending (realistic capping)


## Contact

**Tomasz Solis**
- Email: tomasz.solis@gmail.com
- LinkedIn: [linkedin.com/in/tomaszsolis](https://www.linkedin.com/in/tomaszsolis/)
- GitHub: [github.com/tomasz-solis](https://github.com/tomasz-solis)

Happy to chat about ML, F1, or how many times I broke this before it worked!


**Last Updated:** November 28, 2025  
**Status:** Production-ready with robust missing data handling  
**Current Models:**
- Top 3 Finish: 88.9% accuracy (vs 15% baseline)
- Q3 Qualification: 73.9% accuracy (vs 50% baseline)
- Qualifying Round: 75.3% accuracy (vs 33% baseline)
- Feature Fallback: 5-level hierarchy handles rookies, new circuits, team changes
