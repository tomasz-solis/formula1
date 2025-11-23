# 🏎️ Formula 1 Qualifying Predictor

Predicting F1 qualifying outcomes using machine learning. This started as a way to learn ML properly - turns out predicting exact grid positions is basically impossible (chaos theory is real!), but predicting who makes Q3 or podium? That works pretty well.

**Current Models:** Q3: 73% accurate | Top 3: 89% accurate | Round: 78% accurate

The system learns from each race weekend automatically - no manual retraining needed. Now with proper rookie handling (because Antonelli at Mercedes ≠ random rookie at Williams).

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

---

## The Journey: From Failure to Working Models

### What Didn't Work: Regression

First attempt was predicting exact qualifying positions (P1, P2, P3...). Complete disaster:
- **My model MAE: 3.78 positions**
- **Naive baseline (just use practice times): 3.60 positions**
- My fancy ML model was **worse than doing nothing**

Why? Because qualifying is chaos:
- Verstappen crashes in Q1 → goes from P1 to P20
- Red flag at wrong time → random driver in P3
- Rain in Q3 → everything shuffles

The model basically learned "past position = future position" and gave up trying to predict actual patterns.

### What Works: Classification

Switched to yes/no questions instead:
- "Will this driver make Q3?" (top 10)
- "Will they get pole/P2/P3?" (podium)
- "Which round will they reach?" (Q1/Q2/Q3)

**Immediately better results:**
- Q3 prediction: 73% accurate (vs 50% if you just guess)
- Top 3 prediction: 89% accurate (vs 15% baseline)
- Stopped trying to predict the unpredictable, started predicting patterns

---

## What Makes It Actually Learn

The system has smart features that handle real F1 scenarios:

### 1. **Team-Based Rookie Handling** (The Latest Addition)

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
- 2025 has 5 rookies (~25% of the grid)
- Model was treating "no historical data" as "probably midfield"
- Now it knows Antonelli (Mercedes) ≠ Bortoleto (Sauber)
- Rookie prediction error dropped ~0.4 positions

The system tracks team performance separately, regenerates baselines after each race, and intelligently fills missing features based on team context. Much better than just saying "idk, probably P10?"

### 2. **Recent Form Tracking**

Looks at last 5 races to catch momentum:
- Lawson suddenly performing well at RB? Model adjusts up
- Verstappen has 3 bad weekends? Model accounts for it
- Filters out outliers (pit lane starts, DNQs) to avoid noise

### 3. **Team Change Detection**

Drivers moving teams (Hamilton to Ferrari, Sainz to Williams) are tricky:
- Use 60% team historical performance at that circuit
- Mix in 40% driver's own historical performance
- Gradually shift to full driver data as races accumulate

Example: Sainz at Williams in 2025
- Monaco prediction: 60% Williams history (rough) + 40% Sainz history (good) = realistic P14-16
- After 5 races at Williams: Model uses his actual Williams performance

### 4. **Smart Missing Data Handling**

Because real F1 data is messy:
- Team names change (AlphaTauri → RB, Alfa Romeo → Sauber)
- Circuits get added mid-season (Las Vegas 2023)
- Rookies have zero history
- Returnees (Colapinto → back to reserve → back to race seat)

The system canonicalizes team names, fills missing values intelligently based on context, and tracks data availability per feature. No more "NaN means guess randomly".

---

## The Self-Learning Part (MLOps)

This was the real learning goal - building a production ML system that improves automatically:

### How It Works

**After each race weekend:**
1. Run `python main.py --from 2022 --to 2025` to extract new data
2. System detects: "Hey, there's 1 new race!"
3. Automatically regenerates team baselines with new data
4. Retrains all 3 models with updated features
5. Compares new models vs current: Are they better?
6. If yes → deploys new version automatically
7. API picks up new models within 60 seconds (no restart!)

**What gets tracked:**
```
models/
├── v20251122_125637/  ← Version from Nov 22
│   ├── q3_classifier.pkl
│   ├── metadata.json (accuracies, training date)
├── v20251201_083045/  ← Version from Dec 1 (after Qatar)
│   ├── q3_classifier.pkl
│   ├── metadata.json
├── team_baselines.json      ← Regenerates after each race
├── training_history.json    ← Shows improvement over time
└── active_version.txt       ← API uses this
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

---

## Current Performance

| Model | What It Predicts | Baseline | Actual | Status |
|-------|-----------------|----------|--------|--------|
| Q3 | Will driver make top 10? | 50% | **73%** | ✅ Production |
| Top 3 | Will driver podium? | 15% | **89%** | ✅ Production |
| Round | Which round (Q1/Q2/Q3)? | 33% | **78%** | ✅ Production |

**Recent improvements:**
- Rookie prediction error: -0.4 positions (after team-based priors)
- Team change handling: More accurate first-race predictions
- Missing data handling: Zero crashes on incomplete data

### What "Production" Means Here

- **REST API**: FastAPI with Swagger docs at `/docs`
- **Dynamic loading**: API reloads models when new version deployed
- **Health checks**: `/health` endpoint shows model status
- **Version tracking**: Know exactly which model version served which prediction
- **Graceful degradation**: Missing features? Use team baseline, don't crash
- **Error handling**: Proper validation, not just 500 errors

Basically, it's deployable to actual users (if anyone wanted F1 predictions from me lol).

---

## Project Structure

```
formula1/
├── main.py                    # Data pipeline (extracts from FastF1)
├── helpers/
│   ├── auto_retrain.py       # Auto-retraining after new races
│   ├── feature_engineering.py # Feature generation
│   ├── historical_features.py # Time-series features
│   ├── team_priors.py        # NEW: Team baseline computation
│   ├── team_name_mapping.py  # NEW: Handle team rebranding
│   └── validation.py         # Data quality checks
├── data/
│   ├── features/
│   │   └── ml_features.parquet  # All training data
│   └── predictions/ssot/        # Official results
├── models/
│   ├── q3_classifier.pkl        # Active models
│   ├── top3_classifier.pkl
│   ├── round_classifier.pkl
│   ├── team_baselines.json      # NEW: Team performance baselines
│   ├── v20251122_125637/        # Version history
│   └── training_history.json
└── api/
    ├── main.py                  # FastAPI server
    ├── predictor.py             # Prediction logic
    └── dynamic_model_loader.py  # Auto-reload models
```

---

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

### Make Predictions
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

---

## Features That Actually Matter

After trying 50+ features, these are what the models actually use:

**Top Features for Q3 Prediction:**
1. `dry_avg_position` (24%) - How they normally qualify in dry
2. `wet_avg_position` (18%) - How they qualify in wet
3. `team_recent_avg` (17%) - Team's current form
4. `team_baseline_quali` (8%) - NEW: Team performance baseline
5. `recent_avg_position` (4%) - Driver's last 5 races
6. `circuit_avg_position` (3%) - History at this track

**Key insight**: Weather-adjusted features matter. The model learned that some drivers (Sainz, Alonso, Norris) are rain specialists - they punch above their weight when it's wet. Also learned that team context is crucial for drivers with limited history.

---

## Things I Learned Building This

### Data Leakage is Sneaky
First version had 99% accuracy. Suspicious.

Turned out I was accidentally including test set data in the training features:
- "Circuit average" included races from 2024... when predicting 2024
- Basically giving the model the answers
- Fixed it, accuracy dropped to realistic 73% (honest)

**Lesson**: Always validate your train/test split isn't contaminated

### Small Datasets Force You to be Smart
F1 only has ~20 races per year × 20 drivers = 400 data points/year

Can't just throw data at the problem. Had to:
- Use careful feature engineering
- Blend historical baselines for new drivers
- Handle missing data intelligently (rookie at new circuit = what?)

**Lesson**: Feature engineering > more data (when you can't get more data)

### Classification > Regression for Chaos
Predicting "P1 vs P2 vs P3..." is too granular when:
- Strategy varies (fuel loads, tire choice)
- Red flags shuffle everything
- One mistake in Q3 = P1 → P10

But "Will they make Q3?" averages out the noise.

**Lesson**: Match your problem granularity to your data's predictability

### Missing Data Isn't Always Random
First approach: Fill NaN with median. Simple, wrong.

Problem: Rookie with no history ≠ bad driver with bad history. Both had NaN features, but very different meanings.

Solution: Context-aware filling
- Rookies get team baseline + uncertainty
- Circuit rookies get driver baseline at similar tracks
- Missing weather data gets interpolated from nearby sessions

**Lesson**: Understand WHY data is missing before you fill it

### Team Context Matters More Than Expected
Tried to predict rookies using just driver-level features. Failed.

Antonelli at Mercedes was getting predicted at P15 (midfield) because "no data = assume average". But Mercedes isn't average.

Added team baselines → rookie predictions improved significantly. The model learned "unknown Mercedes driver" is way different from "unknown Williams driver".

**Lesson**: Domain knowledge > generic ML tricks. F1 teams vary wildly in performance.

---

## What's Next

**Short term (learning):**
- Add sprint qualifying predictions
- Try gradient boosting (XGBoost) vs Random Forest
- Feature engineering: tire compound effects, track evolution

**Medium term (production):**
- Docker deployment
- Monitoring dashboard (how's the model doing this season?)
- A/B testing new model versions
- Automated data quality alerts

**Long term (maybe):**
- Race predictions (not just qualifying)
- Strategy optimization ("should Hamilton pit now?")
- Real-time predictions during FP sessions

---

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

**Recent additions:**
- Team baseline computation (JSON storage)
- Canonicalized team names (handles rebranding)
- Smart NaN filling (context-aware imputation)

---

## Contact

**Tomasz Solis**
- Email: tomasz.solis@gmail.com
- LinkedIn: [linkedin.com/in/tomaszsolis](https://www.linkedin.com/in/tomaszsolis/)
- GitHub: [github.com/tomasz-solis](https://github.com/tomasz-solis)

Happy to chat about ML, F1, or how many times I broke this before it worked!

---

**Last Updated:** November 23, 2025  
**Status:** Classification models production-ready with team-based rookie handling  
**Current Models:**
- Top 3 Finish: 89.1% accuracy (vs 15% baseline)
- Q3 Qualification: 78.8% accuracy (vs 50% baseline)
- Qualifying Round: 70.0% accuracy (vs 33% baseline)
- Rookie Predictions: -0.4 position error improvement via team priors