# 🏎️ Formula 1 Qualifying Predictor

Predicting F1 qualifying outcomes using machine learning. This started as a way to learn ML properly - turns out predicting exact grid positions is basically impossible (chaos theory is real!), but predicting who makes Q3 or podium? That works pretty well.

**Current Models:** Q3: 73% accurate | Top 3: 89% accurate | Round: 78% accurate

The system learns from each race weekend automatically - no manual retraining needed.

---

## Why This Project Exists

I'm a senior product analyst trying to move into ML/data science. Instead of just doing Kaggle competitions, I wanted to build something real:
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

The system has three smart features that handle real F1 scenarios:

### 1. **Rookie Handling**
New drivers (Antonelli, Bortoleto, Hadjar, Bearman in 2025) get special treatment:
- Start with team baseline performance
- Gradually blend in their own results as they get more races
- In wet conditions, reduce their predicted performance (rookies struggle in rain)

Example: Antonelli at Mercedes
- First race: Uses Mercedes team average (assume ~P8)
- After 3 races: 70% team baseline + 30% his actual results
- After 10 races: Fully his own performance profile

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
- Monaco prediction: 60% Williams history (bad) + 40% Sainz history (good) = realistic P15ish
- After 5 races at Williams: Model uses his actual Williams performance

---

## The Self-Learning Part (MLOps)

This was the real learning goal - building a production ML system that improves automatically:

### How It Works

**After each race weekend:**
1. Run `python main.py --from 2022 --to 2025` to extract new data
2. System detects: "Hey, there's 1 new race!"
3. Automatically retrains all 3 models with the new data
4. Compares new models vs current: Are they better?
5. If yes → deploys new version automatically
6. API picks up new models within 60 seconds (no restart!)

**What gets tracked:**
```
models/
├── v20251122_125637/  ← Version from Nov 22
│   ├── q3_classifier.pkl
│   ├── metadata.json (accuracies, training date)
├── v20251201_083045/  ← Version from Dec 1 (after Qatar)
│   ├── q3_classifier.pkl
│   ├── metadata.json
├── training_history.json  ← Shows improvement over time
└── active_version.txt     ← API uses this
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

This project does the real thing. It's not just "I trained a model", it's "I built a self-improving system".

---

## Current Performance

| Model | What It Predicts | Baseline | Actual | Status |
|-------|-----------------|----------|--------|--------|
| Q3 | Will driver make top 10? | 50% | **73%** | ✅ Production |
| Top 3 | Will driver podium? | 15% | **89%** | ✅ Production |
| Round | Which round (Q1/Q2/Q3)? | 33% | **78%** | ✅ Production |

### What "Production" Means Here

- **REST API**: FastAPI with Swagger docs at `/docs`
- **Dynamic loading**: API reloads models when new version deployed
- **Health checks**: `/health` endpoint shows model status
- **Version tracking**: Know exactly which model version served which prediction
- **Error handling**: Graceful failures, not crashes

Basically, it's deployable to actual users (if anyone wanted F1 predictions from me lol).

---

## Project Structure

```
formula1/
├── main.py                    # Data pipeline (extracts from FastF1)
├── helpers/
│   ├── auto_retrain.py       # Auto-retraining after new races
│   ├── feature_engineering.py
│   └── historical_features.py
├── data/
│   ├── features/
│   │   └── ml_features.parquet  # All training data
│   └── predictions/ssot/        # Official results
├── models/
│   ├── q3_classifier.pkl        # Active models
│   ├── top3_classifier.pkl
│   ├── round_classifier.pkl
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
# 3. Trains 3 classification models
# 4. Saves to models/ directory
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
4. `recent_avg_position` (4%) - Driver's last 5 races
5. `circuit_avg_position` (3%) - History at this track

**Key insight**: Weather-adjusted features matter. The model learned that some drivers (Sainz, Alonso, Norris) are rain specialists - they punch above their weight when it's wet.

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

---

## Contact

**Tomasz Solis**
- Email: tomasz.solis@gmail.com
- LinkedIn: [linkedin.com/in/tomaszsolis](https://www.linkedin.com/in/tomaszsolis/)
- GitHub: [github.com/tomasz-solis](https://github.com/tomasz-solis)

Happy to chat about ML, F1, or how many times I broke this before it worked!

---

**Last Updated:** November 18, 2024  
**Status:** Classification models production-ready  
**Current Models:**
- Top 3 Finish: 89.1% accuracy (vs 15% baseline)
- Q3 Qualification: 78.8% accuracy (vs 50% baseline)
- Qualifying Round: 70.0% accuracy (vs 33% baseline)