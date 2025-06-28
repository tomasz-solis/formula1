# Formula 1 Predictor

A modular toolkit for extracting, profiling and predicting Formula 1 performance, built on top of FastF1 (with OpenF1 fallback) and Ergast.

---

## Features

1. **Circuit Typology & Profiles**  
   - Corner counts by speed (slow/medium/fast), chicanes  
   - Braking-event counts  
   - Average & top speed, speed-zone percentages  
   - DRS zone counts & lengths  
   - Weather (air & track temperature, rainfall)  
   - Altitude from Open-Meteo  

2. **Driver-Level Telemetry Features**  
   - Full-throttle ratio  
   - Tire-degradation slope (per-stint lap-time trend)  
   - DRS flap activations (bit-mask aware, 2018–2025)  
   - Braking intensity (max/mean decel in g)  

3. **Automated Caching & Incremental Updates**  
   - Stores all circuit profiles in `data/circuit_profiles.csv`  
   - On race-weekends, automatically appends only sessions that have *actually started*  
   - Uses FastF1 → OpenF1 fallback for missing sessions  
   - Skips any session whose UTC start date is still in the future  

4. **Predictive Modeling Ready**  
   - Merge circuit and driver features into a single training table  
   - Easily pipeline into scikit-learn for classification (win probability), regression (lap-time ranking) or clustering (track typology)  

---

## Installation

```bash
git clone https://github.com/your-user/f1-predictor.git
cd f1-predictor
pip install -r requirements.txt
```

# How it works?

1. Session Loading
Tries FastF1 (with telemetry + laps), and if that fails or no laps are present, falls back to OpenF1 REST API.

2. Circuit Profiles
Loops over all past sessions in each season’s Ergast-provided schedule, extracts telemetry & metadata, and writes to a cache.

3. Incremental Updates
On subsequent runs, only looks at events whose FP1 has passed AND each individual session’s scheduled UTC timestamp ≤ now.

4. Driver Features
For each driver’s fastest lap, telemetry is parsed to compute throttle ratio, braking events, DRS activations, tyre wear slope, etc.

5. Merging & Modeling
Circuit & driver tables are joined on (year, event, session, location), ready for scikit-learn pipelines (scaling, PCA, clustering or supervised models).

# Future additions:

- Heatmaps per Track Cluster
- Predictive Power of FP1-2-3 for Q & R
- Predicting Quali and Race order
