# 🏎️ Formula 1 Performance Analytics & Predictive Pipeline

A modular and extensible project to explore, analyze, and model Formula 1 telemetry and session data. The current focus is on building **session-level performance profiles**, **driver telemetry analytics**, and **track classification** for exploratory and predictive purposes.

Future versions will evolve into a **cloud-deployed application** with automated pipelines and interactive visualizations.

---

## 📌 Project Structure

```
formula1/
├── data/ # Session and intermediate data (excluded from version control)
├── f1env/ # Optional: conda/venv environment (excluded from version control)
├── helpers/ # Core logic modules (utils)
│ ├── circuit_utils.py # Track-level clustering & profiling
│ ├── driver_utils.py # Telemetry-based driver metrics
│ ├── general_utils.py # Shared utility functions
│ ├── predictive_utils.py # Predictive modeling and feature engineering
│ └── init.py
├── circuit.ipynb # EDA and clustering for track profiles
├── wip.ipynb # Working notebook for prototyping
├── LICENSE
├── README.md # Project overview and documentation
└── requirements.txt # Package dependencies
```


---

## ✅ Current Features

### 🔍 Exploratory Analysis
- Telemetry and weather data extraction per session
- DRS, braking intensity, throttle ratio, tire degradation proxy
- Session and track-level summaries

### 🧠 Clustering & Profiles
- Track classification using PCA + KMeans (or other clustering algorithms)
- Grouping by `track_id`, with customizable feature selection
- Output profiles for circuit similarity analysis

### 🛠 Modular Utilities
- Functions separated into logical modules for reuse and extensibility
- Clean handling of missing values, scaling, and transformation pipelines


## 🗺️ Roadmap

### Short-Term Goals (In Progress)
- ✅ Track clustering via circuit profiles
- ✅ Driver telemetry-based metric extraction
- ⏳ Predictive modeling for Qualifying & Race pace
- ⏳ Refactor processing logic into `main.py`

### Medium-Term Goals (Upcoming)
- 📦 Year-by-year pipeline: incrementally build data as weekends progress
- 🧪 Proper train/test splitting for model evaluation
- ☁️ Auto-processing pipeline outputting to local or cloud (e.g. S3)

### Long-Term Vision
- 🚀 Hosted web app on AWS (e.g. EC2 + S3 or Lightsail + Streamlit/FastAPI)
- 📊 Interactive dashboards for Qualifying, Race pace, and strategy insights
- 🏁 Real-time updates during Grand Prix weekends

## 📦 Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/f1project.git
cd f1project/formula1
```

### 2. Set up virtual environment (optional but recommended)
```bash
python -m venv f1env
source f1env/bin/activate   # or f1env\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. How to Use
- circuit.ipynb: Run track clustering and visualize PCA output.

# 🤝 Acknowledgements
- FastF1 — telemetry and timing data

- OpenF1 — alternative data source

- The broader F1 data and fan community ❤️

# Contact

Let me know if you'd like help customizing:
- Example screenshots/plots
- A contributing section
- Streamlit/FastAPI scaffolding for your web app
- or GitHub Actions for automating the pipeline later

tomasz.solis@gmail.com

