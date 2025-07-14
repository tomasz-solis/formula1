# 🏎️ Formula 1 Performance Analytics & Predictive Pipeline

A modular and extensible project to explore, analyze, and model Formula 1 telemetry and session data. The current focus is on building **session-level performance profiles**, **driver telemetry analytics**, and **track classification** for exploratory and predictive purposes.

Future versions will evolve into a **cloud-deployed application** with automated pipelines and interactive visualizations.

---

## Project Structure

```
formula1/
├── main.py                  # Entry point for running pipeline
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── EDA/                     # Exploratory Jupyter notebooks
│   ├── general.ipynb        # General exploratory notebook - source of inspiration
│   └── wip.ipynb            # Work-in-progress testing notebook
├── data/                    # Cached and generated data
│   ├── .fastf1_cache/       # FastF1 session cache
│   ├── circuit/             # Circuit profile CSVs
│   ├── driver/              # Driver profile CSVs
│   └── driver_timing/       # Driver timing Parquet files
└── helpers/                 # Pipeline helper modules
    ├── __init__.py
    ├── general_utils.py     # Session loading, caching, schedule helpers
    ├── driver_utils.py      # Driver performance feature extraction
    └── circuit_utils.py     # Circuit metadata and analytics
```

---

## Current Features

### Data Pipeline

- Builds or updates **circuit**, **driver**, and **driver timing** profiles
- Supports filtering to a specific Grand Prix via `--gp "Event Name"`

### Exploratory Analysis

- Telemetry and weather data extraction per session
- DRS activations, braking intensity, throttle ratio, tire degradation proxy
- Session and track-level summaries

### Clustering & Profiles

- Track classification using PCA + KMeans (or other clustering algorithms)
- Grouping by `track_id`, with customizable feature selection
- Output profiles for circuit similarity analysis

### Modular Utilities

- Functions separated into logical modules for reuse and extensibility
- Clean handling of missing values, scaling, and transformation pipelines

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/tomasz-solis/formula1.git
   cd formula1
   ```

2. **Set up virtual environment** (optional but recommended)

   ```bash
   python -m venv f1env
   source f1env/bin/activate   # or f1env\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

## Usage

### Run the data pipeline

```bash
# Process seasons 2022 through 2025 for all events
git pull && python main.py --from 2022 --to 2025

# Process only the British Grand Prix each year
git pull && python main.py --from 2022 --to 2025 --gp "British Grand Prix"
```

### Analytics Notebooks

- **EDA/general.ipynb**: Run track clustering and visualize PCA output
- Forecast Qualifying & Race pace based on practice sessions
- Profile CSVs and Parquet telemetry files are saved under `data/`

## Roadmap

### Short-Term Goals (In Progress)

- ✅ Track clustering via circuit profiles
- ✅ Driver telemetry-based metric extraction
- ✅ Refactor processing logic into `main.py`
- ⏳ Enhanced driver profiles: sector and mini-sector times
- ⏳ Predictive modeling for Qualifying & Race pace

### Medium-Term Goals (Upcoming)

- ⏳ Unified logging outputs across modules
- ⏳ Optimize `main.py` for multi-year incremental builds
- 📦 Incremental pipeline building as weekends progress
- 🧪 Proper train/test splitting for model evaluation
- ☁️ Automate pipeline outputs (e.g., AWS S3 integration)

### Long-Term Vision

- 🚀 Hosted web app (Streamlit/FastAPI on AWS)
- 📊 Interactive dashboards for Qualifying, Race pace, and strategy insights
- 🏁 Real-time updates during Grand Prix weekends

## Acknowledgements

- [Mirco Bartolozzi](https://www.linkedin.com/in/mirco-bartolozzi/) — Formula Data Analysis inspiration
- **FastF1** — telemetry and timing data
- **OpenF1** — alternative data source
- The broader F1 data and fan community ❤️

## Contact

For help customizing or extending this project:

- [tomasz.solis@gmail.com](mailto\:tomasz.solis@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/tomaszsolis/)

---

*Last updated: July 14, 2025*

