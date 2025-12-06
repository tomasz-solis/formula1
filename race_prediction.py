"""
F1 Qualifying Prediction Script - Enhanced Version

Supports:
- Normal race weekends (FP1, FP2, FP3)
- Sprint weekends (FP1, Sprint Qualifying)
- Easy race configuration at the top
- Recent form tracking
- Team-based priors for rookies and team changes
"""

import requests
import pandas as pd
import json
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

pdir = Path("predictions")
pdir.mkdir(exist_ok=True, parents=True)

# ============================================================================
# 🏁 RACE CONFIGURATION - CHANGE THIS FOR EACH RACE
# ============================================================================

# Current race details
RACE_CONFIG = {
    "circuit": "Abu Dhabi Grand Prix",  # ← CHANGE THIS (must match data exactly)
    "year": 2025,
    "is_sprint_weekend": False,      # ← CHANGE THIS (True for sprint, False for normal)
    
    # Weather conditions
    "weather": {
        "avg_rainfall": 0.0,        # ← CHANGE THIS (mm/h)
        "avg_track_temp": 40.0,     # ← CHANGE THIS (°C)
        "avg_air_temp": 28.0        # ← CHANGE THIS (°C)
    }
}

# Practice session names for data loading
SESSION_CONFIG = {
    "sprint": ['FP1', 'Sprint Qualifying'],
    "normal": ['FP1', 'FP2', 'FP3']
}

# ============================================================================
# DRIVER AND TEAM CONFIGURATION
# ============================================================================

# 2025 Grid
DRIVERS_2025 = [
    "VER", "TSU",  # Red Bull (Tsunoda replaced Perez)
    "NOR", "PIA",  # McLaren
    "LEC", "HAM",  # Ferrari (Hamilton moved from Mercedes)
    "RUS", "ANT",  # Mercedes (Antonelli rookie)
    "ALO", "STR",  # Aston Martin
    "COL", "GAS",  # Alpine (Colapinto)
    "OCO", "BEA",  # Haas (Bearman rookie)
    "LAW", "HAD",  # RB (Lawson, Hadjar rookie)
    "ALB", "SAI",  # Williams (Sainz from Ferrari)
    "HUL", "BOR"   # Sauber (Bortoleto rookie)
]

# Team mapping for context
TEAM_MAPPING = {
    "VER": "Red Bull", "TSU": "Red Bull",
    "NOR": "McLaren", "PIA": "McLaren",
    "LEC": "Ferrari", "HAM": "Ferrari",
    "RUS": "Mercedes", "ANT": "Mercedes",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "COL": "Alpine", "GAS": "Alpine",
    "OCO": "Haas", "BEA": "Haas",
    "LAW": "RB", "HAD": "RB",
    "ALB": "Williams", "SAI": "Williams",
    "HUL": "Sauber", "BOR": "Sauber"
}

# Expected qualifying positions by team (2024/2025 pace)
TEAM_EXPECTED_QUALI_POSITION = {
    "Red Bull": 3.5,      # Still strong but not dominant
    "McLaren": 2.5,       # Fastest car
    "Ferrari": 5.5,       # Competitive
    "Mercedes": 5.0,      # Improving
    "Aston Martin": 7.0,  # Midfield
    "Alpine": 16.0,       # Lower midfield
    "Williams": 10.5,     # Back of midfield
    "RB": 11.0,           # Midfield
    "Haas": 10.0,         # Struggling
    "Sauber": 12.0        # Back markers
}

# Rookies (no historical data)
ROOKIES = ["ANT", "BEA", "HAD", "BOR"]

# Recent team changes (historical data needs adjustment)
TEAM_CHANGES = {
    "HAM": {"from_team": "Mercedes", "to_team": "Ferrari"},
    "SAI": {"from_team": "Ferrari", "to_team": "Williams"},
    "TSU": {"from_team": "RB", "to_team": "Red Bull"},
    "HUL": {"from_team": "Haas", "to_team": "Sauber"}
}

# API endpoint
API_BASE = "http://127.0.0.1:8000"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_recent_form(year: int = 2025, lookback_races: int = 5) -> Optional[pd.DataFrame]:
    """
    Load recent qualifying form for all drivers.
    
    Calculates average qualifying position over last N races to capture
    current form and momentum.
    
    Args:
        year: Current year
        lookback_races: Number of recent races to consider (default: 5)
        
    Returns:
        DataFrame with driver recent form, or None if not available
    """
    try:
        features_file = Path("data/features/ml_features.parquet")
        
        if not features_file.exists():
            print("⚠️  No historical features found - using team baseline only")
            return None
        
        df = pd.read_parquet(features_file)
        
        # Filter to current year, before this race
        current_year = df[df['year'] == year].copy()
        
        if len(current_year) == 0:
            print(f"⚠️  No {year} data found for recent form")
            return None
        
        # Get last N races for each driver
        form_data = []
        for driver in current_year['driver'].unique():
            driver_races = current_year[current_year['driver'] == driver].copy()
            driver_races = driver_races.sort_values('year').tail(lookback_races)
            
            if len(driver_races) >= 2:  # Need at least 2 races for form
                recent_avg = driver_races['qualifying_position'].mean()
                recent_std = driver_races['qualifying_position'].std()
                num_races = len(driver_races)
                
                # Season average (for comparison)
                season_avg = current_year[current_year['driver'] == driver]['qualifying_position'].mean()
                
                # Form factor: recent vs season
                form_delta = recent_avg - season_avg
                
                form_data.append({
                    'driver': driver,
                    'recent_avg_position': recent_avg,
                    'recent_position_std': recent_std,
                    'season_avg_position': season_avg,
                    'form_delta': form_delta,
                    'num_recent_races': num_races,
                    'is_improving': form_delta < -0.5,
                    'is_declining': form_delta > 0.5
                })
        
        if len(form_data) == 0:
            return None
        
        form_df = pd.DataFrame(form_data)
        print(f"✅ Loaded recent form: {len(form_df)} drivers, last {lookback_races} races")
        
        return form_df
        
    except Exception as e:
        print(f"⚠️  Could not load recent form: {e}")
        return None


def load_practice_data(circuit: str, is_sprint: bool) -> Optional[pd.DataFrame]:
    """
    Load practice session data for current race.
    
    Args:
        circuit: Circuit name
        is_sprint: True if sprint weekend
        
    Returns:
        DataFrame with practice/sprint data, or None if not available
    """
    try:
        driver_file = Path("data/driver/2025_driver_profiles.csv")
        
        if not driver_file.exists():
            print("⚠️  No practice data found")
            return None
        
        df = pd.read_csv(driver_file)
        
        # Get appropriate sessions
        sessions = SESSION_CONFIG['sprint'] if is_sprint else SESSION_CONFIG['normal']
        
        practice_data = df[
            (df['event'] == circuit) & 
            (df['session'].isin(sessions))
        ].copy()
        
        if len(practice_data) == 0:
            print(f"⚠️  No {circuit} practice data found")
            return None
        
        sessions_found = practice_data['session'].unique().tolist()
        print(f"✅ Loaded data: {sessions_found}")
        
        return practice_data
        
    except Exception as e:
        print(f"⚠️  Could not load practice data: {e}")
        return None


def get_practice_baseline(
    driver: str, 
    practice_data: Optional[pd.DataFrame],
    is_sprint: bool
) -> Optional[float]:
    """
    Calculate driver baseline from practice sessions.
    
    For sprint weekends: prioritize Sprint Qualifying (70%) over FP1 (30%)
    For normal weekends: prioritize FP3 (50%) over FP1/FP2 (25% each)
    
    Args:
        driver: Driver abbreviation
        practice_data: DataFrame with practice data
        is_sprint: True if sprint weekend
        
    Returns:
        Weighted average position, or None if no data
    """
    if practice_data is None or len(practice_data) == 0:
        return None
    
    driver_data = practice_data[practice_data['driver'] == driver]
    
    if len(driver_data) == 0:
        return None
    
    # Check for position column
    if 'position' not in driver_data.columns:
        return None
    
    positions = {}
    for _, row in driver_data.iterrows():
        session = row['session']
        pos = row.get('position')
        
        if pd.notna(pos):
            positions[session] = pos
    
    if len(positions) == 0:
        return None
    
    # Calculate weighted average
    weighted_pos = 0.0
    total_weight = 0.0
    
    if is_sprint:
        # Sprint weekend: Sprint Quali > FP1
        if 'Sprint Qualifying' in positions:
            weighted_pos += positions['Sprint Qualifying'] * 0.7
            total_weight += 0.7
        if 'FP1' in positions:
            weighted_pos += positions['FP1'] * 0.3
            total_weight += 0.3
    else:
        # Normal weekend: FP3 > FP2 > FP1
        if 'FP3' in positions:
            weighted_pos += positions['FP3'] * 0.5
            total_weight += 0.5
        if 'FP2' in positions:
            weighted_pos += positions['FP2'] * 0.3
            total_weight += 0.3
        if 'FP1' in positions:
            weighted_pos += positions['FP1'] * 0.2
            total_weight += 0.2
    
    if total_weight == 0:
        return None
    
    return weighted_pos / total_weight


def get_team_adjustment(driver: str) -> Dict[str, float]:
    """
    Get feature adjustments based on team context.
    
    Args:
        driver: Driver abbreviation
        
    Returns:
        Dictionary of feature adjustments
    """
    team = TEAM_MAPPING.get(driver, "Unknown")
    expected_pos = TEAM_EXPECTED_QUALI_POSITION.get(team, 10.5)
    
    adjustments = {}
    
    # Rookies: Use team average
    if driver in ROOKIES:
        adjustments = {
            '_is_rookie': True,
            '_team_expected': expected_pos,
            '_note': f'Rookie on {team}'
        }
        print(f"   🆕 {driver} (Rookie): Using {team} baseline (P{expected_pos:.1f})")
    
    # Team changes: Blend old history with new team
    elif driver in TEAM_CHANGES:
        old_team = TEAM_CHANGES[driver]['from_team']
        new_team = TEAM_CHANGES[driver]['to_team']
        new_expected = TEAM_EXPECTED_QUALI_POSITION.get(new_team, 10.5)
        
        adjustments = {
            '_team_change': True,
            '_new_team_position': new_expected,
            '_note': f'Team change: {old_team} → {new_team}'
        }
        print(f"   🔄 {driver} (Team change): {old_team} → {new_team} (P{new_expected:.1f})")
    
    return adjustments


def enhance_prediction_with_context(
    base_prediction: Dict,
    driver: str,
    practice_position: Optional[float],
    team_adjustment: Dict,
    recent_form: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Enhance API prediction with practice data, team context, and recent form.
    
    Args:
        base_prediction: Original API prediction
        driver: Driver abbreviation
        practice_position: Practice average position (if available)
        team_adjustment: Team-based adjustments
        recent_form: DataFrame with recent form data (optional)
        
    Returns:
        Enhanced prediction dictionary
    """
    enhanced = base_prediction.copy()
    
    # PRACTICE DATA ADJUSTMENT
    if practice_position is not None:
        print(f"   📊 Practice: P{practice_position:.1f}", end=" ")
        
        # Adjust Q3 probability based on practice performance
        if practice_position <= 5:
            # Strong practice → boost Q3 probability
            boost = min(0.15, (6 - practice_position) * 0.03)
            enhanced['q3']['probability'] = min(0.95, enhanced['q3']['probability'] + boost)
            
        elif practice_position >= 16:
            # Weak practice → reduce Q3 probability
            penalty = min(0.15, (practice_position - 15) * 0.03)
            enhanced['q3']['probability'] = max(0.05, enhanced['q3']['probability'] - penalty)
        
        enhanced['q3']['will_make_q3'] = enhanced['q3']['probability'] >= 0.5
        print(f"→ Q3: {enhanced['q3']['probability']:.1%}")
    
    # TEAM CHANGE ADJUSTMENT
    elif '_team_change' in team_adjustment:
        new_team_pos = team_adjustment['_new_team_position']
        current_prob = enhanced['q3']['probability']
        
        # Calculate new team baseline probability
        if new_team_pos <= 7:
            new_team_baseline = 0.80
        elif new_team_pos <= 10:
            new_team_baseline = 0.55
        elif new_team_pos <= 13:
            new_team_baseline = 0.40
        else:
            new_team_baseline = 0.25
        
        # Blend: 70% historical skill + 30% new team baseline
        blended_prob = (current_prob * 0.7) + (new_team_baseline * 0.3)
        
        # Cap based on team move direction
        if new_team_pos > 10:  # Downgrade
            max_prob = new_team_baseline + 0.15
            enhanced['q3']['probability'] = min(blended_prob, max_prob)
        elif new_team_pos < 8:  # Upgrade
            min_prob = new_team_baseline - 0.10
            enhanced['q3']['probability'] = max(blended_prob, min_prob)
        else:  # Lateral
            enhanced['q3']['probability'] = blended_prob
        
        enhanced['q3']['will_make_q3'] = enhanced['q3']['probability'] >= 0.5
        print(f"→ Team-adjusted Q3: {enhanced['q3']['probability']:.1%}")
    
    # ROOKIE ADJUSTMENT
    elif '_is_rookie' in team_adjustment:
        team_expected = team_adjustment['_team_expected']
        
        # Adjust based on expected team position
        if team_expected <= 7:
            enhanced['q3']['probability'] = max(enhanced['q3']['probability'], 0.65)
            enhanced['q3']['will_make_q3'] = True
        elif team_expected >= 14:
            enhanced['q3']['probability'] = min(enhanced['q3']['probability'], 0.35)
            enhanced['q3']['will_make_q3'] = False
        
        print(f"→ Rookie-adjusted Q3: {enhanced['q3']['probability']:.1%}")
    
    # RECENT FORM ADJUSTMENT (applied after all base adjustments)
    if recent_form is not None and driver in recent_form['driver'].values:
        driver_form = recent_form[recent_form['driver'] == driver].iloc[0]
        
        form_delta = driver_form['form_delta']
        recent_avg = driver_form['recent_avg_position']
        
        # Form adjustment: -1.0 position = +5% probability
        form_adjustment = -form_delta * 0.05
        form_adjustment = np.clip(form_adjustment, -0.10, 0.10)
        
        old_prob = enhanced['q3']['probability']
        enhanced['q3']['probability'] = np.clip(old_prob + form_adjustment, 0.05, 0.95)
        enhanced['q3']['will_make_q3'] = enhanced['q3']['probability'] >= 0.5
        
        if abs(form_adjustment) >= 0.03:
            trend = "📈" if form_delta < 0 else "📉"
            print(f"   {trend} Form: P{recent_avg:.1f} (Δ{form_delta:+.1f}) → {old_prob:.1%} → {enhanced['q3']['probability']:.1%}")
    
    return enhanced


# ============================================================================
# API INTERACTION
# ============================================================================

def predict_driver(driver: str, config: Dict) -> Dict:
    """
    Get predictions for a single driver.
    
    Args:
        driver: Driver abbreviation (e.g., "VER")
        config: Race configuration dictionary
        
    Returns:
        Prediction results dictionary
    """
    url = f"{API_BASE}/predict/all"
    
    payload = {
        "driver": driver,
        "circuit": config["circuit"],
        "year": config["year"],
        **config["weather"]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP {response.status_code} for {driver}")
        print(f"   Payload sent: {payload}")
        try:
            error_detail = response.json()
            print(f"   API says: {error_detail}")
        except:
            print(f"   Raw response: {response.text[:500]}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed for {driver}: {e}")
        return None


def predict_full_grid(drivers: List[str], config: Dict) -> pd.DataFrame:
    """
    Predict outcomes for all drivers with context enhancements.
    
    Args:
        drivers: List of driver abbreviations
        config: Race configuration dictionary
        
    Returns:
        DataFrame with predictions for all drivers
    """
    results = []
    
    # Load practice data
    practice_data = load_practice_data(
        config["circuit"], 
        config["is_sprint_weekend"]
    )
    
    # Load recent form
    recent_form = load_recent_form(year=config["year"], lookback_races=5)
    
    weekend_type = "SPRINT" if config["is_sprint_weekend"] else "NORMAL"
    print(f"\n🏁 Predicting {config['circuit']} {config['year']} ({weekend_type} WEEKEND)")
    print(f"   Weather: {config['weather']['avg_air_temp']:.0f}°C air, {config['weather']['avg_track_temp']:.0f}°C track")
    print(f"   Grid: {len(drivers)} drivers\n")
    
    for driver in drivers:
        print(f"   {driver:3s} ({TEAM_MAPPING.get(driver, 'Unknown'):12s}): ", end="")
        
        # Get base prediction from API
        base_pred = predict_driver(driver, config)
        
        if not base_pred:
            print("❌")
            continue
        
        # Get practice baseline
        practice_pos = get_practice_baseline(
            driver, 
            practice_data, 
            config["is_sprint_weekend"]
        )
        
        # Get team adjustments
        team_adj = get_team_adjustment(driver)
        
        # Enhance prediction
        pred = enhance_prediction_with_context(
            base_pred, 
            driver, 
            practice_pos, 
            team_adj, 
            recent_form
        )
        
        # Get recent form data for this driver
        form_delta = None
        recent_avg_pos = None
        if recent_form is not None and driver in recent_form['driver'].values:
            driver_form = recent_form[recent_form['driver'] == driver].iloc[0]
            form_delta = driver_form['form_delta']
            recent_avg_pos = driver_form['recent_avg_position']
        
        results.append({
            'driver': driver,
            'team': TEAM_MAPPING.get(driver, "Unknown"),
            'practice_avg_position': practice_pos if practice_pos else None,
            'recent_form_avg': recent_avg_pos,
            'form_delta': form_delta,
            'will_make_q3': pred['q3']['will_make_q3'],
            'q3_probability': pred['q3']['probability'],
            'q3_confidence': pred['q3']['confidence'],
            'will_make_top3': pred['top3']['will_make_top3'],
            'top3_probability': pred['top3']['probability'],
            'top3_confidence': pred['top3']['confidence'],
            'predicted_round': pred['round']['predicted_round'],
            'q1_prob': pred['round']['probabilities'].get('Q1', 0.0),
            'q2_prob': pred['round']['probabilities'].get('Q2', 0.0),
            'q3_prob_round': pred['round']['probabilities'].get('Q3', 0.0),
            'round_confidence': pred['round']['confidence']
        })
    
    return pd.DataFrame(results)


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_predictions(df: pd.DataFrame, config: Dict) -> None:
    """
    Print formatted prediction results.
    
    Args:
        df: DataFrame with predictions
        config: Race configuration dictionary
    """
    weekend_type = "SPRINT WEEKEND" if config["is_sprint_weekend"] else "NORMAL WEEKEND"
    
    print("\n" + "="*90)
    print(f"🏁 {config['circuit'].upper()} {config['year']} - QUALIFYING PREDICTIONS ({weekend_type})")
    print("="*90)
    
    # Sort by Q3 probability
    df_sorted = df.sort_values('q3_probability', ascending=False).reset_index(drop=True)
    
    print("\n🏆 TOP 10 PREDICTION (Q3)")
    print("-" * 90)
    print(f"{'Pos':<5} {'Driver':<8} {'Team':<15} {'Practice':<10} {'Q3 Prob':<12} {'Confidence':<12}")
    print("-" * 90)
    
    for idx, row in df_sorted.head(10).iterrows():
        practice_str = f"P{row['practice_avg_position']:.1f}" if pd.notna(row['practice_avg_position']) else "-"
        print(f"{idx+1:<5} {row['driver']:<8} {row['team']:<15} {practice_str:<10} {row['q3_probability']:>6.1%}      {row['q3_confidence']:<12}")
    
    print("\n🥉 DRIVERS 11-20 (Q2/Q1)")
    print("-" * 90)
    print(f"{'Pos':<5} {'Driver':<8} {'Team':<15} {'Practice':<10} {'Round':<10} {'Q3 Prob':<12}")
    print("-" * 90)
    
    for idx, row in df_sorted.tail(10).iterrows():
        practice_str = f"P{row['practice_avg_position']:.1f}" if pd.notna(row['practice_avg_position']) else "-"
        print(f"{idx+1:<5} {row['driver']:<8} {row['team']:<15} {practice_str:<10} {row['predicted_round']:<10} {row['q3_probability']:>6.1%}")
    
    print("\n🏁 TOP 3 PREDICTION (PODIUM)")
    print("-" * 90)
    print(f"{'Pos':<5} {'Driver':<8} {'Team':<15} {'Top 3?':<10} {'Probability':<12}")
    print("-" * 90)
    
    df_top3 = df.sort_values('top3_probability', ascending=False).reset_index(drop=True)
    for idx, row in df_top3.head(10).iterrows():
        top3_status = "🥇 YES" if idx < 3 else "❌ NO"
        print(f"{idx+1:<5} {row['driver']:<8} {row['team']:<15} {top3_status:<10} {row['top3_probability']:>6.1%}")
    
    print("\n📊 ENHANCEMENTS APPLIED")
    print("-" * 90)
    rookies_count = len([d for d in df['driver'] if d in ROOKIES])
    team_changes_count = len([d for d in df['driver'] if d in TEAM_CHANGES])
    practice_data_count = df['practice_avg_position'].notna().sum()
    form_data_count = df['form_delta'].notna().sum()
    
    print(f"   Rookies with team baseline: {rookies_count}")
    print(f"   Team changes adjusted: {team_changes_count}")
    print(f"   Drivers with practice data: {practice_data_count}/{len(df)}")
    print(f"   Drivers with recent form: {form_data_count}/{len(df)}")
    
    print("\n" + "="*90)


def save_predictions(df: pd.DataFrame, config: Dict):
    """
    Save predictions to CSV.
    
    Args:
        df: DataFrame with predictions
        config: Race configuration dictionary
    """
    # Clean circuit name for filename
    circuit_name = config["circuit"].replace(" ", "_").replace("Grand_Prix", "GP")
    filename = f"{circuit_name}_{config['year']}_predictions.csv"
    
    df.to_csv(f'{pdir}/{filename}', index=False)
    print(f"\n💾 Predictions saved to: {pdir}/{filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution."""
    
    # Check API is running
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        if response.status_code != 200:
            print("❌ API is not healthy. Start it with:")
            print("   cd api && uvicorn main:app --reload")
            return
    except requests.exceptions.RequestException:
        print("❌ API is not running. Start it with:")
        print("   cd api && uvicorn main:app --reload")
        return
    
    print("✅ API is running")
    
    # Get predictions
    predictions_df = predict_full_grid(DRIVERS_2025, RACE_CONFIG)
    
    if len(predictions_df) == 0:
        print("❌ No predictions generated")
        return
    
    # Format and display
    format_predictions(predictions_df, RACE_CONFIG)
    
    # Save to CSV
    save_predictions(predictions_df, RACE_CONFIG)
    
    print("\n✅ Predictions complete!")
    print("\n💡 To predict for next race:")
    print("   1. Update RACE_CONFIG at the top of this file")
    print("   2. Set circuit name, year, sprint status, and weather")
    print("   3. Run: python race_prediction_v2.py")


if __name__ == "__main__":
    main()