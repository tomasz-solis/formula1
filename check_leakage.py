"""
Script to check for data leakage in historical features.

This checks whether 2024 predictions use 2024 data in their historical features,
which would be leakage since we're testing on 2024.
"""

import pandas as pd
import numpy as np

def check_leakage():
    """Check for potential data leakage in the feature dataset."""
    
    print("=" * 70)
    print("🔍 DATA LEAKAGE DETECTION")
    print("=" * 70)
    
    # Load the features
    df = pd.read_parquet('data/features/ml_features_2022_2025.parquet')
    
    # Filter to qualifying sessions only
    df_qual = df[df['qualifying_position'].notna()].copy()
    
    print(f"\n✅ Loaded {len(df_qual)} qualifying sessions")
    print(f"   Years: {sorted(df_qual['year'].unique())}")
    print(f"   Train years (2022-2023): {len(df_qual[df_qual['year'] < 2024])} sessions")
    print(f"   Test year (2024): {len(df_qual[df_qual['year'] == 2024])} sessions")
    
    # Check 1: Circuit history for 2024 races
    print("\n" + "=" * 70)
    print("CHECK 1: Circuit History for 2024 Races")
    print("=" * 70)
    
    # For each circuit in 2024, check if circuit_avg_position could include 2024 data
    circuits_2024 = df_qual[df_qual['year'] == 2024]['event'].unique()
    
    for circuit in circuits_2024[:3]:  # Check first 3 circuits
        circuit_data = df_qual[df_qual['event'] == circuit].copy()
        circuit_2024 = circuit_data[circuit_data['year'] == 2024]
        circuit_history = circuit_data[circuit_data['year'] < 2024]
        
        if len(circuit_2024) > 0:
            sample = circuit_2024.iloc[0]
            print(f"\n📍 {circuit}:")
            print(f"   2024 circuit_avg_position: {sample.get('circuit_avg_position', 'N/A'):.2f}")
            print(f"   Historical races available: {len(circuit_history)} (years: {sorted(circuit_history['year'].unique())})")
            
            # Calculate what the average SHOULD be (using only 2022-2023)
            if len(circuit_history) > 0 and 'qualifying_position' in circuit_history.columns:
                expected_avg = circuit_history.groupby('driver')['qualifying_position'].mean().mean()
                print(f"   Expected avg from 2022-2023 data: {expected_avg:.2f}")
    
    # Check 2: Recent form for early 2024 races
    print("\n" + "=" * 70)
    print("CHECK 2: Recent Form for Early 2024 Races")
    print("=" * 70)
    
    # Get 2024 races in order
    races_2024 = df_qual[df_qual['year'] == 2024].copy()
    
    # Try to determine round order
    if 'EventDate' in races_2024.columns:
        races_2024 = races_2024.sort_values('EventDate')
        races_2024['round'] = races_2024.groupby('event').ngroup() + 1
    else:
        # Fallback to event order
        races_2024['round'] = races_2024.groupby('event').ngroup() + 1
    
    # Check first 3 rounds
    for round_num in [1, 2, 3]:
        round_data = races_2024[races_2024['round'] == round_num]
        if len(round_data) > 0:
            event_name = round_data['event'].iloc[0]
            has_recent = round_data['recent_avg_position'].notna().sum()
            total = len(round_data)
            
            print(f"\n🏁 Round {round_num} - {event_name}:")
            print(f"   Drivers with recent_avg_position: {has_recent}/{total}")
            
            if round_num == 1:
                if has_recent > 0:
                    print(f"   ✅ GOOD: Uses 2023 data for recent form")
                else:
                    print(f"   ⚠️  WARNING: No recent form data (might be expected)")
            else:
                if has_recent > 0:
                    sample = round_data[round_data['recent_avg_position'].notna()].iloc[0]
                    print(f"   Sample recent_avg_position: {sample['recent_avg_position']:.2f}")
                    print(f"   ⚠️  QUESTION: Does this use 2024 Rounds 1-{round_num-1}?")
                    print(f"   If YES → LEAKAGE (using test set to predict test set)")
    
    # Check 3: Within-year dependencies
    print("\n" + "=" * 70)
    print("CHECK 3: Within-Year Dependencies in Test Set")
    print("=" * 70)
    
    # Check if predictions for later 2024 races depend on earlier 2024 races
    has_recent_2024 = races_2024['recent_avg_position'].notna().sum()
    total_2024 = len(races_2024)
    
    print(f"\n2024 races with recent_avg_position: {has_recent_2024}/{total_2024} ({100*has_recent_2024/total_2024:.1f}%)")
    
    if has_recent_2024 > 0:
        print("\n🚨 POTENTIAL ISSUE:")
        print("   If recent_avg_position for 2024 races uses earlier 2024 races,")
        print("   this creates a dependency chain in the test set.")
        print("   ")
        print("   Example: Predicting Monaco 2024 (Round 8)")
        print("   - Uses Rounds 3-7 of 2024 for recent_avg_position")
        print("   - But Rounds 3-7 are ALSO in the test set")
        print("   - This means we need to predict Rounds 3-7 FIRST")
        print("   - Error compounds: bad prediction → affects next prediction")
        print("\n   ✅ SOLUTION:")
        print("   For test set evaluation, compute recent_avg_position using")
        print("   ONLY training data (2022-2023), NOT other 2024 races.")
    
    # Check 4: Feature distributions
    print("\n" + "=" * 70)
    print("CHECK 4: Feature Distribution Comparison")
    print("=" * 70)
    
    train = df_qual[df_qual['year'] < 2024]
    test = df_qual[df_qual['year'] == 2024]
    
    key_features = ['circuit_avg_position', 'recent_avg_position', 'form_trend']
    
    for feature in key_features:
        if feature in df_qual.columns:
            train_mean = train[feature].mean()
            test_mean = test[feature].mean()
            train_std = train[feature].std()
            test_std = test[feature].std()
            
            print(f"\n{feature}:")
            print(f"   Train: μ={train_mean:.2f}, σ={train_std:.2f}")
            print(f"   Test:  μ={test_mean:.2f}, σ={test_std:.2f}")
            
            # If means are very similar, might indicate leakage
            if abs(train_mean - test_mean) < 0.5:
                print(f"   ⚠️  Very similar means - check if this is expected")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print("\n✅ Train/Test split is time-based (2022-2023 vs 2024)")
    print("\n⚠️  POTENTIAL ISSUES TO INVESTIGATE:")
    print("1. Does circuit_avg_position for 2024 include any 2024 data?")
    print("2. Does recent_avg_position for 2024 use other 2024 races?")
    print("3. If #2 is YES, do we need to recompute for fair evaluation?")
    print("\n🔧 NEXT STEP:")
    print("Review helpers/historical_features.py to see how these features")
    print("are computed and verify they use only past data.")

if __name__ == "__main__":
    check_leakage()