"""
Diagnostic script to find which merge is creating duplicate rows.

This will test each feature dataframe for duplicates on their merge keys.
"""

import pandas as pd
import sys

print("🔍 DUPLICATE DIAGNOSIS")
print("=" * 80)

# Load the problematic data
try:
    df = pd.read_parquet('data/features/ml_features_2022_2025.parquet')
    print(f"✅ Loaded: {df.shape}")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    sys.exit(1)

# Test 1: Check for duplicates in base data
print("\n📊 TEST 1: Base data duplicates")
print("-" * 80)
base_dups = df.duplicated(subset=['year', 'event', 'driver'], keep=False)
dup_count = base_dups.sum()
print(f"Duplicates on ['year', 'event', 'driver']: {dup_count}")

if dup_count > 0:
    print("\n⚠️ FOUND DUPLICATES IN BASE DATA!")
    print("\nExample duplicates:")
    dup_examples = df[base_dups].sort_values(['driver', 'event', 'year']).head(20)
    print(dup_examples[['driver', 'event', 'year', 'qualifying_position']])
    
    # Show which columns differ
    print("\n🔍 Which columns have different values in duplicates?")
    test_driver = dup_examples.iloc[0]['driver']
    test_event = dup_examples.iloc[0]['event']
    test_year = dup_examples.iloc[0]['year']
    
    test_rows = df[
        (df['driver'] == test_driver) &
        (df['event'] == test_event) &
        (df['year'] == test_year)
    ]
    
    print(f"\nRows for {test_driver}, {test_event}, {test_year}:")
    print(f"Total rows: {len(test_rows)}")
    
    # Check each column for variation
    varying_cols = []
    for col in test_rows.columns:
        if test_rows[col].nunique() > 1:
            varying_cols.append(col)
    
    print(f"\nColumns with different values: {len(varying_cols)}")
    print(varying_cols)
    
    if varying_cols:
        print("\nSample values from varying columns:")
        print(test_rows[['driver', 'event', 'year'] + varying_cols[:5]])

# Test 2: Check feature dataframes for duplicates on their merge keys
print("\n\n📊 TEST 2: Feature-level duplicate check")
print("-" * 80)
print("Re-computing features to check merge key uniqueness...\n")

try:
    # Load driver profiles
    driver_profiles = []
    for year in [2022, 2023, 2024, 2025]:
        try:
            year_df = pd.read_csv(f'data/driver/driver_profiles_{year}.csv')
            year_df['year'] = year
            driver_profiles.append(year_df)
        except:
            pass
    
    if driver_profiles:
        driver_df = pd.concat(driver_profiles, ignore_index=True)
        print(f"✅ Loaded driver profiles: {driver_df.shape}")
        
        # Check for session-level duplicates
        print("\nChecking session-level data...")
        session_dups = driver_df.duplicated(subset=['year', 'event', 'session_type', 'driver'], keep=False)
        if session_dups.sum() > 0:
            print(f"⚠️ FOUND {session_dups.sum()} session-level duplicates!")
            print("These will multiply during aggregation")
        else:
            print("✅ No session-level duplicates")
        
        # Aggregate and check
        print("\nAggregating to driver-race level...")
        agg_dict = {
            'session_date': 'first',
            'team': 'first',
        }
        
        # Add optional columns
        for col in ['max_throttle_ratio', 'avg_rainfall']:
            if col in driver_df.columns:
                agg_dict[col] = 'mean'
        
        driver_agg = driver_df.groupby(
            ['year', 'event', 'driver'],
            as_index=False
        ).agg(agg_dict)
        
        print(f"After aggregation: {driver_agg.shape}")
        
        # Check for duplicates after aggregation
        agg_dups = driver_agg.duplicated(subset=['year', 'event', 'driver'], keep=False)
        if agg_dups.sum() > 0:
            print(f"❌ STILL {agg_dups.sum()} duplicates after aggregation!")
        else:
            print("✅ No duplicates after aggregation")
            
except Exception as e:
    print(f"❌ Failed to test feature computation: {e}")

# Test 3: Specific merge key tests
print("\n\n📊 TEST 3: Merge key cardinality")
print("-" * 80)

# Test circuit features (should be 1 per event-year)
circuit_cols = [c for c in df.columns if 'circuit' in c.lower() and 'avg' not in c]
if circuit_cols:
    print("\n🏁 Circuit features:")
    circuit_dups = df.duplicated(subset=['event', 'year'] + circuit_cols[:1], keep=False)
    unique_circuits = df[['event', 'year']].drop_duplicates().shape[0]
    print(f"   Unique (event, year): {unique_circuits}")
    print(f"   Total rows: {len(df)}")
    print(f"   Ratio: {len(df) / unique_circuits:.1f}x")
    if len(df) / unique_circuits > 20:
        print("   ⚠️ This looks suspicious - should be ~20 drivers per event")

# Test weather features (should be 1 per driver-year)
weather_cols = [c for c in df.columns if 'wet' in c or 'dry' in c]
if weather_cols:
    print("\n🌧️ Weather features:")
    unique_driver_years = df[['driver', 'year']].drop_duplicates().shape[0]
    print(f"   Unique (driver, year): {unique_driver_years}")
    print(f"   Total rows: {len(df)}")
    print(f"   Ratio: {len(df) / unique_driver_years:.1f}x")
    if len(df) / unique_driver_years > 25:
        print("   ⚠️ This looks suspicious - should be ~20-24 events per year")

print("\n" + "=" * 80)
print("✅ DIAGNOSIS COMPLETE")
print("\nNEXT STEPS:")
print("1. If duplicates found in base data → problem is in the merge operations")
print("2. If duplicates in session-level data → problem is in data collection")
print("3. If cardinality ratios are wrong → identify which feature is multiplying rows")
