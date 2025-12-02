import pandas as pd
import numpy as np
from math import pi, sin, cos

# Configuration
INPUT_PATH = "traffic_accidents.csv"
OUTPUT_TRAIN_PATH = "traffic_accidents_train.csv"
OUTPUT_TEST_PATH = "traffic_accidents_test.csv"

# !!! CRITICAL FOR LAB 3: Do not drop unknown rows yet. 
# We need them for the MVI comparison step.
DROP_UNKNOWN_ROWS = False 

# Variable Groups
NUMERIC_WITH_CONVERSION = ["crash_date"]
CYCLICAL = ["crash_hour", "crash_day_of_week", "crash_month"]
ORDINAL = ["first_crash_type", "traffic_control_device", "intersection_related_i", 
           "road_defect", "roadway_surface_cond", "trafficway_type",
           "alignment", "lighting_condition", "crash_type", "damage", 
           "most_severe_injury", "prim_contributory_cause"]
NOMINAL = ["weather_condition"]

# Static Mappings (Pre-defined)
ordinal_mappings = {
    "intersection_related_i": {"N": 0, "Y": 1},
    "crash_type": {"NO INJURY / DRIVE AWAY": 0, "INJURY AND / OR TOW DUE TO CRASH": 1},
    "lighting_condition": {
        "DAYLIGHT": 1, "DAWN": 2, "DUSK": 3, 
        "DARKNESS, LIGHTED ROAD": 4, "DARKNESS": 5,
    },
    "damage": {"$500 OR LESS": 1, "$501 - $1,500": 2, "OVER $1,500": 3},
    "most_severe_injury": {
        "NO INDICATION OF INJURY": 1, "REPORTED, NOT EVIDENT": 2,
        "NONINCAPACITATING INJURY": 3, "INCAPACITATING INJURY": 4, "FATAL": 5,
    },
    # These will be filled dynamically using ONLY training data
    "traffic_control_device": {},
    "first_crash_type": {},
    "trafficway_type": {},
    "alignment": {},
    "roadway_surface_cond": {},
    "road_defect": {},
    "prim_contributory_cause": {},
}

def compute_injury_proportions(df_train, feature, target_col="crash_type", injury_label="INJURY AND / OR TOW DUE TO CRASH"):
    """
    Computes encoding map based on TRAINING data only.
    """
    if feature not in df_train.columns or target_col not in df_train.columns:
        return {}
    
    tmp = df_train[[feature, target_col]].dropna()
    counts = tmp.groupby([feature, target_col]).size().unstack(fill_value=0)
    
    if injury_label not in counts.columns:
        counts[injury_label] = 0
        
    totals = counts.sum(axis=1)
    proportions = (counts[injury_label] / totals).fillna(0.0).round(2)
    return proportions.to_dict()

def encode_cyclic_variables(df, cols):
    for c in cols:
        if c in df.columns:
            x_max = df[c].max()
            df[c + "_sin"] = df[c].apply(lambda x: round(sin(2 * pi * x / x_max), 6))
            df[c + "_cos"] = df[c].apply(lambda x: round(cos(2 * pi * x / x_max), 6))
            df = df.drop(columns=[c])
    return df

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH)

    # 1. Sort by Date (Crucial for Time Series Split)
    if "crash_date" in df.columns:
        df['crash_date'] = pd.to_datetime(df['crash_date'])
        df = df.sort_values('crash_date')
    
    # 2. Split Data (70% Train, 30% Test) BEFORE Encoding (prevents leakage)
    split_index = int(len(df) * 0.7)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    print(f"Data Split: {len(train_df)} Train rows, {len(test_df)} Test rows.")

    # 3. Learn Dynamic Mappings from TRAIN set only
    dynamic_cols = ["first_crash_type", "traffic_control_device", "trafficway_type", 
                   "alignment", "roadway_surface_cond", "road_defect", "prim_contributory_cause"]
    
    print("Computing dynamic mappings from Training data...")
    for col in dynamic_cols:
        if col in train_df.columns:
            # We use the raw 'train_df' to learn the pattern
            mapping = compute_injury_proportions(train_df, col)
            ordinal_mappings[col] = mapping

    # Helper function to apply all encodings
    def apply_encoding(dataset):
        data = dataset.copy()
        
        # A. Ordinal Encoding
        for col in ORDINAL:
            actual_col = col if col in data.columns else ("lightning_condition" if "lightning_condition" in data.columns else None)
            if actual_col:
                mapping = ordinal_mappings.get(actual_col) or ordinal_mappings.get(col, {})
                # If a value in Test was never seen in Train, it might become NaN or keep string. 
                # For this lab, we will force replace.
                data[actual_col] = data[actual_col].replace(mapping)
                # Convert any remaining unmapped strings to NaN (treat as missing)
                data[actual_col] = pd.to_numeric(data[actual_col], errors='coerce')

        # B. Nominal (One-Hot) Encoding
        nominal_existing = [c for c in NOMINAL if c in data.columns]
        if nominal_existing:
            data = pd.get_dummies(data, columns=nominal_existing, dummy_na=False)

        # C. Cyclical Encoding
        data = encode_cyclic_variables(data, CYCLICAL)
        
        # D. Date Conversion
        if "crash_date" in data.columns:
             data["crash_date"] = (data["crash_date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

        return data

    # 4. Apply Mappings to both Train and Test
    print("Applying encodings...")
    train_encoded = apply_encoding(train_df)
    test_encoded = apply_encoding(test_df)
    
    # Ensure columns match (One-Hot might create different cols if test has different weather)
    # We align them to the Train columns, filling missing cols with 0
    train_cols = train_encoded.columns
    test_encoded = test_encoded.reindex(columns=train_cols, fill_value=0)

    # 5. Save
    train_encoded.to_csv(OUTPUT_TRAIN_PATH, index=False)
    test_encoded.to_csv(OUTPUT_TEST_PATH, index=False)
    
    print(f"Done! Created {OUTPUT_TRAIN_PATH} and {OUTPUT_TEST_PATH}")

if __name__ == "__main__":
    main()