import pandas as pd
import numpy as np
from math import pi, sin, cos

# Configuration
INPUT_PATH = "traffic_accidents.csv"
OUTPUT_TRAIN_PATH = "traffic_accidents_train.csv"
OUTPUT_TEST_PATH = "traffic_accidents_test.csv"

DROP_UNKNOWN_ROWS = False 

# Variable Groups
NUMERIC_WITH_CONVERSION = ["crash_date"]

CYCLICAL = ["crash_hour", "crash_day_of_week", "crash_month"]

# Columns that will remain ordinal (fixed, non-target encodings)
ORDINAL = [
    "intersection_related_i",
    "lighting_condition",
    "crash_type",
    "damage",
]

# Previously nominal; now handled via custom encodings below
NOMINAL = []

# Static Mappings (Pre-defined)
ordinal_mappings = {
    "intersection_related_i": {"N": 0, "Y": 1},
    "crash_type": {"NO INJURY / DRIVE AWAY": 0, "INJURY AND / OR TOW DUE TO CRASH": 1},
    "lighting_condition": {
        "DAYLIGHT": 1, "DAWN": 2, "DUSK": 3, 
        "DARKNESS, LIGHTED ROAD": 4, "DARKNESS": 5,
    },
    "damage": {"$500 OR LESS": 1, "$501 - $1,500": 2, "OVER $1,500": 3},
    # Additional ordinal mappings per requirements
    "roadway_surface_cond": {
        "OTHER": 0,
        "DRY": 1,
        "SAND, MUD, DIRT": 2,
        "WET": 3,
        "SNOW OR SLUSH": 4,
        "ICE": 5,
    },
    "road_defect": {
        "OTHER": 0,
        "NO DEFECTS": 1,
        "WORN SURFACE": 2,
        "SHOULDER DEFECT": 3,
        "RUT, HOLES": 4,
        "DEBRIS ON ROADWAY": 5,
    },
}

# Columns to drop entirely
DROP_COLS = [
    "most_severe_injury",
    "injuries_total",
    "injuries_fatal",
    "injuries_incapacitating",
    "injuries_non_incapacitating",
    "injuries_reported_not_evident",
    "injuries_no_indication",
]

def safe_upper(val):
    if pd.isna(val):
        return ""
    try:
        return str(val).upper()
    except Exception:
        return ""

def encode_alignment_features(series):
    # Create two derived variables from 'alignment'
    curv_vals = []
    grade_vals = []
    for v in series.fillna(""):
        u = safe_upper(v)
        # Variable 1: CURVE -> 0, STRAIGHT -> 1
        if "CURVE" in u:
            curv_vals.append(0)
        elif "STRAIGHT" in u:
            curv_vals.append(1)
        else:
            curv_vals.append(np.nan)
        # Variable 2: LEVEL -> 0, GRADE -> 1, HILLCREST -> 2
        if "LEVEL" in u:
            grade_vals.append(0)
        elif "GRADE" in u:
            grade_vals.append(1)
        elif "HILLCREST" in u:
            grade_vals.append(2)
        else:
            grade_vals.append(np.nan)
    return pd.Series(curv_vals), pd.Series(grade_vals)

def categorize_prim_cause(val):
    u = safe_upper(val)
    if any(x in u for x in ["PHONE", "TEXTING", "DISTRACTION", "ELECTRONIC"]):
        return "Distracted"
    elif any(x in u for x in ["DISREGARDING", "STOP SIGN", "YIELD", "RED LIGHT"]):
        return "Sign_Signal_Violation"
    elif any(x in u for x in ["SPEED", "FOLLOWING", "ERRATIC", "RECKLESS"]):
        return "Speed_Aggressive"
    elif any(x in u for x in ["ALCOHOL", "DRINKING", "DRUGS", "PHYSICAL"]):
        return "Impairment"
    elif any(x in u for x in ["IMPROPER", "WRONG SIDE", "TURNING"]):
        return "Improper_Maneuver"
    elif any(x in u for x in ["ANIMAL", "WEATHER", "VISION", "ROAD", "OBSTRUCTED"]):
        return "External_Factor"
    else:
        return "Other"

weather_risk_map = {
    "CLEAR": 0,
    "CLOUDY/OVERCAST": 0,
    "RAIN": 1,
    "OTHER": 1,
    "FOG/SMOKE/HAZE": 2,
    "BLOWING SAND, SOIL, DIRT": 2,
    "SEVERE CROSS WIND GATE": 2,
    "SNOW": 3,
    "SLEET/HAIL": 3,
    "BLOWING SNOW": 4,
    "FREEZING RAIN/DRIZZLE": 4,
}

trafficway_danger_map = {
    "ALLEY": 0,
    "DRIVEWAY": 0,
    "PARKING LOT": 0,
    "DIVIDED - W/MEDIAN BARRIER": 1,
    "ONE-WAY": 1,
    "RAMP": 1,
    "NOT DIVIDED": 2,
    "TRAFFIC ROUTE": 2,
    "CENTER TURN LANE": 2,
    "DIVIDED - W/MEDIAN (NOT RAISED)": 2,
    "NOT REPORTED": 2,
    "UNKNOWN": 2,
    "UNKNOWN INTERSECTION TYPE": 2,
    "OTHER": 2,
    "FOUR WAY": 3,
    "T-INTERSECTION": 3,
    "Y-INTERSECTION": 3,
    "L-INTERSECTION": 3,
    "FIVE POINT, OR MORE": 3,
    "ROUNDABOUT": 3,
}

crash_severity_map = {
    "SIDESWIPE SAME DIRECTION": 0,
    "PARKED MOTOR VEHICLE": 0,
    "REAR TO REAR": 0,
    "OTHER OBJECT": 0,
    "REAR END": 1,
    "REAR TO FRONT": 1,
    "REAR TO SIDE": 1,
    "ANIMAL": 1,
    "OTHER NONCOLLISION": 1,
    "TURNING": 2,
    "ANGLE": 2,
    "FIXED OBJECT": 2,
    "SIDESWIPE OPPOSITE DIRECTION": 3,
    "HEAD ON": 4,
    "OVERTURNED": 4,
    "TRAIN": 4,
    "PEDESTRIAN": 4,
    "PEDALCYCLIST": 4,
}

traffic_control_device_group_map = {
    # Level 1
    "OTHER WARNING SIGN": 1,
    "OTHER REG. SIGN": 1,
    "NO PASSING": 1,
    "BICYCLE CROSSING SIGN": 1,
    "PEDESTRIAN CROSSING SIGN": 1,
    "SCHOOL ZONE": 1,
    # Level 2
    "STOP SIGN/FLASHER": 2,
    "YIELD": 2,
    # Level 3
    "TRAFFIC SIGNAL": 3,
    "FLASHING CONTROL SIGNAL": 3,
    "POLICE/FLAGMAN": 3,
    # Level 4
    "RAILROAD CROSSING GATE": 4,
    "RR CROSSING SIGN": 4,
    "OTHER RAILROAD CROSSING": 4,
}

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

    # 3. No target encoding: columns formerly target-encoded will be one-hot encoded
    print("Target encoding removed. Will one-hot encode high-cardinality categoricals.")

    # Helper function to apply all encodings
    def apply_encoding(dataset):
        data = dataset.copy()
        
        # 0. Drop specified columns
        drop_existing = [c for c in DROP_COLS if c in data.columns]
        if drop_existing:
            data = data.drop(columns=drop_existing)
        
        # A. Ordinal Encoding (fixed mappings only)
        for col in ORDINAL:
            if col in data.columns:
                mapping = ordinal_mappings.get(col, {})
                data[col] = data[col].replace(mapping)
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # A.1 Additional Ordinal Mappings specified
        for col in ["roadway_surface_cond", "road_defect"]:
            if col in data.columns and col in ordinal_mappings:
                # Ensure matching by uppercase
                data[col] = data[col].apply(lambda v: ordinal_mappings[col].get(safe_upper(v), np.nan))
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # B. Custom Encodings
        # B1. Alignment -> two variables
        if "alignment" in data.columns:
            curv, grade = encode_alignment_features(data["alignment"])
            data["alignment_curvature"] = pd.to_numeric(curv, errors='coerce')
            data["alignment_grade"] = pd.to_numeric(grade, errors='coerce')
            data = data.drop(columns=["alignment"])  # remove original

        # B2. Weather Condition -> risk levels (numeric)
        if "weather_condition" in data.columns:
            data["weather_condition"] = data["weather_condition"].apply(lambda v: weather_risk_map.get(safe_upper(v), np.nan))
            data["weather_condition"] = pd.to_numeric(data["weather_condition"], errors='coerce')

        # B3. Trafficway Type -> danger levels (numeric)
        if "trafficway_type" in data.columns:
            data["trafficway_type"] = data["trafficway_type"].apply(lambda v: trafficway_danger_map.get(safe_upper(v), np.nan))
            data["trafficway_type"] = pd.to_numeric(data["trafficway_type"], errors='coerce')

        # B4. First Crash Type -> severity levels (numeric)
        if "first_crash_type" in data.columns:
            data["first_crash_type"] = data["first_crash_type"].apply(lambda v: crash_severity_map.get(safe_upper(v), np.nan))
            data["first_crash_type"] = pd.to_numeric(data["first_crash_type"], errors='coerce')

        # B5. Primary Contributory Cause -> grouped then one-hot
        if "prim_contributory_cause" in data.columns:
            data["prim_contributory_cause_group"] = data["prim_contributory_cause"].apply(categorize_prim_cause)
            data = data.drop(columns=["prim_contributory_cause"])  # drop original textual

        # B6. Traffic Control Device -> group levels then one-hot
        if "traffic_control_device" in data.columns:
            data["traffic_control_device_group"] = data["traffic_control_device"].apply(lambda v: traffic_control_device_group_map.get(safe_upper(v), "Other"))
            data = data.drop(columns=["traffic_control_device"])  # drop original textual

        # B7. One-Hot for derived groups
        oh_cols = [c for c in ["prim_contributory_cause_group", "traffic_control_device_group"] if c in data.columns]
        if oh_cols:
            data = pd.get_dummies(data, columns=oh_cols, dummy_na=False)

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