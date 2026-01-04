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

# Primary Contributory Cause ordered by perceived danger (low → high)
prim_contributory_cause_risk_map = {
    # Lowest / unknown applicability
    "NOT APPLICABLE": 1,
    "UNABLE TO DETERMINE": 2,
    "RELATED TO BUS STOP": 3,
    "OBSTRUCTED CROSSWALKS": 4,

    # External factors and environment
    "ANIMAL": 5,
    "EVASIVE ACTION DUE TO ANIMAL, OBJECT, NONMOTORIST": 6,
    "WEATHER": 7,
    "VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)": 8,
    "EQUIPMENT - VEHICLE CONDITION": 9,
    "ROAD CONSTRUCTION/MAINTENANCE": 10,
    "ROAD ENGINEERING/SURFACE/MARKING DEFECTS": 11,

    # Driver condition/skill
    "PHYSICAL CONDITION OF DRIVER": 12,
    "DRIVING SKILLS/KNOWLEDGE/EXPERIENCE": 13,

    # Generally lower-risk infractions
    "TURNING RIGHT ON RED": 14,
    "BICYCLE ADVANCING LEGALLY ON RED LIGHT": 15,
    "MOTORCYCLE ADVANCING LEGALLY ON RED LIGHT": 16,
    "IMPROPER BACKING": 17,
    "IMPROPER TURNING/NO SIGNAL": 18,
    "IMPROPER LANE USAGE": 19,
    "IMPROPER OVERTAKING/PASSING": 20,
    "FOLLOWING TOO CLOSELY": 21,
    "FAILING TO YIELD RIGHT-OF-WAY": 22,

    # Sign/marking violations
    "DISREGARDING ROAD MARKINGS": 23,
    "DISREGARDING OTHER TRAFFIC SIGNS": 24,
    "DISREGARDING YIELD SIGN": 25,
    "DISREGARDING STOP SIGN": 26,
    "DISREGARDING TRAFFIC SIGNALS": 27,

    # Distractions
    "CELL PHONE USE OTHER THAN TEXTING": 28,
    "DISTRACTION - FROM OUTSIDE VEHICLE": 29,
    "DISTRACTION - FROM INSIDE VEHICLE": 30,
    "DISTRACTION - OTHER ELECTRONIC DEVICE (NAVIGATION DEVICE, DVD PLAYER, ETC.)": 31,
    "TEXTING": 32,

    # Speed and aggressive behavior
    "FAILING TO REDUCE SPEED TO AVOID CRASH": 33,
    "EXCEEDING SAFE SPEED FOR CONDITIONS": 34,
    "EXCEEDING AUTHORIZED SPEED LIMIT": 35,
    "OPERATING VEHICLE IN ERRATIC, RECKLESS, CARELESS, NEGLIGENT OR AGGRESSIVE MANNER": 36,

    # Very high-risk behaviors
    "PASSING STOPPED SCHOOL BUS": 37,
    "DRIVING ON WRONG SIDE/WRONG WAY": 38,
    "HAD BEEN DRINKING (USE WHEN ARREST IS NOT MADE)": 39,
    "UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)": 40,
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


"""
Weather conditions were grouped and ordered by perceived crash dangerousness. Clear and overcast conditions represent minimal risk. Precipitation, visibility loss, wind, and winter conditions progressively increase crash likelihood and severity. Each category receives a unique ordinal value while preserving the risk-based ordering.
"""
weather_risk_map = {
    # Lowest perceived crash danger
    "CLEAR": 1,
    "CLOUDY/OVERCAST": 2,

    # Low–moderate perceived crash danger
    "RAIN": 3,
    "OTHER": 4,

    # Moderate perceived crash danger
    "FOG/SMOKE/HAZE": 5,
    "BLOWING SAND, SOIL, DIRT": 6,
    "SEVERE CROSS WIND GATE": 7,

    # High perceived crash danger
    "SNOW": 8,
    "SLEET/HAIL": 9,

    # Highest perceived crash danger
    "BLOWING SNOW": 10,
    "FREEZING RAIN/DRIZZLE": 11,
}

"""
Trafficway types were grouped and ordered by perceived crash dangerousness. Limited-access and low-speed areas rank lowest. Mid-level values reflect standard road configurations with moderate conflict points. Highest values correspond to intersections and complex geometries with increased conflict density. Each category is assigned a unique ordinal value while preserving this risk-based ordering.
"""
trafficway_danger_map = {
    # Lowest perceived crash danger
    "ALLEY": 1,
    "DRIVEWAY": 2,
    "PARKING LOT": 3,

    # Low–moderate perceived crash danger
    "DIVIDED - W/MEDIAN BARRIER": 4,
    "ONE-WAY": 5,
    "RAMP": 6,

    # Moderate perceived crash danger
    "NOT DIVIDED": 7,
    "TRAFFIC ROUTE": 8,
    "CENTER TURN LANE": 9,
    "DIVIDED - W/MEDIAN (NOT RAISED)": 10,
    "NOT REPORTED": 11,
    "UNKNOWN": 12,
    "UNKNOWN INTERSECTION TYPE": 13,
    "OTHER": 14,

    # High perceived crash danger
    "FOUR WAY": 15,
    "T-INTERSECTION": 16,
    "Y-INTERSECTION": 17,
    "L-INTERSECTION": 18,
    "FIVE POINT, OR MORE": 19,
    "ROUNDABOUT": 20,
}

"""
Crash types were grouped and ordered by perceived severity based on typical impact forces and injury risk. Minor contact and low-energy collisions rank lowest. Multi-directional impacts and fixed objects increase severity. Head-on, overturning, train, and vulnerable road user crashes represent the highest severity. Unique ordinal values preserve this risk-based ordering.
"""
crash_severity_map = {
    # Lowest perceived crash severity
    "SIDESWIPE SAME DIRECTION": 1,
    "PARKED MOTOR VEHICLE": 2,
    "REAR TO REAR": 3,
    "OTHER OBJECT": 4,

    # Low–moderate perceived crash severity
    "REAR END": 5,
    "REAR TO FRONT": 6,
    "REAR TO SIDE": 7,
    "ANIMAL": 8,
    "OTHER NONCOLLISION": 9,

    # Moderate perceived crash severity
    "TURNING": 10,
    "ANGLE": 11,
    "FIXED OBJECT": 12,

    # High perceived crash severity
    "SIDESWIPE OPPOSITE DIRECTION": 13,

    # Highest perceived crash severity
    "HEAD ON": 14,
    "OVERTURNED": 15,
    "TRAIN": 16,
    "PEDESTRIAN": 17,
    "PEDALCYCLIST": 18,
}


"""
Traffic control devices were grouped and ordered by perceived crash dangerousness. Lower values correspond to passive or advisory controls. Higher values correspond to active controls and railroad interactions where crash risk and severity are typically higher. The final encoding preserves this ordering while assigning a unique level to each device.
"""
traffic_control_device_group_map = {
    # Lower perceived crash danger
    "OTHER WARNING SIGN": 1,
    "OTHER REG. SIGN": 2,
    "NO PASSING": 3,
    "BICYCLE CROSSING SIGN": 4,
    "PEDESTRIAN CROSSING SIGN": 5,
    "SCHOOL ZONE": 6,

    # Moderate perceived crash danger
    "STOP SIGN/FLASHER": 7,
    "YIELD": 8,

    # High perceived crash danger
    "TRAFFIC SIGNAL": 9,
    "FLASHING CONTROL SIGNAL": 10,
    "POLICE/FLAGMAN": 11,

    # Highest perceived crash danger
    "RAILROAD CROSSING GATE": 12,
    "RR CROSSING SIGN": 13,
    "OTHER RAILROAD CROSSING": 14,
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
    train_df = df.iloc[:split_index].copy().reset_index(drop=True)
    test_df = df.iloc[split_index:].copy().reset_index(drop=True)
    
    print(f"Data Split: {len(train_df)} Train rows, {len(test_df)} Test rows.")

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

        # B5. Primary Contributory Cause -> ordinal danger levels (numeric)
        if "prim_contributory_cause" in data.columns:
            data["prim_contributory_cause"] = data["prim_contributory_cause"].apply(
                lambda v: prim_contributory_cause_risk_map.get(safe_upper(v), np.nan)
            )
            data["prim_contributory_cause"] = pd.to_numeric(data["prim_contributory_cause"], errors='coerce')

        # B6. Traffic Control Device -> group levels (numeric)
        if "traffic_control_device" in data.columns:
            data["traffic_control_device_group"] = data["traffic_control_device"].apply(
                lambda v: traffic_control_device_group_map.get(safe_upper(v), np.nan)
            )
            data["traffic_control_device_group"] = pd.to_numeric(data["traffic_control_device_group"], errors='coerce')
            data = data.drop(columns=["traffic_control_device"])  # drop original textual

        # No one-hot encoding applied; keep all derived groups as numeric

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