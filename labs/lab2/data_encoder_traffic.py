from pathlib import Path
from math import pi, sin, cos
import pandas as pd

#!/usr/bin/env python3
"""Encode Chicago traffic accidents dataset according to variable types.

Variable groups:
ORDINAL: intersection_related_i, lighting_condition (or lightning_condition), crash_type (must be last column), damage, most_severe_injury
NOMINAL (one-hot): traffic_control_device, weather_condition, first_crash_type, trafficway_type, alignment, roadway_surface_cond, road_defect, prim_contributory_cause
NUMERIC_WITH_CONVERSION: crash_date (timestamp string -> epoch seconds)
NUMERIC_NO_CONVERSION: num_units, injuries_total, injuries_fatal, injuries_incapacitating, injuries_non_incapacitating, injuries_reported_not_evident, injuries_no_indication
CYCLICAL: crash_hour, crash_day_of_week, crash_month (each -> _sin/_cos columns)

Notes:
 - Ordinal variables are encoded via explicit mapping dictionaries with DataFrame.replace.
 - If an expected key is missing (dataset variation), dynamic mapping is generated (alphabetical) and printed for manual refinement.
 - Cyclical variables are dropped after adding *_sin/*_cos.
 - crash_type integer-encoded column placed last in the output.
"""


INPUT_PATH = Path(__file__).parent / ".." / ".." / "classification" / "traffic_accidents.csv"
OUTPUT_PATH = Path(__file__).parent / ".." / ".." / "classification" / "traffic_accidents_encoded.csv"

# Variable groups (for clarity; not all are directly used as lists later)
NUMERIC_NO_CONVERSION = [
    "num_units", "injuries_total", "injuries_fatal", "injuries_incapacitating",
    "injuries_non_incapacitating", "injuries_reported_not_evident", "injuries_no_indication"
]
CYCLICAL = ["crash_hour", "crash_day_of_week", "crash_month"]
NUMERIC_WITH_CONVERSION = ["crash_date"]
ORDINAL = ["first_crash_type", "traffic_control_device", "intersection_related_i", "lighting_condition", "crash_type", "damage", "most_severe_injury"]

NOMINAL = [
    "weather_condition", "first_crash_type", "trafficway_type", "alignment",
    "roadway_surface_cond", "road_defect", "prim_contributory_cause"
]

# !!! To see explanation of first_crash_type and traffic_control_device check in images the distribution of crash_type per variable per class !!!
ordinal_mappings: dict[str, dict] = {
    "intersection_related_i": {"N": 0, "Y": 1},
    # Visibility / lighting: daylight best -> darkest worst
    "lighting_condition": {
        "DAYLIGHT": 1,
        "DAWN": 2,
        "DUSK": 3,
        "DARKNESS, LIGHTED ROAD": 4,
        "DARKNESS": 5,
    },
    "crash_type": {"NO INJURY / DRIVE AWAY": 0, "INJURY AND / OR TOW DUE TO CRASH": 1},
    "damage": {
        "$500 OR LESS": 1,
        "$501 - $1,500": 2,
        "OVER $1,500": 3,
    },
    "most_severe_injury": {
        "NO INDICATION OF INJURY": 1,
        "REPORTED, NOT EVIDENT": 2,
        "NON-INCAPACITATING INJURY": 3,
        "INCAPACITATING INJURY": 4,
        "FATAL": 5,
    },
    "traffic_control_device": {
        "PEDESTRIAN CROSSING SIGN": 0.777,
        "NO PASSING": 0.750,
        "FLASHING CONTROL SIGNAL": 0.607,
        "SCHOOL ZONE": 0.576,
        "OTHER RAILROAD CROSSING": 0.565,
        "YIELD": 0.558,
        "OTHER WARNING SIGN": 0.526,
        "STOP SIGN/FLASHER": 0.497,
        "OTHER": 0.479,
        "DELINEATORS": 0.471,
        "BICYCLE CROSSING SIGN": 0.455,
        "LANE USE MARKING": 0.438,
        "NO CONTROLS": 0.424,
        "TRAFFIC SIGNAL": 0.424,
        "OTHER REG. SIGN": 0.398,
        "RAILROAD CROSSING GATE": 0.385,
        "POLICE/FLAGMAN": 0.317,
        "RR CROSSING SIGN": 0.278,
        "UNKNOWN": 0.275,
    },
    "first_crash_type": {
        "OVERTURNED": 0.906,
        "PEDESTRIAN": 0.896,
        "TRAIN": 0.875,
        "PEDALCYCLIST": 0.756,
        "FIXED OBJECT": 0.717,
        "HEAD ON": 0.677,
        "ANGLE": 0.604,
        "OTHER NONCOLLISION": 0.594,
        "OTHER OBJECT": 0.582,
        "REAR TO SIDE": 0.441,
        "TURNING": 0.418,
        "SIDESWIPE OPPOSITE DIRECTION": 0.355,
        "PARKED MOTOR VEHICLE": 0.321,
        "REAR END": 0.250,
        "ANIMAL": 0.169,
        "REAR TO FRONT": 0.145,
        "SIDESWIPE SAME DIRECTION": 0.141,
        "REAR TO REAR": 0.082,
    },
}


def generate_mapping(values: list) -> dict:
    """Generate alphabetical mapping if no predefined mapping is supplied."""
    return {v: i for i, v in enumerate(sorted(values, key=lambda x: str(x)))}


def encode_cyclic_variables(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        x_max = df[c].max()
        # Avoid division by zero if constant column
        if x_max == 0:
            df[c + "_sin"] = 0.0
            df[c + "_cos"] = 1.0
        else:
            df[c + "_sin"] = df[c].apply(lambda x: round(sin(2 * pi * x / x_max), 6))
            df[c + "_cos"] = df[c].apply(lambda x: round(cos(2 * pi * x / x_max), 6))
    # Drop originals
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing)

def main():
    df = pd.read_csv(INPUT_PATH)

    # crash_date -> epoch seconds (NUMERIC_WITH_CONVERSION)
    if "crash_date" in df.columns:
        dt = pd.to_datetime(df["crash_date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        df["crash_date"] = (dt - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

    # Ordinal encoding via replace
    for col in ORDINAL:
        actual_col = col if col in df.columns else None
        # handle alternative lightning vs lighting spelling
        if actual_col is None and col == "lighting_condition" and "lightning_condition" in df.columns:
            actual_col = "lightning_condition"
        if actual_col is None:
            continue
        mapping = ordinal_mappings.get(actual_col, {}) or ordinal_mappings.get(col, {})
        uniques = sorted([u for u in df[actual_col].dropna().unique()], key=lambda x: str(x))
        # Populate crash_type mapping dynamically if empty
        if actual_col == "crash_type" and (not mapping):
            mapping = generate_mapping(uniques)
            ordinal_mappings[actual_col] = mapping
            print(f"Generated crash_type ordinal mapping: {mapping}")
        # Validate mapping covers all unique values; if not regenerate & warn
        if set(uniques) - set(mapping.keys()):
            missing = set(uniques) - set(mapping.keys())
            auto_map = generate_mapping(uniques)
            print(f"Warning: Missing keys for {actual_col}: {missing}. Auto mapping used: {auto_map}")
            mapping = auto_map
        df[actual_col] = df[actual_col].replace(mapping)

    # Nominal one-hot encoding
    nominal_existing = [c for c in NOMINAL if c in df.columns]
    if nominal_existing:
        dummies = pd.get_dummies(df[nominal_existing], prefix=nominal_existing, dummy_na=False)
        df = pd.concat([df.drop(columns=nominal_existing), dummies], axis=1)

    # Cyclical encoding
    df = encode_cyclic_variables(df, CYCLICAL)

    # Ensure crash_type is last if present
    if "crash_type" in df.columns:
        cols = [c for c in df.columns if c != "crash_type"] + ["crash_type"]
        df = df[cols]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Encoded dataset written to: {OUTPUT_PATH}")
    if "crash_type" in ordinal_mappings and ordinal_mappings["crash_type"]:
        print(f"crash_type mapping used: {ordinal_mappings['crash_type']}")

if __name__ == "__main__":
    main()