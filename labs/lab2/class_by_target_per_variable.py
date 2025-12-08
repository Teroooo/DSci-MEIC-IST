import os

import matplotlib.pyplot as plt
import pandas as pd

# Adjust paths as needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "..", "classification", "traffic_accidents.csv")

# Adjust paths as needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "..", "classification", "traffic_accidents.csv")


NOMINAL = [
    #"traffic_control_device",
    #"weather_condition",
    #"first_crash_type",
    #"trafficway_type",
    #"alignment",
    #"roadway_surface_cond",
    #"road_defect",
    #"prim_contributory_cause",
    #"most_severe_injury",
    #"injuries_total",
    #"injuries_fatal",
    #"injuries_incapacitating",
    #"injuries_non_incapacitating",
    #"injuries_reported_not_evident",
    #"injuries_no_indication"
    #"weather_condition",
    #"first_crash_type",
]

TARGET = "crash_type"
TARGET_NO_INJURY = "NO INJURY / DRIVE AWAY"
TARGET_INJURY = "INJURY AND / OR TOW DUE TO CRASH"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def compute_percentages(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by feature class with columns:
    - pct_no_injury
    - pct_injury
    and (optionally) 'count' with total rows per class.
    Percentages are conditional on the feature value and sum to 1 (100%) per class.
    """
    # Filter only the two target classes of interest
    df_filtered = df[df[TARGET].isin([TARGET_NO_INJURY, TARGET_INJURY])].copy()

    # Drop rows where the feature is missing
    df_filtered = df_filtered.dropna(subset=[feature])

    # Count occurrences per (feature, target)
    counts = (
        df_filtered.groupby([feature, TARGET])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure both target columns exist
    for col in [TARGET_NO_INJURY, TARGET_INJURY]:
        if col not in counts.columns:
            counts[col] = 0

    # Total per feature class
    total = counts.sum(axis=1)

    # Avoid division by zero
    pct_no_injury = counts[TARGET_NO_INJURY] / total
    pct_injury = counts[TARGET_INJURY] / total

    result = pd.DataFrame(
        {
            "pct_no_injury": pct_no_injury,
            "pct_injury": pct_injury,
            "count": total,
        }
    )

    # Order categories from lowest to highest probability of NO INJURY / DRIVE AWAY
    result = result.sort_values("pct_no_injury", ascending=True)

    return result


def plot_stacked_bar(percentages: pd.DataFrame, feature: str, max_categories: int = 15):
    """
    Makes a stacked bar chart for the given feature.

    Each bar is a feature class; bars show percentages of:
      - NO INJURY / DRIVE AWAY
      - INJURY AND / OR TOW DUE TO CRASH
    Percentages sum to 100% per bar.
    """
    # Optionally limit number of categories for readability
    if len(percentages) > max_categories:
        percentages = percentages.iloc[:max_categories]

    categories = percentages.index.astype(str)
    no_injury = percentages["pct_no_injury"].values * 100
    injury = percentages["pct_injury"].values * 100

    x = range(len(categories))

    plt.figure(figsize=(max(8, len(categories) * 0.6), 6))

    # Put INJURY at the bottom, NO INJURY stacked on top
    plt.bar(x, injury, label=TARGET_INJURY)
    plt.bar(x, no_injury, bottom=injury, label=TARGET_NO_INJURY)

    plt.xticks(x, categories, rotation=45, ha="right")
    # Hide class names on the x-axis
    #plt.xticks(x, [""] * len(categories))
    plt.ylabel("Percentage of records (%)")
    plt.title(f"{feature} vs {TARGET}")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()


def print_injury_percentages(percentages: pd.DataFrame, feature: str):
    """Print, for a given feature, an ordinal-style dict for INJURY share."""
    print(f'"{feature}": {{')

    # percentages index holds the classes; pct_injury is between 0 and 1
    for cls, row in percentages.iterrows():
        pct = row["pct_injury"] * 100
        print(f'    "{cls}": {pct:.1f},')

    print("},")


def main():
    df = load_data(DATA_PATH)

    # Ensure required columns exist
    missing_cols = [c for c in NOMINAL + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    for feature in NOMINAL:
        percentages = compute_percentages(df, feature)
        print_injury_percentages(percentages, feature)
        plot_stacked_bar(percentages, feature)
        plt.show()


if __name__ == "__main__":
    main()