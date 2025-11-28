# =====================================================
# FLIGHT STATUS PREDICTION — Outliers
# =====================================================

from pandas import read_csv, DataFrame, Series
from dslabs_functions import NR_STDEV,get_variable_types,determine_outlier_thresholds_for_var



# Approach 1: Dropping Outliers
# ------------------
def dropping_outliers(data: DataFrame) -> None:
    print("\nApproach 1: Dropping Outliers")
    n_std: int = NR_STDEV
    numeric_vars: list[str] = get_variable_types(data)["numeric"]
    if numeric_vars is not None:
        df: DataFrame = data.copy(deep=True)
        summary5: DataFrame = data[numeric_vars].describe()
        for var in numeric_vars:
            top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(
                summary5[var]
            )
            outliers: Series = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
            df.drop(outliers.index, axis=0, inplace=True)
        df.to_csv(f"{file_tag}_drop_outliers.csv", index=False)
        print(f"Dropped {data.shape[0] - df.shape[0]} records")
        print(f"Data after dropping outliers: {df.shape[0]} records and {df.shape[1]} variables")
    else:
        print("There are no numeric variables")


# Approach 2: Truncating outliers
# ----------------------
def truncating_outliers(data: DataFrame,file_tag: str) -> None:
    print("\nApproach 3: Truncating Outliers")
    numeric_vars: list[str] = get_variable_types(data)["numeric"]
    if numeric_vars:
        df: DataFrame = data.copy(deep=True)
        summary5: DataFrame = data[numeric_vars].describe()

        for var in numeric_vars:
            top, bottom = determine_outlier_thresholds_for_var(summary5[var])
            df[var] = df[var].apply(
                lambda x: top if x > top else bottom if x < bottom else x
            )
        df.to_csv(f"{file_tag}_truncate_outliers.csv", index=False)
        print(f"Data after dropping outliers: {df.shape}")
        print(df.describe())
    else:
        print("There are no numeric variables")

def main() -> None:
    filename: str = "/content/Combined_Flights_balanced_v2.csv"
    file_tag: str = "Flights"
    data: DataFrame = read_csv(filename)
    print(data.shape)
    #dropping_outliers(data,file_tag)
    #truncating_outliers(data,file_tag)