# =====================================================
# FLIGHT STATUS PREDICTION — Missing Values (MV)
# =====================================================

from dslabs_functions import mvi_by_dropping, mvi_by_filling, get_variable_types
from pandas import read_csv, DataFrame




# APPROACH 1 — DROPPING (threshold-based removal)
def dropping_high_mv_columns(data_frame: DataFrame, threshold: float) -> DataFrame:
    print("\nApproach 1: Dropping Missing Values")
    # 1 - zero
    colunas_remover = [
        'ArrTime', 'ArrDelayMinutes', 'ArrDelay', 'ArrDel15',
        'ArrivalDelayGroups', 'AirTime', 'ActualElapsedTime',
        'WheelsOn', 'TaxiIn'
    ]

    data_frame.drop(columns=colunas_remover, inplace=True)
    colunas_fill = ['DepTime', 'DepDelayMinutes','DepDelay', 'Tail_Number',
                    'DepDel15' ,'DepartureDelayGroups','TaxiOut','WheelsOff']
    for coluna in colunas_fill:
        data_frame[coluna].fillna(0, inplace=True)
    df_drop_threshold: DataFrame = mvi_by_dropping(data_frame,
        min_pct_per_variable=0.7,
        min_pct_per_record=0.9)


    print(f"After dropping: {df_drop_threshold.shape}")

    df_drop_threshold.to_csv("Combined_Flights_imputed_mv_approach1.csv", index=False)
    print("File saved: Combined_Flights_imputed_mv_approach1.csv")


# APPROACH 2 — IMPUTATION (filling missing values)
def impute_missing_values(data_frame: DataFrame) -> None:
    print("\nApproach 2: Imputing Missing Values")
    # Strategies:
    #   "frequent"
    #   "constant"
    #   "knn"
    strategy_used = "constant"

    data_frame: DataFrame = mvi_by_filling(data_frame, strategy=strategy_used)

    print(f"After imputation ({strategy_used}): {data_frame.shape}")

    data_frame.to_csv("Combined_Flights_imputed_mv_approach2.csv", index=False)
    print("File saved: Combined_Flights_imputed_mv_approach2.csv")

def main():
    filename: str = "Combined_Flights_2022.csv"
    Combined_Flights: DataFrame = read_csv(filename)
    print(f"Dataset loaded: {Combined_Flights.shape}")
    impute_missing_values(Combined_Flights)
