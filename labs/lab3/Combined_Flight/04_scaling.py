# =====================================================
# FLIGHT STATUS PREDICTION — Scaling
# =====================================================

from pandas import read_csv, DataFrame, Series,concat
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np



# Approach 1: ABS Scaler
# ----------------
def mxcore_scaling(data: DataFrame, target: str, file_tag: str) -> None:
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove('Cancelled')
    # 1) SHIFT PARA DEIXAR TODOS OS VALORES >= 0
    min_val = data[numeric_cols].min().min()
    if min_val < 0:
        shift = abs(min_val)
        data[numeric_cols] = data[numeric_cols] + shift

    vars: list[str] = data.columns.to_list()
    target_data: Series = data.pop(target)

    scaler: MaxAbsScaler = MaxAbsScaler()
    df_mxcore = DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    df_mxcore[target] = target_data

    df_mxcore.to_csv(f"{file_tag}_scaled_mxcore.csv", index=False)



# Approach 2: MinMax Scaler
# ------------------
def minmax_scaling(data: DataFrame, target: str, file_tag: str) -> None:
    target_data: Series = data.pop(target)
    scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    df_minmax = DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    df_minmax[target] = target_data
    df_minmax.to_csv(f"{file_tag}_scaled_minmax.csv", index=False)

# Results:
def plot_scaling_results(data: DataFrame, df_mxcore: DataFrame, df_minmax: DataFrame, file_tag: str) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(50, 10), squeeze=False)

    axs[0, 0].set_title("Original data")
    data.boxplot(ax=axs[0, 0])
    axs[0, 0].tick_params(labelrotation=90)

    axs[0, 1].set_title("ABS normalization")
    df_mxcore.boxplot(ax=axs[0, 1])
    axs[0, 1].tick_params(labelrotation=90)

    axs[0, 2].set_title("MinMax normalization")
    df_minmax.boxplot(ax=axs[0, 2])
    axs[0, 2].tick_params(labelrotation=90)

    plt.tight_layout()
    plt.savefig(f"{file_tag}_scaling.png")
    plt.show()
    plt.clf()


def main():
    file_tag: str = "Combined_flight"
    data: DataFrame = read_csv("/content/Combined_Flights_train_v3_Outliers.csv")
    target: str = "Cancelled"
    print(f"Dataset nr records={data.shape[0]}", f"nr variables={data.shape[1]}")
    mxcore_scaling(data, target, file_tag)
    minmax_scaling(data, target, file_tag)

if __name__ == "__main__":
    main()