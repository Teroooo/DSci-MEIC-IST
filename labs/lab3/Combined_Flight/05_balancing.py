# =====================================================
# FLIGHT STATUS PREDICTION — Data Balancing
# =====================================================
from dslabs_functions import plot_bar_chart
from pandas import read_csv, DataFrame, Series, concat
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from numpy import ndarray


# Approach 1: Oversampling
# ------------------
def oversampling_balancing(data: DataFrame, target: str, file_tag: str,positive_class:int,negative_class:int) -> None:
    df_positives: Series = data[data[target] == positive_class]
    df_negatives: Series = data[data[target] == negative_class]
    df_pos_sample: DataFrame = DataFrame(df_positives.sample(len(df_negatives), replace=True))
    df_over: DataFrame = concat([df_pos_sample, df_negatives], axis=0)
    df_over.to_csv(f"{file_tag}_over.csv", index=False)

    print("Minority class=", positive_class, ":", len(df_pos_sample))
    print("Majority class=", negative_class, ":", len(df_negatives))
    print("Proportion:", round(len(df_pos_sample) / len(df_negatives), 2), ": 1")



# Approach 2: SMOTE
# ------------------

def smote_balancing(data: DataFrame, target: str, file_tag: str,positive_class:int,negative_class:int) -> None:

    smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=42)
    y = data.pop(target).values
    X: ndarray = data.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(data.columns) + [target]
    df_smote.to_csv(f"{file_tag}_smote.csv", index=False)

    smote_target_count: Series = Series(smote_y).value_counts()
    print("Minority class=", positive_class, ":", smote_target_count[positive_class])
    print("Majority class=", negative_class, ":", smote_target_count[negative_class])
    print(
        "Proportion:",
        round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),
        ": 1",
    )
    print(df_smote.shape)

    def balanceClass() -> None:
        file = "/content/Combined_Flights_2022.csv"
        target = "Cancelled"
        original: DataFrame = read_csv(f"{file}", sep=",", decimal=".")

        target_count: Series = original[target].value_counts()
        positive_class = target_count.idxmin()
        negative_class = target_count.idxmax()

        print("Minority class=", positive_class, ":", target_count[positive_class])
        print("Majority class=", negative_class, ":", target_count[negative_class])
        print("Proportion:",round(target_count[positive_class] / target_count[negative_class], 2),": 1",)
        values: dict[str, list] = {"Original": [target_count[positive_class], target_count[negative_class]]}
        plt.figure()
        plt.plot_bar_chart(target_count.index.to_list(), target_count.to_list(), title="Class balance")
        plt.show()

def main():
    file_tag: str = "Combined_flight"
    data_1: DataFrame = read_csv("/content/Combined_Flights_test_v3_Outliers.csv")
    data: DataFrame = read_csv("/content/Combined_Flights_train_v3_Outliers.csv")
    data = concat([data_1, data], axis=0)
    target: str = "Canselled"
    positive_class: int = 1
    negative_class: int = 0