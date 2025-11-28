# =====================================================
# FLIGHT STATUS PREDICTION — Data Parcing
# =====================================================
from pandas import read_csv, DataFrame, concat
from sklearn.model_selection import train_test_split
from dslabs_functions import plot_multibar_chart
import matplotlib.pyplot as plt
from numpy import array, ndarray
import pandas as pd

def balance_data(file,target,outputFile) -> None:
    data = pd.read_csv(file)
    # 2. Separar classes
    cancelled_1 = data[data[target] == 1]
    cancelled_0 = data[data[target] == 0]

    print(len(cancelled_1), len(cancelled_0))

    cancelled_0_sampled: DataFrame = cancelled_0.sample(frac=0.4, random_state=21,replace=False)
    balanced: DataFrame = pd.concat([cancelled_1, cancelled_0_sampled], axis=0)
    balanced: DataFrame = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced.to_csv(outputFile, index=False)

# TRAIN AND TEST SPLIT
# ------------------
def train_test_split_function(target,file,outputTrain,outputTest) -> None:
    balanced = pd.read_csv(file)
    y: array = balanced.pop(target).to_list()
    X: ndarray = balanced.values

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    train: DataFrame = concat([DataFrame(trnX, columns=balanced.columns), DataFrame(trnY, columns=[target])], axis=1)
    train.to_csv(outputTrain, index=False)

    test: DataFrame = concat([DataFrame(tstX, columns=balanced.columns), DataFrame(tstY, columns=[target])], axis=1)
    test.to_csv(outputTest, index=False)

def main() -> None:
    target: str = "Cancelled"
    balance_data("/content/Combined_Flights_truncate.csv", target,"Combined_Flights_balanced_v2.csv")
    train_test_split_function(target,"/content/Flights_truncate_outliers.csv","Combined_Flights_train_v3_Outliers.csv", "Combined_Flights_test_v3_Outliers.csv")

if __name__ == "__main__":
    main()