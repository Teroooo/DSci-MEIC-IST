from dslabs_functions import evaluate_approach, plot_multibar_chart, mvi_by_filling,select_low_variance_variables,HEIGHT, apply_feature_selection
from pandas import read_csv, DataFrame, concat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from imblearn.over_sampling import SMOTE


file_tag = "Version_2_Balancing"

# Load Data
train = read_csv("/content/Combined_Flights_train_v3_Outliers.csv")
test = read_csv("/content/Combined_Flights_test_v3_Outliers.csv")
print(train.columns)
print(test.columns)

# ------------------------------
# SAMPLE TRAIN
# ------------------------------
cancelled_1 = train[train["Cancelled"] == 1]
cancelled_0 = train[train["Cancelled"] == 0]

print('Train original:', len(cancelled_1), len(cancelled_0))

cancelled_1_sampled = cancelled_1.sample(n=80000, random_state=18, replace=False)
cancelled_0_sampled = cancelled_0.sample(n=160000, random_state=19, replace=False)

train = concat([cancelled_1_sampled, cancelled_0_sampled], axis=0)

# ------------------------------
# SAMPLE TEST
# ------------------------------
cancelled_1 = test[test["Cancelled"] == 1]
cancelled_0 = test[test["Cancelled"] == 0]

print('Test original:', len(cancelled_1), len(cancelled_0))

cancelled_1_sampled = cancelled_1.sample(n=30000, random_state=13, replace=False)
cancelled_0_sampled = cancelled_0.sample(n=80000, random_state=12, replace=False)

test = concat([cancelled_1_sampled, cancelled_0_sampled], axis=0)

# ------------------------------
# HANDLE MISSING VALUES
# ------------------------------
train = mvi_by_filling(train, strategy="constant")
test = mvi_by_filling(test, strategy="constant")


# ------------------------------
# SMOTE
# ------------------------------
numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove('Cancelled')

y = train["Cancelled"]
X = train[numeric_cols]

smote = SMOTE(sampling_strategy="minority", random_state=42)
X_res, y_res = smote.fit_resample(X, y)

train = DataFrame(X_res, columns=numeric_cols)
train["Cancelled"] = y_res.values

print("AFTER SMOTE:")
print(train["Cancelled"].value_counts())
print("Shape:", train.shape)

# ------------------------------
# SCALE FEATURES
# ------------------------------


scaler = MinMaxScaler()
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])

print("Train target distribution:\n", train['Cancelled'].value_counts())
print("Test target distribution:\n", test['Cancelled'].value_counts())

# ------------------
# FEATURE SELECTION: Dropping Low Variance Variables
# ------------------

vars2drop: list[str] = select_low_variance_variables(
    train, max_threshold=0.05, target="Cancelled")
train_cp, test_cp = apply_feature_selection(
    train, test, vars2drop, filename="Flights_FS",
    tag="lowvar")
plt.figure(figsize=(2 * HEIGHT, HEIGHT))


# ------------------------------
# EVALUATE
# ------------------------------
plt.figure()
results = evaluate_approach(train_cp, test_cp, target="Cancelled", metric="recall")

if results:
    plot_multibar_chart(["NB", "KNN"], results,
                        title=f"{file_tag} evaluation", percentage=True)
    plt.savefig(f"{file_tag}_eval.png")
    plt.show()
else:
    print("Evaluation results are empty.")
