import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import read_csv, DataFrame
from seaborn import heatmap
from sklearn.model_selection import train_test_split


import sys
import os

# Add parent directory to path to import from labs/utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.dslabs_functions import HEIGHT, plot_multi_scatters_chart
from utils.dslabs_functions import get_variable_types


run_sparsity_analysis: bool = False
run_sparsity_per_class_analysis: bool = True
run_sampling: bool = True
sampling_amount: float = 0.001

filename = "../../classification/Combined_Flights_2022.csv"
savefig_path_prefix = "images/sparsity/Combined_Flights_2022/Combined_Flights_2022"

data: DataFrame = read_csv(filename)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(savefig_path_prefix), exist_ok=True)


if run_sampling:
    #data = data.sample(frac=sampling_amount, random_state=1)
    data, _ = train_test_split(
    data,
    train_size=sampling_amount,
    stratify=data['Cancelled'],
    random_state=42
)
print(data['Cancelled'].value_counts())
# data = data.dropna() !PLS DONT DO IT HERE!
traffic_accidents_vars: list = data.columns.to_list()

# ------------------
# Sparsity analysis
# ------------------


if traffic_accidents_vars and run_sparsity_analysis:
    print("Printing sparsity analysis for credit score...")
    n: int = len(traffic_accidents_vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(traffic_accidents_vars)):
        print(f"i: {i}")
        var1: str = traffic_accidents_vars[i]
        for j in range(i + 1, len(traffic_accidents_vars)):
            var2: str = traffic_accidents_vars[j]
            plot_multi_scatters_chart(data, var1, var2, ax=axs[i, j - 1])
    plt.tight_layout()
    print("Saving image for sparsity study...")
    plt.savefig(f"{savefig_path_prefix}_sparsity_study.png")
    print("Image saved")
    plt.close()
else:
    if not traffic_accidents_vars:
        print("Sparsity class: there are no variables.")
    if not run_sparsity_analysis:
        print("Sparsity analysis: skipping.")

# ------------------
# Sparsity per class analysis
# ------------------

if traffic_accidents_vars and run_sparsity_per_class_analysis:
    print("Printing sparsity per class analysis for credit score...")
    target = "Cancelled"

    n: int = len(traffic_accidents_vars) - 1
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(traffic_accidents_vars)):
        var1: str = traffic_accidents_vars[i]
        print(f"i: {i}/{n}")
        if i+1 < len(traffic_accidents_vars):
            for j in range(i + 1, len(traffic_accidents_vars)):
                print(f"    j: {j}/{n}")
                var2: str = traffic_accidents_vars[j]
                if var1 == target or var2 == target:
                    var3 = ""
                    subset = data[[var1, var2]].dropna()
                else:
                    var3 = target
                    subset = data[[var1, var2, var3]].dropna()
                plot_multi_scatters_chart(subset, var1, var2, var3, ax=axs[i, j - 1])

    print("charts plotted, starting single plot")
    plt.tight_layout()
    print("Saving image for sparsity per class study...")
    plt.savefig(f"{savefig_path_prefix}_sparsity_per_class_study.png")
    print("Image saved")
    plt.close()
else:
    if not traffic_accidents_vars:
        print("Sparsity per class: there are no variables.")
    if not run_sparsity_per_class_analysis:
        print("Sparsity per class analysis: skipping.")

plt.close("all")
print("Done!")
