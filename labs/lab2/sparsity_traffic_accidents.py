import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import read_csv, DataFrame
from seaborn import heatmap

import sys
import os

# Add parent directory to path to import from labs/utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.dslabs_functions import HEIGHT, plot_multi_scatters_chart
from utils.dslabs_functions import get_variable_types


run_sparsity_analysis: bool = False
run_sparsity_per_class_analysis: bool = True
run_sampling: bool = True
sampling_amount: float = 0.01

filename = "../../classification/traffic_accidents.csv"
savefig_path_prefix = "images/sparsity/traffic_accidents"

data: DataFrame = read_csv(filename)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(savefig_path_prefix), exist_ok=True)


if run_sampling:
    data = data.sample(frac=sampling_amount, random_state=42)

data = data.dropna()
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
    print("Saving image for credit score sparsity study...")
    plt.savefig(f"{savefig_path_prefix}_sparsity_study.png")
    print("Image saved")
    # plt.show()
    plt.clf()
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
    target = "crash_type"

    n: int = len(traffic_accidents_vars) - 1
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(traffic_accidents_vars)):
        var1: str = traffic_accidents_vars[i]
        for j in range(i + 1, len(traffic_accidents_vars)):
            var2: str = traffic_accidents_vars[j]
            plot_multi_scatters_chart(data, var1, var2, target, ax=axs[i, j - 1])
    plt.tight_layout()
    print("Saving image for credit score sparsity per class study...")
    plt.savefig(f"{savefig_path_prefix}_sparsity_per_class_study.png")
    print("Image saved")
    # plt.show()
    plt.clf()
else:
    if not traffic_accidents_vars:
        print("Sparsity per class: there are no variables.")
    if not run_sparsity_per_class_analysis:
        print("Sparsity per class analysis: skipping.")
