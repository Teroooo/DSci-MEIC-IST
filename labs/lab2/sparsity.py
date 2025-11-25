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

run_credit_score_sparsity_analysis: bool = False
run_credit_score_sparsity_per_class_analysis: bool = False
run_credit_score_correlation_analysis: bool = True
run_sampling: bool = False
sampling_amount: float = 0.001

credit_score_filename: str = "../../classification/traffic_accidents_encoded.csv"
credit_score_savefig_path_prefix: str = "images/dimensionality/traffic_accidents/traffic_accidents"

# credit_score_filename: str = "../../classification/Combined_Flights_2022.csv"
# credit_score_savefig_path_prefix: str = "images/dimensionality/Combined_Flights_2022/Combined_Flights_2022"

credit_score_data: DataFrame = read_csv(credit_score_filename)

if run_sampling:
    credit_score_data = credit_score_data.sample(frac=sampling_amount, random_state=42)

credit_score_data = credit_score_data.dropna()
credit_score_vars: list = credit_score_data.columns.to_list()

# ------------------
# Correlation analysis
# ------------------

if credit_score_vars and run_credit_score_correlation_analysis:
    print("Printing correlation analysis for credit score...")

    encoded_credit_score_filename = "../../classification/traffic_accidents_encoded.csv"
    encoded_credit_score_data: DataFrame = read_csv(encoded_credit_score_filename, na_values="")

    credit_score_corr_mtx: DataFrame = encoded_credit_score_data.corr().abs()

    plt.figure(figsize=(10, 9))
    heatmap(
        abs(credit_score_corr_mtx),
        xticklabels=credit_score_corr_mtx.columns,
        yticklabels=credit_score_corr_mtx.columns,
        annot=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    plt.tight_layout()
    print("Saving image for credit score correlation analysis...")
    plt.savefig(f"{credit_score_savefig_path_prefix}_correlation_analysis.png")
    plt.show()
    print("Image saved")
    plt.clf()
else:
    if not credit_score_vars:
        print("Correlation analysis: there are no variables.")
    if not run_credit_score_correlation_analysis:
        print("Correlation analysis: skipping.")


# ------------------
# Sparsity analysis
# ------------------


if credit_score_vars and run_credit_score_sparsity_analysis:
    print("Printing sparsity analysis for credit score...")
    n: int = len(credit_score_vars) - 1
    fig: Figure
    axs: ndarray
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(credit_score_vars)):
        print(f"i: {i}")
        var1: str = credit_score_vars[i]
        for j in range(i + 1, len(credit_score_vars)):
            var2: str = credit_score_vars[j]
            plot_multi_scatters_chart(credit_score_data, var1, var2, ax=axs[i, j - 1])
    plt.tight_layout()
    print("Saving image for credit score sparsity study...")
    plt.savefig(f"{credit_score_savefig_path_prefix}_sparsity_study.png")
    print("Image saved")
    # plt.show()
    plt.clf()
else:
    if not credit_score_vars:
        print("Sparsity class: there are no variables.")
    if not run_credit_score_sparsity_analysis:
        print("Sparsity analysis: skipping.")

# ------------------
# Sparsity per class analysis
# ------------------

if credit_score_vars and run_credit_score_sparsity_per_class_analysis:
    print("Printing sparsity per class analysis for credit score...")
    target = "Credit_Score"

    n: int = len(credit_score_vars) - 1
    fig, axs = plt.subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
    for i in range(len(credit_score_vars)):
        var1: str = credit_score_vars[i]
        for j in range(i + 1, len(credit_score_vars)):
            var2: str = credit_score_vars[j]
            plot_multi_scatters_chart(credit_score_data, var1, var2, target, ax=axs[i, j - 1])
    plt.tight_layout()
    print("Saving image for credit score sparsity per class study...")
    plt.savefig(f"{credit_score_savefig_path_prefix}_sparsity_per_class_study.png")
    print("Image saved")
    # plt.show()
    plt.clf()
else:
    if not credit_score_vars:
        print("Sparsity per class: there are no variables.")
    if not run_credit_score_sparsity_per_class_analysis:
        print("Sparsity per class analysis: skipping.")
