import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
import sys
import os

# Add parent directory to path to import from labs/utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.dslabs_functions import get_variable_types, plot_bar_chart


credit_score_filename: str = "../../classification/traffic_accidents.csv"
credit_score_savefig_path_prefix: str = "images/dimensionality/traffic_accidents/traffic_accidents"

# credit_score_filename: str = "../../classification/Combined_Flights_2022.csv"
# credit_score_savefig_path_prefix: str = "images/dimensionality/Combined_Flights_2022/Combined_Flights_2022"

credit_score_data: DataFrame = read_csv(credit_score_filename)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(credit_score_savefig_path_prefix), exist_ok=True)

run_credit_score_records_variables_analysis: bool = True
run_credit_score_variable_types_analysis: bool = True
run_credit_score_missing_values_analysis: bool = True

# ------------------
# Ration between Nr. of records and Nr. of variables
# ------------------

if run_credit_score_records_variables_analysis:
    print(f"Credit Score Data: {credit_score_data.shape[0]} records, {credit_score_data.shape[1]} variables")

    plt.figure()
    credit_score_values: dict[str, int] = {"nr records": credit_score_data.shape[0],
                                           "nr variables": credit_score_data.shape[1]}
    plot_bar_chart(
        list(credit_score_values.keys()), list(credit_score_values.values()), title="Nr of records vs nr variables"
    )
    plt.tight_layout()
    plt.savefig(f"{credit_score_savefig_path_prefix}_records_variables.png")
    plt.show()
    plt.clf()
else:
    print("Ration between Nr. of records and Nr. of variables: skipping analysis.")

# ------------------
# Nr. of variables per type
# ------------------

if run_credit_score_variable_types_analysis:
    credit_score_variable_types: dict[str, list] = get_variable_types(credit_score_data)
    print(f"Credit Score Data: {len(credit_score_variable_types)} variable types")

    counts: dict[str, int] = {}
    for tp in credit_score_variable_types.keys():
        counts[tp] = len(credit_score_variable_types[tp])

    plt.figure()
    plot_bar_chart(
        list(counts.keys()), list(counts.values()), title="Nr of variables per type"
    )
    plt.tight_layout()
    plt.savefig(f"{credit_score_savefig_path_prefix}_variable_types.png")
    plt.show()
    plt.clf()
else:
    print("Nr. of variables per type: skipping analysis.")

# ------------------
# Nr. missing values per variable
# ------------------

if run_credit_score_missing_values_analysis:
    credit_score_mv: dict[str, int] = {}
    for var in credit_score_data.columns:
        nr: int = credit_score_data[var].isnull().sum()
        if nr > 0:
            credit_score_mv[var] = nr

    plt.figure()
    plot_bar_chart(
        list(credit_score_mv.keys()),
        list(credit_score_mv.values()),
        title="Nr of missing values per variable",
        xlabel="variables",
        ylabel="nr missing values",
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{credit_score_savefig_path_prefix}_mv.png")
    plt.show()
    plt.clf()
else:
    print("Nr. missing values per variable: skipping analysis.")