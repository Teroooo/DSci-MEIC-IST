from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show
from seaborn import heatmap
import os

# Define get_variable_types function locally to avoid import issues
def get_variable_types(data: DataFrame) -> dict[str, list]:
    """Get variable types from DataFrame"""
    variable_types = {"numeric": [], "binary": [], "categorical": []}
    
    for col in data.columns:
        unique_values = data[col].nunique()
        
        if data[col].dtype in ['int64', 'float64']:
            if unique_values == 2:
                variable_types["binary"].append(col)
            else:
                variable_types["numeric"].append(col)
        else:
            variable_types["categorical"].append(col)
    
    return variable_types

script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"[INFO] Script directory: {script_dir}")

file_tag = "Combined_Flights_2022"
filename = os.path.join(script_dir, '..', '..', '..', 'classification', 'Combined_flight_v1.csv')

# Read the data
print(f"[INFO] Reading data from: {filename}")
data: DataFrame = read_csv(filename, na_values="")
print(f"[INFO] Data shape before dropna: {data.shape}")
data = data.dropna()
print(f"[INFO] Data shape after dropna: {data.shape}")

# Calculate correlation matrix for all numeric columns
print("[INFO] Calculating correlation matrix...")
corr_mtx: DataFrame = data.corr().abs()
print(f"[INFO] Correlation matrix shape: {corr_mtx.shape}")

if corr_mtx.shape[0] > 0:
    print("[INFO] Creating correlation heatmap...")
    figure(figsize=(14, 12))
    heatmap(
        abs(corr_mtx),
        xticklabels=corr_mtx.columns,
        yticklabels=corr_mtx.columns,
        annot=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    from matplotlib import pyplot as plt
    plt.tight_layout()
    
    output_path = os.path.join(script_dir, 'images', file_tag, 'sparsity', f'{file_tag}_correlation_analysis.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[INFO] Saving correlation analysis to: {output_path}")
    savefig(output_path)
    print(f"[SUCCESS] Image saved!")
    show()
    plt.clf()
else:
    print("[ERROR] No numeric variables found for correlation analysis.")
