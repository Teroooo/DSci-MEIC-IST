from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
import numpy as np

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.dslabs_functions import plot_line_chart, HEIGHT, plot_ts_multivariate_chart

file_tag = "traffic"
target = "Total"
data: DataFrame = read_csv(
    "../../forecasting/TrafficTwoMonth.csv",
    sep=",",
    decimal=".",
)
# Use sequential index instead of parsing incomplete datetime
data = data.reset_index(drop=True)
data.index.name = "Record"

print("Nr. Records = ", data.shape)
print("First record", data.index[0])
print("Last record", data.index[-1])
print("Time range:", data['Time'].iloc[0], "to", data['Time'].iloc[-1])

# Encode categorical columns
data_encoded = data.copy()

day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

data_encoded["day_of_week"] = data_encoded["Day of the week"].map(day_map)

data_encoded["day_of_week_sin"] = np.sin(2 * np.pi * data_encoded["day_of_week"] / 7)
data_encoded["day_of_week_cos"] = np.cos(2 * np.pi * data_encoded["day_of_week"] / 7)

# Encode 'Traffic Situation' (low=0, normal=1, high=2, heavy=3)
traffic_mapping = {'low': 0, 'normal': 1, 'high': 2, 'heavy': 3}
data_encoded['Traffic Situation'] = data_encoded['Traffic Situation'].map(traffic_mapping)

# Drop original string columns and intermediate numeric day column
data_encoded = data_encoded.drop(columns=['Time', 'Day of the week', 'day_of_week'])

print("Columns:", data_encoded.columns.tolist())
print("Index name:", data_encoded.index.name)
print("Index type:", type(data_encoded.index))

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Plot and save
plot_ts_multivariate_chart(data_encoded, title=f"{file_tag} {target}")
savefig(f"images/{file_tag}_dimesnionality.png", dpi=300, bbox_inches='tight')
print(f"Plot saved to images/{file_tag}_dimesnionality.png")
show()