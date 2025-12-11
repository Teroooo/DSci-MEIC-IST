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

# Use sequential index
data = data.reset_index(drop=True)
data.index.name = "Record"

print("Nr. Records = ", data.shape)
print("First record", data.index[0])
print("Last record", data.index[-1])
print("Time range:", data['Time'].iloc[0], "to", data['Time'].iloc[-1])
print("Date range:", data['Date'].iloc[0], "to", data['Date'].iloc[-1])

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

# Encode 'Traffic Situation'
traffic_mapping = {'low': 0, 'normal': 1, 'high': 2, 'heavy': 3}
data_encoded['Traffic Situation'] = data_encoded['Traffic Situation'].map(traffic_mapping)

# Drop original string columns
data_encoded = data_encoded.drop(columns=['Time', 'Day of the week', 'day_of_week'])

# Create images directory
os.makedirs("images/dimensionality", exist_ok=True)

# 1. RECORD LEVEL
print("\n=== Record-level data ===")
print("Nr. Records = ", data_encoded.shape)
plot_ts_multivariate_chart(data_encoded, title=f"{file_tag} {target} - Record Level")
savefig(f"images/dimensionality/{file_tag}_record.png", dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to images/dimensionality/{file_tag}_record.png")

# 2. HOURLY
print("\n=== Hourly aggregation ===")
records_per_hour = 4
data_hourly = data_encoded.copy()
data_hourly['Hour'] = data_hourly.index // records_per_hour

hourly_data = data_hourly.groupby('Hour').agg({
    'Date': 'first',
    'CarCount': 'sum',
    'BikeCount': 'sum',
    'BusCount': 'sum',
    'TruckCount': 'sum',
    'Total': 'sum',
    'Traffic Situation': 'mean',
    'day_of_week_sin': 'mean',
    'day_of_week_cos': 'mean'
}).reset_index()

hourly_data.index = hourly_data['Hour']
hourly_data.index.name = 'Hour'
hourly_data = hourly_data.drop(columns=['Hour', 'Date'])

print("Nr. Hours = ", hourly_data.shape)
plot_ts_multivariate_chart(hourly_data, title=f"{file_tag} {target} - Hourly")
savefig(f"images/dimensionality/{file_tag}_hourly.png", dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to images/dimensionality/{file_tag}_hourly.png")

# 3. DAILY
print("\n=== Daily aggregation ===")
records_per_day = 96
data_daily = data_encoded.copy()
data_daily['Day'] = data_daily.index // records_per_day

daily_data = data_daily.groupby('Day').agg({
    'Date': 'first',
    'CarCount': 'sum',
    'BikeCount': 'sum',
    'BusCount': 'sum',
    'TruckCount': 'sum',
    'Total': 'sum',
    'Traffic Situation': 'mean',
    'day_of_week_sin': 'mean',
    'day_of_week_cos': 'mean'
}).reset_index()

daily_data.index = daily_data['Day']
daily_data.index.name = 'Day'
daily_data = daily_data.drop(columns=['Day', 'Date'])

print("Nr. Days = ", daily_data.shape)
plot_ts_multivariate_chart(daily_data, title=f"{file_tag} {target} - Daily")
savefig(f"images/dimensionality/{file_tag}_daily.png", dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to images/dimensionality/{file_tag}_daily.png")

# 4. WEEKLY
print("\n=== Weekly aggregation ===")
records_per_week = records_per_day * 7
data_weekly = data_encoded.copy()
data_weekly['Week'] = data_weekly.index // records_per_week

weekly_data = data_weekly.groupby('Week').agg({
    'Date': 'first',
    'CarCount': 'sum',
    'BikeCount': 'sum',
    'BusCount': 'sum',
    'TruckCount': 'sum',
    'Total': 'sum',
    'Traffic Situation': 'mean',
    'day_of_week_sin': 'mean',
    'day_of_week_cos': 'mean'
}).reset_index()

weekly_data.index = weekly_data['Week']
weekly_data.index.name = 'Week'
weekly_data = weekly_data.drop(columns=['Week', 'Date'])

print("Nr. Weeks = ", weekly_data.shape)
plot_ts_multivariate_chart(weekly_data, title=f"{file_tag} {target} - Weekly")
savefig(f"images/dimensionality/{file_tag}_weekly.png", dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to images/dimensionality/{file_tag}_weekly.png")

print("\n=== All plots generated successfully! ===")