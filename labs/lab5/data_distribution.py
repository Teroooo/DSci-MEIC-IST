from numpy import sum, array
from pandas import DataFrame, Series, read_csv
from matplotlib.pyplot import figure, show, subplots, gcf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.dslabs_functions import (
    HEIGHT, 
    plot_line_chart, 
    ts_aggregation_by,
    set_chart_labels,
    autocorrelation_study,
    get_lagged_series,
    plot_multiline_chart
)

file_tag = "traffic"
target = "Total"
data: DataFrame = read_csv(
    "../../forecasting/TrafficTwoMonth.csv",
    sep=",",
    decimal=".",
)

# Prepare data with sequential index
data = data.reset_index(drop=True)
data.index.name = "Record"

# Create a Series for the target variable
series: Series = data[target]

# Create images directory
os.makedirs("images/data_distribution", exist_ok=True)

print("=== Time Series Data Distribution Analysis ===\n")

# ========================================
# 1. AGGREGATIONS
# ========================================
print("Creating aggregations...")

# We need to create hourly, daily, weekly series
# Since we have 4 records per hour (every 15 min), let's aggregate
records_per_hour = 4
records_per_day = 96
records_per_week = 672

# Create time-indexed version for easier aggregation
# We'll use record number as a pseudo-time index
series_hourly = series.groupby(series.index // records_per_hour).sum()
series_hourly.index.name = "Hour"

series_daily = series.groupby(series.index // records_per_day).sum()
series_daily.index.name = "Day"

series_weekly = series.groupby(series.index // records_per_week).sum()
series_weekly.index.name = "Week"

print(f"Original (15-min): {len(series)} records")
print(f"Hourly: {len(series_hourly)} records")
print(f"Daily: {len(series_daily)} records")
print(f"Weekly: {len(series_weekly)} records\n")

# ========================================
# 2. PLOT WEEKLY AGGREGATION
# ========================================
print("Plotting weekly aggregation...")
figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series_weekly.index.to_list(),
    series_weekly.to_list(),
    xlabel="weeks",
    ylabel=target,
    title=f"{file_tag} weekly {target}",
)
figure(1).savefig("images/data_distribution/weekly_line.png", dpi=300, bbox_inches='tight')
print("✓ Saved: images/data_distribution/weekly_line.png")

# ========================================
# 3. 5-NUMBER SUMMARY WITH BOXPLOTS
# ========================================
print("\nCreating 5-number summary...")
fig: Figure
axs: array
fig, axs = subplots(2, 2, figsize=(2 * HEIGHT, HEIGHT))
fig.suptitle(f"{file_tag} {target} - 5-Number Summary")

set_chart_labels(axs[0, 0], title="15-MIN (Original)")
axs[0, 0].boxplot(series)

set_chart_labels(axs[0, 1], title="WEEKLY")
axs[0, 1].boxplot(series_weekly)

axs[1, 0].grid(False)
axs[1, 0].set_axis_off()
axs[1, 0].text(0.05, 0.5, str(series.describe()), fontsize="small", verticalalignment='center')

axs[1, 1].grid(False)
axs[1, 1].set_axis_off()
axs[1, 1].text(0.05, 0.5, str(series_weekly.describe()), fontsize="small", verticalalignment='center')

fig.savefig("images/data_distribution/5number_summary.png", dpi=300, bbox_inches='tight')
print("✓ Saved: images/data_distribution/5number_summary.png")

# ========================================
# 4. VARIABLES DISTRIBUTION (HISTOGRAMS)
# ========================================
print("\nCreating distribution histograms...")
grans: list[Series] = [series, series_hourly, series_daily, series_weekly]
gran_names: list[str] = ["15-min", "Hourly", "Daily", "Weekly"]

fig: Figure
axs: array
fig, axs = subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
fig.suptitle(f"{file_tag} {target} - Distribution by Granularity")

for i in range(len(grans)):
    set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=target, ylabel="Nr records")
    axs[i].hist(grans[i].values, bins=20, edgecolor='black')

fig.savefig("images/data_distribution/distributions.png", dpi=300, bbox_inches='tight')
print("✓ Saved: images/data_distribution/distributions.png")
# ========================================
# 5. AUTOCORRELATION - LAGGED SERIES
# ========================================
print("\nAnalyzing autocorrelation with lagged series...")
# Use hourly data for cleaner visualization
fig5 = figure(figsize=(3 * HEIGHT, HEIGHT))
lags = get_lagged_series(series_hourly, 50, 25)  # max_lag=50 hours, delta=25
plot_multiline_chart(
    series_hourly.index.to_list(), 
    lags, 
    xlabel="Hour", 
    ylabel=target,
    title=f"{file_tag} {target} - Lagged Series (Hourly)"
)
fig5.savefig("images/data_distribution/lagged_series.png", dpi=300, bbox_inches='tight')
plt.close(fig5)
print("✓ Saved: images/data_distribution/lagged_series.png")

# ========================================
# 6. AUTOCORRELATION STUDY
# ========================================
print("\nPerforming autocorrelation study...")
# Use hourly data: study up to 24 hours (1 day), every 6 hours
autocorrelation_study(series_hourly, max_lag=24, delta=6)
gcf().savefig("images/data_distribution/autocorrelation_study.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/data_distribution/autocorrelation_study.png")

print("\n=== All distribution analysis plots generated successfully! ===")
show()
print("\n=== All distribution analysis plots generated successfully! ===")
show()