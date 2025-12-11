from pandas import DataFrame, Series, read_csv
from matplotlib.pyplot import figure, show, plot, legend, savefig
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.dslabs_functions import (
    HEIGHT,
    plot_line_chart,
    plot_components,
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

# Create aggregations
records_per_hour = 4
records_per_day = 96

series_hourly = series.groupby(series.index // records_per_hour).sum()
series_hourly.index.name = "Hour"

series_daily = series.groupby(series.index // records_per_day).sum()
series_daily.index.name = "Day"

# Create images directory
os.makedirs("images/data_stationarity", exist_ok=True)

print("=== Time Series Stationarity Analysis ===\n")

# ========================================
# 1. SEASONAL DECOMPOSITION - HOURLY
# ========================================
print("Performing seasonal decomposition for hourly data...")
from statsmodels.tsa.seasonal import seasonal_decompose

# For hourly data, expect daily seasonality (24 hours per day)
decomposition_hourly = seasonal_decompose(series_hourly, model="add", period=24)
fig, axs = plt.subplots(4, 1, figsize=(3 * HEIGHT, 4 * HEIGHT))
fig.suptitle(f"{file_tag} hourly {target}")

axs[0].plot(series_hourly)
axs[0].set_title('Observed')
axs[0].set_ylabel(target)

axs[1].plot(decomposition_hourly.trend)
axs[1].set_title('Trend')
axs[1].set_ylabel(target)

axs[2].plot(decomposition_hourly.seasonal)
axs[2].set_title('Seasonal')
axs[2].set_ylabel(target)

axs[3].plot(decomposition_hourly.resid)
axs[3].set_title('Residual')
axs[3].set_ylabel(target)
axs[3].set_xlabel(series_hourly.index.name)

plt.tight_layout()
plt.savefig("images/data_stationarity/components_hourly.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/data_stationarity/components_hourly.png")

# ========================================
# 2. SEASONAL DECOMPOSITION - DAILY
# ========================================
print("\nPerforming seasonal decomposition for daily data...")
# For daily data, expect weekly seasonality (7 days per week)
decomposition_daily = seasonal_decompose(series_daily, model="add", period=7)
fig, axs = plt.subplots(4, 1, figsize=(3 * HEIGHT, 4 * HEIGHT))
fig.suptitle(f"{file_tag} daily {target}")

axs[0].plot(series_daily)
axs[0].set_title('Observed')
axs[0].set_ylabel(target)

axs[1].plot(decomposition_daily.trend)
axs[1].set_title('Trend')
axs[1].set_ylabel(target)

axs[2].plot(decomposition_daily.seasonal)
axs[2].set_title('Seasonal')
axs[2].set_ylabel(target)

axs[3].plot(decomposition_daily.resid)
axs[3].set_title('Residual')
axs[3].set_ylabel(target)
axs[3].set_xlabel(series_daily.index.name)

plt.tight_layout()
plt.savefig("images/data_stationarity/components_daily.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/data_stationarity/components_daily.png")

# ========================================
# 3. STATIONARITY STUDY - OVERALL MEAN
# ========================================
print("\nPlotting stationarity study with overall mean...")
figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series_hourly.index.to_list(),
    series_hourly.to_list(),
    xlabel=series_hourly.index.name,
    ylabel=target,
    title=f"{file_tag} stationary study - hourly",
    name="original",
)
n: int = len(series_hourly)
plot(series_hourly.index, [series_hourly.mean()] * n, "r-", label="mean")
legend()
plt.savefig("images/data_stationarity/stationarity_overall_mean.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/data_stationarity/stationarity_overall_mean.png")

# ========================================
# 4. STATIONARITY STUDY - BINNED MEAN
# ========================================
print("\nPlotting stationarity study with binned mean...")
BINS = 10
mean_line: list[float] = []

for i in range(BINS):
    segment: Series = series_hourly[i * n // BINS : (i + 1) * n // BINS]
    mean_value: list[float] = [segment.mean()] * (n // BINS)
    mean_line += mean_value
mean_line += [mean_line[-1]] * (n - len(mean_line))

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series_hourly.index.to_list(),
    series_hourly.to_list(),
    xlabel=series_hourly.index.name,
    ylabel=target,
    title=f"{file_tag} stationary study - binned mean",
    name="original",
    show_stdev=True,
)
plot(series_hourly.index, mean_line, "r-", label="binned mean")
legend()
plt.savefig("images/data_stationarity/stationarity_binned_mean.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/data_stationarity/stationarity_binned_mean.png")

# ========================================
# 5. AUGMENTED DICKEY-FULLER TEST
# ========================================
print("\n=== Augmented Dickey-Fuller Test ===")

def eval_stationarity(series: Series, name: str = "") -> bool:
    result = adfuller(series)
    print(f"\n{name}:")
    print(f"  ADF Statistic: {result[0]:.3f}")
    print(f"  p-value: {result[1]:.3f}")
    print("  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.3f}")
    is_stationary = result[1] <= 0.05
    print(f"  → The series {'IS' if is_stationary else 'IS NOT'} stationary")
    return is_stationary

# Test original (15-min) series
eval_stationarity(series, "Original (15-min)")

# Test hourly series
eval_stationarity(series_hourly, "Hourly")

# Test daily series
eval_stationarity(series_daily, "Daily")

print("\n=== All stationarity analysis completed! ===")
show()