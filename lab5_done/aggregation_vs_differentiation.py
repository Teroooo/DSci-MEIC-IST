import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

FILE_PATH = "TrafficTwoMonth.csv"
TARGET = "Total"

def load_data():
    df = pd.read_csv(FILE_PATH)
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.set_index('Time').sort_index()
        
    # Feature Engineering (Cyclical Time)
    if "Day of the week" in df.columns:
        day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
        df["day"] = df["Day of the week"].map(day_map)
        df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)
        
    # DROP THE CHEATING COLUMNS
    # We only keep Target and Time features
    cols_to_keep = [TARGET, 'day_sin', 'day_cos']
    df = df[cols_to_keep].fillna(method='ffill')
    return df

def evaluate(train, test, label):
    # 1. Persistence (Baseline)
    # Predict t using t-1
    last_val = train[TARGET].iloc[-1]
    pred_persist = test[TARGET].shift(1).fillna(last_val)
    rmse_persist = np.sqrt(mean_squared_error(test[TARGET], pred_persist))
    
    # 2. Linear Regression (Trend Only)
    # Create TimeIndex feature (0, 1, 2, 3...)
    X_train = train.drop(columns=[TARGET])
    X_train['TimeIndex'] = np.arange(len(X_train))
    
    X_test = test.drop(columns=[TARGET])
    X_test['TimeIndex'] = np.arange(len(X_train), len(X_train) + len(X_test))
    
    lr = LinearRegression()
    lr.fit(X_train, train[TARGET])
    pred_lr = lr.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(test[TARGET], pred_lr))
    
    print(f"--- {label} ---")
    print(f"   Persistence RMSE: {rmse_persist:.2f}")
    print(f"   Linear Reg RMSE:  {rmse_lr:.2f}")
    return rmse_persist, rmse_lr

def main():
    df = load_data()
    
    # Study 1: Aggregation (Hourly)
    df_agg = df.resample('h').sum() # Sum traffic per hour
    split = int(len(df_agg) * 0.7)
    rmse_agg_p, rmse_agg_lr = evaluate(df_agg.iloc[:split], df_agg.iloc[split:], "Aggregation")
    
    # Study 2: Differentiation (Changes)
    df_diff = df.diff().dropna()
    split = int(len(df_diff) * 0.7)
    rmse_diff_p, rmse_diff_lr = evaluate(df_diff.iloc[:split], df_diff.iloc[split:], "Differentiation")
    
    # Plot
    labels = ['Aggregation', 'Differentiation']
    persist = [rmse_agg_p, rmse_diff_p]
    lr = [rmse_agg_lr, rmse_diff_lr]
    
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(x - width/2, persist, width, label='Persistence', color='gray')
    ax.bar(x + width/2, lr, width, label='Linear Reg', color='blue')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('RMSE'); ax.set_title('Transformation Study (Fixed Leakage)')
    ax.legend(); ax.bar_label(ax.containers[0]); ax.bar_label(ax.containers[1])
    plt.show()
    
    # Winner Decision (Average of both models)
    avg_agg = (rmse_agg_p + rmse_agg_lr)/2
    avg_diff = (rmse_diff_p + rmse_diff_lr)/2
    
    print("\n=== CONCLUSION ===")
    if avg_agg < avg_diff:
        print(f"WINNER: Aggregation (Avg RMSE: {avg_agg:.2f})")
        df_agg.to_csv("traffic_step1_agg.csv")
    else:
        print(f"WINNER: Differentiation (Avg RMSE: {avg_diff:.2f})")
        df_diff.to_csv("traffic_step1_diff.csv")

if __name__ == "__main__":
    main()