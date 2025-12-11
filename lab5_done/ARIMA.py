import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

# Configuration
FILE_PATH = "traffic_step1_diff.csv"
TARGET = "Total"

def main():
    print(f"Loading {FILE_PATH}...")
    try:
        # Load data (No parse_dates needed if index is just integers, but good practice)
        data = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
        # Ensure index is treated effectively (sometimes ARIMA complains about frequency)
        if not data.index.freq:
            data.index = pd.RangeIndex(start=0, stop=len(data), step=1)
    except FileNotFoundError:
        print("ERROR: File not found.")
        return

    # Split Train/Test (Last 30% is Test)
    split_idx = int(len(data) * 0.7)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    print(f"Train Size: {len(train)}")
    print(f"Test Size:  {len(test)}")

    # ==========================================
    # ARIMA Grid Search
    # ==========================================
    print("\n--- Tuning ARIMA Parameters (p, d, q) ---")
    
    # We fix d=0 because the data is ALREADY differentiated (stationary)
    d = 0 
    
    # We test combinations of p (AR) and q (MA)
    p_values = [1, 2, 4, 6]
    q_values = [1, 2, 4, 6]
    
    best_rmse = float('inf')
    best_cfg = None
    best_preds = None

    # Suppress warnings from models that fail to converge
    warnings.filterwarnings("ignore")

    for p in p_values:
        for q in q_values:
            try:
                # 1. Train ARIMA
                # order=(p, d, q)
                model = ARIMA(train[TARGET], order=(p, d, q))
                model_fit = model.fit()
                
                # 2. Forecast
                # Predict steps covering the test set
                start_idx = len(train)
                end_idx = len(train) + len(test) - 1
                preds = model_fit.predict(start=start_idx, end=end_idx)
                
                # 3. Evaluate
                rmse = np.sqrt(mean_squared_error(test[TARGET], preds))
                print(f"   ARIMA({p},{d},{q}): RMSE={rmse:.4f}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_cfg = (p, d, q)
                    best_preds = preds
                    
            except Exception as e:
                print(f"   ARIMA({p},{d},{q}): Failed ({str(e)})")
                continue

    print(f"\n🏆 WINNER: ARIMA{best_cfg} with RMSE={best_rmse:.4f}")

    # ==========================================
    # Plotting
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    # Plot Last 100 points of Train for context
    plt.plot(train.index[-100:], train[TARGET].iloc[-100:], label='Train (Last 100)', color='gray', alpha=0.5)
    
    # Plot Real Future
    plt.plot(test.index, test[TARGET], label='Real Future', color='black', linewidth=1.5)
    
    # Plot Best Prediction
    # Align index for plotting
    best_preds = pd.Series(best_preds.values, index=test.index)
    plt.plot(test.index, best_preds, label=f'ARIMA{best_cfg}', color='red', linestyle='--', linewidth=2)

    plt.title(f"ARIMA Model (Differentiation) - Best Order: {best_cfg}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time Step")
    plt.ylabel("Traffic Change")
    plt.show()

    # Save Best Predictions for Final Comparison
    best_preds.name = "Pred_ARIMA"
    best_preds.to_csv("traffic_preds_arima.csv")
    print("Saved best predictions to 'traffic_preds_arima.csv'")

if __name__ == "__main__":
    main()