import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error

# Configuration
FILE_PATH = "traffic_step1_diff.csv"  # The file created in Step 1
TARGET = "Total"

def main():
    print(f"Loading {FILE_PATH}...")
    try:
        # Load data with Time as index
        data = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"ERROR: {FILE_PATH} not found. Make sure Step 1 saved it!")
        return

    # Split Train/Test (Last 30% is Test)
    split_idx = int(len(data) * 0.7)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    print(f"Train Size: {len(train)}")
    print(f"Test Size:  {len(test)}")

    # ==========================================
    # Exponential Smoothing Parameter Study
    # ==========================================
    print("\n--- Tuning Smoothing Parameter (Alpha) ---")
    
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_rmse = float('inf')
    best_alpha = None
    best_preds = None
    
    # Plotting setup
    plt.figure(figsize=(12, 6))
    plt.plot(train.index[-100:], train[TARGET].iloc[-100:], label='Train (Last 100)', color='gray', alpha=0.5)
    plt.plot(test.index, test[TARGET], label='Real Future', color='black', linewidth=1.5)

    for a in alphas:
        # 1. Fit Model
        # SimpleExpSmoothing is suitable for data without clear trend/seasonality (which differentiation removed)
        model = SimpleExpSmoothing(train[TARGET]).fit(smoothing_level=a, optimized=False)
        
        # 2. Forecast
        # We predict N steps into the future (len(test))
        preds = model.forecast(len(test))
        preds.index = test.index  # Align dates
        
        # 3. Evaluate
        rmse = np.sqrt(mean_squared_error(test[TARGET], preds))
        print(f"   Alpha={a}: RMSE={rmse:.4f}")
        
        # Plot prediction line
        plt.plot(test.index, preds, linestyle='--', linewidth=2, label=f'Alpha={a} (RMSE={rmse:.1f})')
        
        # Keep Best
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = a
            best_preds = preds

    print(f"\n🏆 WINNER: Alpha={best_alpha} with RMSE={best_rmse:.4f}")

    # Finalize Plot
    plt.title(f"Exponential Smoothing (Differentiation) - Best Alpha: {best_alpha}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time")
    plt.ylabel("Traffic Change (Diff)")
    plt.show()

    # Save Best Predictions for Final Comparison (Step 5)
    # We save as a dataframe with Time index
    best_preds.name = "Pred_Smoothing"
    best_preds.to_csv("traffic_preds_smoothing.csv")
    print("Saved best predictions to 'traffic_preds_smoothing.csv'")

if __name__ == "__main__":
    main()