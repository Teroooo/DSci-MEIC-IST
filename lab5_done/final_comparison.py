import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
REAL_DATA_PATH = "traffic_step1_diff.csv"
PREDS_SMOOTHING = "traffic_preds_smoothing.csv"
PREDS_ARIMA = "traffic_preds_arima.csv"
PREDS_LSTM = "traffic_preds_lstm.csv"
TARGET = "Total"

def align_and_evaluate(y_true, y_pred, model_name):
    """
    Robust evaluation that forces alignment by taking the last N common samples.
    This bypasses index mismatch/duplicate errors completely.
    """
    # 1. Ensure inputs are Series
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.iloc[:, 0]
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.iloc[:, 0]

    # 2. Determine common length
    # We take the minimum length of both to handle the LSTM window offset
    min_len = min(len(y_true), len(y_pred))
    
    print(f"   Aligning {model_name}: Real={len(y_true)}, Pred={len(y_pred)} -> Using last {min_len}")

    # 3. Slice the LAST 'min_len' samples from both
    # This assumes predictions align with the end of the test set (standard)
    y_true_aligned = y_true.iloc[-min_len:]
    y_pred_aligned = y_pred.iloc[-min_len:]
    
    # 4. Overwrite Index of Preds to match True (for plotting)
    y_pred_aligned.index = y_true_aligned.index

    # 5. Evaluate
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    r2 = r2_score(y_true_aligned, y_pred_aligned)
    
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}, y_pred_aligned

def main():
    print("Loading all data...")
    try:
        # Load data
        df_real = pd.read_csv(REAL_DATA_PATH, index_col=0, parse_dates=True)
        pred_smooth = pd.read_csv(PREDS_SMOOTHING, index_col=0, parse_dates=True)
        pred_lstm = pd.read_csv(PREDS_LSTM, index_col=0, parse_dates=True)
        
        # Load ARIMA (Handle potential index parsing issues)
        try:
            pred_arima = pd.read_csv(PREDS_ARIMA, index_col=0, parse_dates=True)
        except:
            pred_arima = pd.read_csv(PREDS_ARIMA, index_col=0) # Fallback if dates fail
            
    except Exception as e:
        print(f"ERROR: Could not load files. {e}")
        return

    # Get Test Data (Last 30% of Real Data)
    split_idx = int(len(df_real) * 0.7)
    y_test = df_real[TARGET].iloc[split_idx:]

    print(f"Test Set Range: {y_test.index.min()} to {y_test.index.max()}")

    # ==========================================
    # 1. Evaluation & Alignment
    # ==========================================
    results = []
    
    # Evaluate Smoothing
    res_s, plot_s = align_and_evaluate(y_test, pred_smooth, "Exp. Smoothing")
    results.append(res_s)
    
    # Evaluate ARIMA
    res_a, plot_a = align_and_evaluate(y_test, pred_arima, "ARIMA")
    results.append(res_a)
    
    # Evaluate LSTM
    res_l, plot_l = align_and_evaluate(y_test, pred_lstm, "LSTM")
    results.append(res_l)
    
    results_df = pd.DataFrame(results).set_index('Model')
    print("\n=== FINAL LEADERBOARD ===")
    print(results_df)
    results_df.to_csv("final_results_table.csv")

    # ==========================================
    # 2. Final Comparison Plot
    # ==========================================
    plt.figure(figsize=(14, 7))
    
    # Plot Real Data
    plt.plot(y_test.index, y_test, label='Real Traffic Change', color='black', alpha=0.3, linewidth=2)
    
    # Plot Models (Using the Aligned Data)
    plt.plot(plot_s.index, plot_s, label=f'Smoothing (RMSE={res_s["RMSE"]:.1f})', linestyle='--')
    plt.plot(plot_a.index, plot_a, label=f'ARIMA (RMSE={res_a["RMSE"]:.1f})', linestyle='--')
    plt.plot(plot_l.index, plot_l, label=f'LSTM (RMSE={res_l["RMSE"]:.1f})', color='purple', linewidth=2)

    plt.title("Forecasting Showdown: Smoothing vs ARIMA vs LSTM")
    plt.xlabel("Time")
    plt.ylabel("Traffic Change")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("final_comparison_chart.png")
    plt.show()
    
    print("\nSaved chart to 'final_comparison_chart.png'. Lab 5 Complete!")

if __name__ == "__main__":
    main()