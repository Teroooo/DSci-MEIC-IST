import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
FILE_PATH = "traffic_step1_diff.csv"
TARGET = "Total"

def create_sequences(data, seq_length):
    """
    Converts a list of numbers into 3D sequences (Samples, Time Steps, Features)
    Input: [10, 20, 30, 40, 50], seq_length=3
    Output X: [[10, 20, 30], [20, 30, 40]]
    Output y: [40, 50]
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    print(f"Loading {FILE_PATH}...")
    try:
        df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("ERROR: File not found.")
        return

    # 1. Scaling (Crucial for Neural Networks)
    # LSTMs work best when data is between 0 and 1 (or -1 and 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(df[[TARGET]])

    # 2. Split Data (Last 30% is Test)
    # Note: We split BEFORE creating sequences to respect time order
    split_idx = int(len(data_scaled) * 0.7)
    train_data = data_scaled[:split_idx]
    test_data = data_scaled[split_idx:]
    
    # ==========================================
    # Parameter Study: Sequence Length
    # ==========================================
    print("\n--- Tuning LSTM Sequence Length ---")
    
    seq_lengths = [5, 10, 15]
    best_rmse = float('inf')
    best_seq = None
    best_preds = None
    
    # Store history for loss curves
    best_history = None

    for seq_len in seq_lengths:
        print(f"\nTraining with Sequence Length: {seq_len}...")
        
        # A. Create Sequences
        X_train, y_train = create_sequences(train_data, seq_len)
        X_test, y_test = create_sequences(test_data, seq_len)
        
        # Reshape for LSTM: [samples, time steps, features]
        # Features = 1 (just the 'Total' value)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # B. Build Model
        model = Sequential()
        model.add(LSTM(50, activation='tanh', input_shape=(seq_len, 1)))
        model.add(Dense(1)) # Output layer
        model.compile(optimizer='adam', loss='mse')
        
        # C. Train
        # epochs=20 is usually enough for a quick test. 
        # validation_split=0.1 lets us see the validation loss curve
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, 
                            validation_split=0.1, verbose=0)
        
        # D. Predict
        preds_scaled = model.predict(X_test, verbose=0)
        
        # E. Inverse Scale (Get back to real units)
        preds_real = scaler.inverse_transform(preds_scaled)
        y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # F. Evaluate
        rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
        print(f"   Sequence={seq_len}: RMSE={rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_seq = seq_len
            best_preds = preds_real
            best_history = history

    print(f"\n🏆 WINNER: Sequence Length={best_seq} with RMSE={best_rmse:.4f}")

    # ==========================================
    # Plotting
    # ==========================================
    # 1. Loss Curve (Check for Overfitting)
    plt.figure(figsize=(10, 4))
    plt.plot(best_history.history['loss'], label='Training Loss')
    plt.plot(best_history.history['val_loss'], label='Validation Loss')
    plt.title(f"LSTM Loss Curves (Seq Length {best_seq})")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2. Prediction Plot
    # We need to align the predictions with the correct dates
    # The first 'seq_len' points of test data are consumed to make the first prediction
    test_dates = df.index[split_idx + best_seq:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[split_idx:], df[TARGET].iloc[split_idx:], label='Real Future', color='black', linewidth=1.5)
    plt.plot(test_dates, best_preds, label=f'LSTM (Seq {best_seq})', color='purple', linestyle='--', linewidth=2)
    
    plt.title(f"LSTM Model (Differentiation) - Best Seq: {best_seq}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time")
    plt.ylabel("Traffic Change")
    plt.show()

    # Save Best Predictions for Final Comparison
    # Create DataFrame with correct index
    final_preds_df = pd.DataFrame(best_preds, index=test_dates, columns=['Pred_LSTM'])
    final_preds_df.to_csv("traffic_preds_lstm.csv")
    print("Saved best predictions to 'traffic_preds_lstm.csv'")

if __name__ == "__main__":
    main()