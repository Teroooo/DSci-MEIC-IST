import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Configuration
TRAIN_PATH = "traffic_outliers_train.csv"
TEST_PATH = "traffic_outliers_test.csv"
TARGET = "crash_type"

def evaluate_knn_only(X_train, y_train, X_test, y_test, label):
    """
    Evaluates only KNN for Scaling step, as per project guidelines.
    Returns dictionary with results.
    """
    print(f"   -> Training KNN for: {label}...")
    
    # KNN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_f1 = f1_score(y_test, knn_pred, average='weighted')
    
    print(f"      Result: KNN F1={knn_f1:.4f}")
    
    return {
        "approach": label,
        "knn_f1": knn_f1, 
        "knn_cm": confusion_matrix(y_test, knn_pred)
    }

def plot_results(results_list, best_res):
    """Generates the performance comparison and best confusion matrix."""
    
    # 1. Performance Comparison (Bar Chart)
    labels = [res['approach'] for res in results_list]
    knn_scores = [res['knn_f1'] for res in results_list]

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 6))
    rects = ax.bar(x, knn_scores, width, label='KNN', color='salmon')

    ax.set_ylabel('F1 Score (Weighted)')
    ax.set_title('Scaling Comparison: MinMax vs Standard')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects, padding=3, fmt='%.4f')
    
    # Zoom in Y-axis
    ax.set_ylim(min(knn_scores) * 0.98, max(knn_scores) * 1.02)
    plt.tight_layout()
    plt.savefig('scaling_comparison.png')
    plt.show()

    # 2. Best Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(best_res['knn_cm'], annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix (KNN) - Winner: {best_res['approach']}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('scaling_confusion_matrix.png')
    plt.show()

def main():
    print("Loading Data (Outliers version)...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Separate Features/Target
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    
    # ==========================================
    # Approach 1: MinMax Scaler
    # ==========================================
    print("\n--- Approach 1: MinMax Scaler ---")
    scaler_mm = MinMaxScaler()
    # Fit on Train, Transform Train & Test
    X_train_mm = pd.DataFrame(scaler_mm.fit_transform(X_train), columns=X_train.columns)
    X_test_mm = pd.DataFrame(scaler_mm.transform(X_test), columns=X_test.columns)
    
    res_1 = evaluate_knn_only(X_train_mm, y_train, X_test_mm, y_test, "MinMax Scaler")

    # ==========================================
    # Approach 2: Standard Scaler
    # ==========================================
    print("\n--- Approach 2: Standard Scaler ---")
    scaler_std = StandardScaler()
    # Fit on Train, Transform Train & Test
    X_train_std = pd.DataFrame(scaler_std.fit_transform(X_train), columns=X_train.columns)
    X_test_std = pd.DataFrame(scaler_std.transform(X_test), columns=X_test.columns)
    
    res_2 = evaluate_knn_only(X_train_std, y_train, X_test_std, y_test, "Standard Scaler")

    # ==========================================
    # Compare & Save
    # ==========================================
    if res_2['knn_f1'] > res_1['knn_f1']:
        print("\n🏆 WINNER: Approach 2 (Standard Scaler)")
        best_res = res_2
        best_X_train = X_train_std
        best_X_test = X_test_std
    else:
        print("\n🏆 WINNER: Approach 1 (MinMax Scaler)")
        best_res = res_1
        best_X_train = X_train_mm
        best_X_test = X_test_mm

    # Save for next step (Balancing)
    best_train_df = pd.concat([best_X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    best_test_df = pd.concat([best_X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    best_train_df.to_csv("traffic_scaled_train.csv", index=False)
    best_test_df.to_csv("traffic_scaled_test.csv", index=False)
    
    print("Saved best dataset to 'traffic_scaled_train.csv'")
    
    # Plot
    plot_results([res_1, res_2], best_res)

if __name__ == "__main__":
    main()