import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix

# Configuration
TRAIN_PATH = "traffic_accidents_mvi_train.csv"
TEST_PATH = "traffic_accidents_mvi_test.csv"
TARGET = "crash_type"
Z_THRESHOLD = 3  # The standard cutoff for outliers

def evaluate_models(X_train, y_train, X_test, y_test, label):
    """Trains models and returns a dictionary of results."""
    print(f"   -> Training models for: {label}...")
    
    # 1. Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_f1 = f1_score(y_test, nb_pred, average='weighted')
    
    # 2. KNN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_f1 = f1_score(y_test, knn_pred, average='weighted')
    
    print(f"      Results: NB F1={nb_f1:.4f}, KNN F1={knn_f1:.4f}")
    
    return {
        "approach": label,
        "nb_f1": nb_f1, "nb_cm": confusion_matrix(y_test, nb_pred),
        "knn_f1": knn_f1, "knn_cm": confusion_matrix(y_test, knn_pred)
    }

def plot_results(results_list, best_res):
    """Generates the required charts."""
    
    # 1. Performance Comparison
    labels = [res['approach'] for res in results_list]
    nb_scores = [res['nb_f1'] for res in results_list]
    knn_scores = [res['knn_f1'] for res in results_list]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, nb_scores, width, label='Naïve Bayes', color='skyblue')
    rects2 = ax.bar(x + width/2, knn_scores, width, label='KNN', color='salmon')

    ax.set_ylabel('F1 Score (Weighted)')
    ax.set_title(f'Impact of Outlier Removal (Z-Score > {Z_THRESHOLD})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')
    
    # Zoom in Y-axis
    all_scores = nb_scores + knn_scores
    ax.set_ylim(min(all_scores) * 0.98, max(all_scores) * 1.02)
    plt.tight_layout()
    plt.show()

    # 2. Best Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrices for Winner: {best_res['approach']}", fontsize=16)
    
    sns.heatmap(best_res['nb_cm'], annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title("Naïve Bayes")
    
    sns.heatmap(best_res['knn_cm'], annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
    axes[1].set_title("KNN")
    plt.tight_layout()
    plt.show()

def main():
    print("Loading Data (MVI version)...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Separate Features/Target
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    
    # ==========================================
    # Approach 1: Keep Outliers (Baseline)
    # ==========================================
    print("\n--- Approach 1: Keep Outliers ---")
    res_1 = evaluate_models(X_train, y_train, X_test, y_test, "Keep Outliers")

    # ==========================================
    # Approach 2: Remove Outliers (Z-Score)
    # ==========================================
    print(f"\n--- Approach 2: Remove Outliers (Z-Score > {Z_THRESHOLD}) ---")
    
    # Smart Selection: Apply Z-score only to columns with >2 unique values
    # (Avoids breaking binary/one-hot encoded columns)
    cols_to_check = [col for col in X_train.columns if X_train[col].nunique() > 2]
    print(f"   -> Applying Z-score to {len(cols_to_check)} numeric columns (skipping binary)...")
    
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(X_train[cols_to_check]))
    
    # Find rows where ALL columns are within threshold ( < 3 )
    # (i.e., keep row if it has NO outliers in the checked columns)
    mask = (z_scores < Z_THRESHOLD).all(axis=1)
    
    X_train_clean = X_train[mask]
    y_train_clean = y_train[mask]
    
    n_removed = len(X_train) - len(X_train_clean)
    print(f"   -> Removed {n_removed} outliers ({n_removed/len(X_train):.1%}) from training set.")
    
    res_2 = evaluate_models(X_train_clean, y_train_clean, X_test, y_test, "Remove Outliers")

    # ==========================================
    # Compare & Save
    # ==========================================
    score_1 = res_1['nb_f1'] + res_1['knn_f1']
    score_2 = res_2['nb_f1'] + res_2['knn_f1']
    
    if score_2 > score_1:
        print("\n🏆 WINNER: Approach 2 (Remove Outliers)")
        best_res = res_2
        best_X_train, best_y_train = X_train_clean, y_train_clean
    else:
        print("\n🏆 WINNER: Approach 1 (Keep Outliers)")
        best_res = res_1
        best_X_train, best_y_train = X_train, y_train
        
    # Save for next step (Scaling)
    best_train_df = pd.concat([best_X_train.reset_index(drop=True), best_y_train.reset_index(drop=True)], axis=1)
    best_train_df.to_csv("traffic_outliers_train.csv", index=False)
    test_df.to_csv("traffic_outliers_test.csv", index=False) # Test set never changes here
    
    print("Saved best dataset to 'traffic_outliers_train.csv'")
    
    # Plot
    plot_results([res_1, res_2], best_res)

if __name__ == "__main__":
    main()