import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier

# Configuration
TRAIN_PATH = "traffic_balanced_train.csv"
TEST_PATH = "traffic_balanced_test.csv"
TARGET = "crash_type"

def evaluate_models(X_train, y_train, X_test, y_test, label):
    """Trains NB and KNN on the selected features."""
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
    """Generates the performance comparison and best confusion matrices."""
    
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
    ax.set_title('Feature Selection: SelectKBest vs RFE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')
    
    # Zoom Y-axis
    all_scores = nb_scores + knn_scores
    ax.set_ylim(min(all_scores) * 0.95, max(all_scores) * 1.05)
    plt.tight_layout()
    plt.savefig('feature_selection_comparison.png')
    plt.show()

    # 2. Best Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrices for Winner: {best_res['approach']}", fontsize=16)
    
    sns.heatmap(best_res['nb_cm'], annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title("Naïve Bayes")
    
    sns.heatmap(best_res['knn_cm'], annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
    axes[1].set_title("KNN")
    plt.tight_layout()
    plt.savefig('feature_selection_matrices.png')
    plt.show()

def main():
    print("Loading Data (Balanced version)...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # Determine K (Select top 50% of features)
    n_features = X_train.shape[1]
    k_features = int(n_features * 0.5)
    print(f"Total Features: {n_features}. Selecting Top {k_features}...")

    # ==========================================
    # Approach 1: SelectKBest (Filter)
    # ==========================================
    print(f"\n--- Approach 1: SelectKBest (ANOVA) ---")
    # We use f_classif because StandardScaler produces negative values (Chi2 would fail)
    selector_kbest = SelectKBest(score_func=f_classif, k=k_features)
    
    selector_kbest.fit(X_train, y_train)
    # Get boolean mask of selected features
    mask_1 = selector_kbest.get_support()
    selected_cols_1 = X_train.columns[mask_1]
    
    print(f"   -> Selected {len(selected_cols_1)} features.")
    
    # Transform
    X_train_1 = X_train[selected_cols_1]
    X_test_1 = X_test[selected_cols_1]
    
    res_1 = evaluate_models(X_train_1, y_train, X_test_1, y_test, "SelectKBest")

    # ==========================================
    # Approach 2: RFE (Wrapper)
    # ==========================================
    print(f"\n--- Approach 2: RFE (Recursive Feature Elimination) ---")
    # Use a fast estimator (Decision Tree)
    estimator = DecisionTreeClassifier(random_state=42)
    # Step=0.1 means remove 10% of features at each pass (faster than removing 1 by 1)
    selector_rfe = RFE(estimator, n_features_to_select=k_features, step=0.1)
    
    selector_rfe.fit(X_train, y_train)
    mask_2 = selector_rfe.get_support()
    selected_cols_2 = X_train.columns[mask_2]
    
    print(f"   -> Selected {len(selected_cols_2)} features.")
    
    # Transform
    X_train_2 = X_train[selected_cols_2]
    X_test_2 = X_test[selected_cols_2]
    
    res_2 = evaluate_models(X_train_2, y_train, X_test_2, y_test, "RFE")

    # ==========================================
    # Compare & Save
    # ==========================================
    score_1 = res_1['nb_f1'] + res_1['knn_f1']
    score_2 = res_2['nb_f1'] + res_2['knn_f1']
    
    if score_2 > score_1:
        print("\n🏆 WINNER: Approach 2 (RFE)")
        best_res = res_2
        best_X_train, best_y_train = X_train_2, y_train
        best_X_test = X_test_2
    else:
        print("\n🏆 WINNER: Approach 1 (SelectKBest)")
        best_res = res_1
        best_X_train, best_y_train = X_train_1, y_train
        best_X_test = X_test_1
        
    # Save FINAL DATASET
    best_train_df = pd.concat([best_X_train.reset_index(drop=True), best_y_train.reset_index(drop=True)], axis=1)
    best_test_df = pd.concat([best_X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    best_train_df.to_csv("traffic_final_train.csv", index=False)
    best_test_df.to_csv("traffic_final_test.csv", index=False)
    
    print("Saved FINAL PREPARED DATASET to 'traffic_final_train.csv'")
    
    # Plot
    plot_results([res_1, res_2], best_res)

if __name__ == "__main__":
    main()