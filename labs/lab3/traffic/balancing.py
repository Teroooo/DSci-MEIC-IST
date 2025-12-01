import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Configuration
TRAIN_PATH = "traffic_scaled_train.csv"
TEST_PATH = "traffic_scaled_test.csv"
TARGET = "crash_type"

def evaluate_models(X_train, y_train, X_test, y_test, label):
    """Trains NB and KNN on the balanced data and evaluates on original test data."""
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
    ax.set_title('Balancing Comparison: Undersampling vs SMOTE')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')
    
    # Zoom Y-axis
    all_scores = nb_scores + knn_scores
    ax.set_ylim(min(all_scores) * 0.95, max(all_scores) * 1.05)
    plt.tight_layout()
    plt.savefig('balancing_comparison.png')
    plt.show()

    # 2. Best Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrices for Winner: {best_res['approach']}", fontsize=16)
    
    sns.heatmap(best_res['nb_cm'], annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title("Naïve Bayes")
    
    sns.heatmap(best_res['knn_cm'], annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
    axes[1].set_title("KNN")
    plt.tight_layout()
    plt.savefig('balancing_matrices.png')
    plt.show()

def main():
    print("Loading Data (Scaled version)...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    
    # Check class distribution
    print(f"Original Class Dist: {y_train.value_counts().to_dict()}")

    # ==========================================
    # Approach 1: Undersampling
    # ==========================================
    print("\n--- Approach 1: Random Undersampling ---")
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    print(f"   -> New Size: {len(X_train_rus)}")
    
    res_1 = evaluate_models(X_train_rus, y_train_rus, X_test, y_test, "Undersampling")

    # ==========================================
    # Approach 2: SMOTE
    # ==========================================
    print("\n--- Approach 2: SMOTE ---")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"   -> New Size: {len(X_train_smote)}")
    
    res_2 = evaluate_models(X_train_smote, y_train_smote, X_test, y_test, "SMOTE")

    # ==========================================
    # Compare & Save
    # ==========================================
    score_1 = res_1['nb_f1'] + res_1['knn_f1']
    score_2 = res_2['nb_f1'] + res_2['knn_f1']
    
    if score_2 > score_1:
        print("\n🏆 WINNER: Approach 2 (SMOTE)")
        best_res = res_2
        best_X_train, best_y_train = X_train_smote, y_train_smote
    else:
        print("\n🏆 WINNER: Approach 1 (Undersampling)")
        best_res = res_1
        best_X_train, best_y_train = X_train_rus, y_train_rus
        
    # Save for next step (Feature Selection)
    best_train_df = pd.concat([best_X_train.reset_index(drop=True), best_y_train.reset_index(drop=True)], axis=1)
    best_test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    best_train_df.to_csv("traffic_balanced_train.csv", index=False)
    best_test_df.to_csv("traffic_balanced_test.csv", index=False)
    
    print("Saved best dataset to 'traffic_balanced_train.csv'")
    
    # Plot
    plot_results([res_1, res_2], best_res)

if __name__ == "__main__":
    main()