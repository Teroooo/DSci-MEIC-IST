import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer

# Configuration
TRAIN_PATH = "traffic_accidents_train.csv"
TEST_PATH = "traffic_accidents_test.csv"
TARGET = "crash_type"

def evaluate_models(X_train, y_train, X_test, y_test, label):
    """Trains models and returns a dictionary of results."""
    
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
    
    return {
        "approach": label,
        "nb_f1": nb_f1,
        "nb_cm": confusion_matrix(y_test, nb_pred),
        "knn_f1": knn_f1,
        "knn_cm": confusion_matrix(y_test, knn_pred)
    }

def plot_performance(results_list):
    """
    Plots a grouped bar chart comparing F1 scores for all approaches.
    """
    # Prepare data for plotting
    labels = [res['approach'] for res in results_list]
    nb_scores = [res['nb_f1'] for res in results_list]
    knn_scores = [res['knn_f1'] for res in results_list]

    x = np.arange(len(labels))  # label locations
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, nb_scores, width, label='Naïve Bayes', color='skyblue')
    rects2 = ax.bar(x + width/2, knn_scores, width, label='KNN', color='salmon')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1 Score (Weighted)')
    ax.set_title('Performance Comparison: Drop Rows vs. Imputation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value labels on top of bars
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')
    
    # Adjust Y-axis to make small differences visible
    # We find the min and max score to center the view
    all_scores = nb_scores + knn_scores
    min_score = min(all_scores)
    max_score = max(all_scores)
    ax.set_ylim(min_score * 0.95, max_score * 1.05)

    plt.tight_layout()
    plt.savefig('comparison_chart.png') # Save comparison chart
    plt.show()

def plot_confusion_matrices(result_dict):
    """
    Plots the confusion matrices for the winning approach.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrices for Best Approach: {result_dict['approach']}", fontsize=16)

    # Helper to plot one matrix
    def plot_cm(cm, ax, title):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    plot_cm(result_dict['nb_cm'], axes[0], "Naïve Bayes")
    plot_cm(result_dict['knn_cm'], axes[1], "k-Nearest Neighbors")

    plt.tight_layout()
    plt.savefig('confusion_matrices.png') # Save matrices
    plt.show()

def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Setup Features/Target
    X_train_raw = train_df.drop(columns=[TARGET])
    y_train_raw = train_df[TARGET]
    X_test_raw = test_df.drop(columns=[TARGET])
    y_test_raw = test_df[TARGET]

    # --- Approach 1: Drop Rows ---
    print("Evaluating Approach 1 (Drop Rows)...")
    train_dropped = train_df.dropna()
    test_dropped = test_df.dropna()
    
    res_1 = evaluate_models(
        train_dropped.drop(columns=[TARGET]), train_dropped[TARGET],
        test_dropped.drop(columns=[TARGET]), test_dropped[TARGET],
        "Drop Rows"
    )

    # --- Approach 2: Impute Mode ---
    print("Evaluating Approach 2 (Impute Mode)...")
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(X_train_raw)
    
    X_train_imp = pd.DataFrame(imputer.transform(X_train_raw), columns=X_train_raw.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)
    
    res_2 = evaluate_models(
        X_train_imp, y_train_raw,
        X_test_imp, y_test_raw,
        "Impute Mode"
    )

    # --- PLOTTING ---
    # 1. Compare Performances
    plot_performance([res_1, res_2])

    # 2. Compare Total Score to find Winner
    score_1 = res_1['nb_f1'] + res_1['knn_f1']
    score_2 = res_2['nb_f1'] + res_2['knn_f1']
    
    if score_2 > score_1:
        print(f"\nWINNER: {res_2['approach']}")
        best_res = res_2
    else:
        print(f"\nWINNER: {res_1['approach']}")
        best_res = res_1
        
    # 3. Plot Matrices for Winner
    plot_confusion_matrices(best_res)

if __name__ == "__main__":
    main()