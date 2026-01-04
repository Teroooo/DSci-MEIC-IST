import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from dslabs_functions import plot_multibar_chart, run_NB, run_KNN, plot_confusion_matrix

# _________________________________ CONFIGURATION _________________________________
# Input files (Output from Balancing Step - Undersampling won)
TRAIN_PATH = "traffic_accidents_balanced_train.csv"
TEST_PATH = "traffic_accidents_balanced_test.csv"

# Output files
OUTPUT_TRAIN = "traffic_accidents_selected_train.csv"
OUTPUT_TEST = "traffic_accidents_selected_test.csv"

TARGET = "crash_type"
CLASS_EVAL_METRICS = ["accuracy", "recall", "precision", "auc", "f1"]

# Thresholds
VARIANCE_THRESHOLD = 0.8  # Remove features with variance < 0.1
CORRELATION_THRESHOLD = 0.75 # Remove features with correlation > 0.9

def evaluate_approach(train: pd.DataFrame, test: pd.DataFrame, target: str = "class") -> dict[str, list]:
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    
    eval_dict: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric="accuracy")
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric="accuracy")

    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            nb_score = eval_NB.get(met, 0.0)
            knn_score = eval_KNN.get(met, 0.0)
            eval_dict[met] = [nb_score, knn_score]
            
    return eval_dict

def get_confusion_matrix(train, test, target, model_type="KNN"):
    trn_y = train.pop(target).values
    trn_x = train.values
    tst_y = test.pop(target).values
    tst_x = test.values
    
    if model_type == "NB":
        clf = GaussianNB()
    else:
        clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean') 
        
    clf.fit(trn_x, trn_y)
    prd_y = clf.predict(tst_x)
    labels = list(pd.unique(trn_y))
    labels.sort()
    
    return confusion_matrix(tst_y, prd_y, labels=labels), labels

def plot_variance_analysis(train, target):
    """
    Plots the variance of each feature to help decide the threshold.
    """
    # Calculate variance (exclude target)
    data = train.drop(columns=[target])
    variances = data.var().sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(variances))
    plt.barh(y_pos, variances, align='center', alpha=0.7)
    plt.yticks(y_pos, variances.index, fontsize=8)
    plt.xlabel('Variance')
    plt.title('Feature Variance Analysis')
    
    # Draw a vertical line for the threshold we are about to use
    plt.axvline(x=VARIANCE_THRESHOLD, color='r', linestyle='--', label=f'Threshold={VARIANCE_THRESHOLD}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("images/traffic_selection_variance_plot.png")
    print("   Variance plot saved to images/traffic_selection_variance_plot.png")
    # plt.show() # Uncomment if you want to see it live

def drop_redundant_features(train, test, threshold=0.9):
    """
    Drops features that are highly correlated (redundant).
    """
    # Calculate correlation matrix
    corr_matrix = train.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Make sure we don't drop the target!
    if TARGET in to_drop:
        to_drop.remove(TARGET)
        
    print(f"   Dropping {len(to_drop)} redundant features: {to_drop}")
    
    train_reduced = train.drop(columns=to_drop)
    test_reduced = test.drop(columns=to_drop)
    
    return train_reduced, test_reduced

def run_feature_selection():
    print("Loading Data (from Balancing Step)...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    
    # 0. Variance Analysis (Plot Only)
    print("\n--- ANALYZING VARIANCE ---")
    plot_variance_analysis(train_raw, TARGET)

    # _________________________________ APPROACH 1: Low Variance Filter _________________________
    print(f"\n--- APPROACH 1: Drop Low Variance (Threshold < {VARIANCE_THRESHOLD}) ---")
    
    # We use VarianceThreshold from sklearn
    # Note: We fit on X, but we need to keep DataFrame structure
    X_train = train_raw.drop(columns=[TARGET])
    y_train = train_raw[TARGET]
    X_test = test_raw.drop(columns=[TARGET])
    
    sel = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    sel.fit(X_train)
    
    # Get mask of kept features
    kept_features = X_train.columns[sel.get_support()]
    dropped_features = [col for col in X_train.columns if col not in kept_features]
    
    print(f"   Dropping {len(dropped_features)} features: {dropped_features}")
    
    # Reconstruct DataFrames
    train_var = pd.concat([X_train[kept_features].reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_var = pd.concat([X_test[kept_features].reset_index(drop=True), test_raw[TARGET].reset_index(drop=True)], axis=1)
    
    results_var = evaluate_approach(train_var.copy(), test_var.copy(), target=TARGET)
    
    if results_var:
        plt.figure()
        plot_multibar_chart(
            ["NB", "KNN"], results_var, title="Low Variance Filter Evaluation", percentage=True
        )
        plt.savefig("images/traffic_selection_variance_eval.png")
        print("   Evaluation saved: images/traffic_selection_variance_eval.png")

    # _________________________________ APPROACH 2: Redundant Features (Correlation) ____________
    print(f"\n--- APPROACH 2: Drop Redundant Features (Corr > {CORRELATION_THRESHOLD}) ---")
    
    train_corr, test_corr = drop_redundant_features(train_raw.copy(), test_raw.copy(), threshold=CORRELATION_THRESHOLD)
    
    results_corr = evaluate_approach(train_corr.copy(), test_corr.copy(), target=TARGET)
    
    if results_corr:
        plt.figure()
        plot_multibar_chart(
            ["NB", "KNN"], results_corr, title="Redundancy Removal Evaluation", percentage=True
        )
        plt.savefig("images/traffic_selection_correlation_eval.png")
        print("   Evaluation saved: images/traffic_selection_correlation_eval.png")

    # _________________________________ COMPARISON & SELECTION _________________________________
    # Compare KNN Accuracy
    acc_var_knn = results_var['accuracy'][1]
    acc_corr_knn = results_corr['accuracy'][1]
    
    print(f"\nRESULTS (KNN Accuracy): Variance Filter={acc_var_knn:.4f} vs Redundancy Filter={acc_corr_knn:.4f}")
    
    best_df_train = None
    best_df_test = None
    best_approach_name = ""
    best_model_type = "" 
    
    if acc_corr_knn > acc_var_knn:
        print("🏆 WINNER: Redundancy (Correlation) Filter")
        best_df_train = train_corr
        best_df_test = test_corr
        best_approach_name = "Redundancy Filter"
        best_model_type = "NB" if results_corr['accuracy'][0] > results_corr['accuracy'][1] else "KNN"
    else:
        print("🏆 WINNER: Low Variance Filter")
        best_df_train = train_var
        best_df_test = test_var
        best_approach_name = "Variance Filter"
        best_model_type = "NB" if results_var['accuracy'][0] > results_var['accuracy'][1] else "KNN"

    # Save Best Dataset
    best_df_train.to_csv(OUTPUT_TRAIN, index=False)
    best_df_test.to_csv(OUTPUT_TEST, index=False)
    print(f"   Best dataset saved to {OUTPUT_TRAIN}")

    # _________________________________ CONFUSION MATRIX (Best Model) _________________________________
    print(f"\nGenerating Confusion Matrix for the Winner: {best_model_type} ({best_approach_name})...")
    
    cnf_mtx, labels = get_confusion_matrix(best_df_train.copy(), best_df_test.copy(), TARGET, best_model_type)
    
    plt.figure()
    plot_confusion_matrix(cnf_mtx, labels)
    plt.title(f"Confusion Matrix - {best_model_type} ({best_approach_name})")
    
    plt.tight_layout()
    plt.savefig("images/traffic_selection_best_cm.png")
    print("   CM saved to images/traffic_selection_best_cm.png")
    
    plt.show()

if __name__ == "__main__":
    run_feature_selection()