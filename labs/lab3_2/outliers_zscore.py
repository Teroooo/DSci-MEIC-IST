import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from numpy import ndarray
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from dslabs_functions import plot_multibar_chart, run_NB, run_KNN, plot_confusion_matrix

# _________________________________ CONFIGURATION _________________________________
TRAIN_PATH = "traffic_accidents_mvi_train.csv"
TEST_PATH = "traffic_accidents_mvi_test.csv"

# Output files
OUTPUT_TRAIN = "traffic_accidents_outliers_zscore_train.csv"
OUTPUT_TEST = "traffic_accidents_outliers_zscore_test.csv"

TARGET = "crash_type"
CLASS_EVAL_METRICS = ["accuracy", "recall", "precision", "auc", "f1"]
Z_THRESHOLD = 3  # Standard cutoff (3 standard deviations)

def evaluate_approach(train: pd.DataFrame, test: pd.DataFrame, target: str = "class") -> dict[str, list]:
    """
    Runs NB and KNN and returns a dict with all 5 metrics for the plot.
    Format: {'accuracy': [nb_score, knn_score], ...}
    """
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    
    eval_dict: dict[str, list] = {}

    # Run Models
    # We use accuracy as the driver but run_* functions return all metrics if configured correctly
    # or we extract them manually if run_* only returns one. 
    # Assuming dslabs run_NB/KNN returns a dict of metrics.
    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric="accuracy")
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric="accuracy")

    # Structure data for dslabs plot_multibar_chart
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            nb_score = eval_NB.get(met, 0.0)
            knn_score = eval_KNN.get(met, 0.0)
            eval_dict[met] = [nb_score, knn_score]
            
    return eval_dict

def get_confusion_matrix(train, test, target, model_type="NB"):
    """
    Helper to generate the confusion matrix for the final best model
    """
    trn_y = train.pop(target).values
    trn_x = train.values
    tst_y = test.pop(target).values
    tst_x = test.values
    
    if model_type == "NB":
        clf = GaussianNB()
    else:
        # Default dslabs params
        clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean') 
        
    clf.fit(trn_x, trn_y)
    prd_y = clf.predict(tst_x)
    labels = list(pd.unique(trn_y))
    labels.sort()
    
    return confusion_matrix(tst_y, prd_y, labels=labels), labels

def run_zscore_analysis():
    print("Loading MVI data...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    # _________________________________ APPROACH 1: KEEP OUTLIERS (Baseline) _________________________________
    print("\n--- APPROACH 1: KEEP OUTLIERS ---")
    results_keep = evaluate_approach(train_raw.copy(), test_raw.copy(), target=TARGET)
    
    if results_keep:
        plt.figure()
        plot_multibar_chart(
            ["NB", "KNN"], results_keep, title="Keep Outliers Evaluation", percentage=True
        )
        plt.savefig("images/traffic_outliers_zscore_keep_eval.png")
        print("   Chart saved: images/traffic_outliers_zscore_keep_eval.png")

    # _________________________________ APPROACH 2: REMOVE OUTLIERS (Z-Score) ________________________________
    print(f"\n--- APPROACH 2: REMOVE OUTLIERS (Z-Score > {Z_THRESHOLD}) ---")
    
    # Copy data to avoid touching original
    train_z = train_raw.copy()
    
    # We must calculate Z-scores only on numeric columns. 
    # Since previous steps encoded everything, we assume all are numeric.
    # Note: Z-score on categorical/binary data (like 0/1) is mathematically possible but sometimes aggressive.
    # Ideally, we exclude the target class from Z-score calculation.
    
    cols_to_check = train_z.columns.drop(TARGET)
    
    # Calculate Z-scores
    # nan_policy='omit' helps if there are NaNs (though we should have imputed them)
    z_scores = train_z[cols_to_check].apply(zscore, nan_policy='omit')
    
    # Create mask: Keep rows where ALL columns are within threshold
    # abs(z) < 3
    mask = (np.abs(z_scores) < Z_THRESHOLD).all(axis=1)
    
    train_clean = train_z[mask].copy()
    
    print(f"   Original Rows: {len(train_raw)}")
    print(f"   Rows after Z-Score cleanup: {len(train_clean)}")
    print(f"   Dropped {len(train_raw) - len(train_clean)} outliers.")

    results_remove = {}
    if len(train_clean) > 0:
        results_remove = evaluate_approach(train_clean.copy(), test_raw.copy(), target=TARGET)

        if results_remove:
            plt.figure()
            plot_multibar_chart(
                ["NB", "KNN"], results_remove, title="Remove Outliers (Z-Score) Evaluation", percentage=True
            )
            plt.savefig("images/traffic_outliers_zscore_remove_eval.png")
            print("   Chart saved: images/traffic_outliers_zscore_remove_eval.png")
    else:
        print("   WARNING: Z-Score removed ALL rows. Check data or threshold.")

    # _________________________________ COMPARISON & SELECTION _________________________________
    
    acc_keep_nb = results_keep['accuracy'][0]
    acc_keep_knn = results_keep['accuracy'][1]
    
    # Handle case where remove failed
    if results_remove:
        acc_remove_nb = results_remove['accuracy'][0]
        acc_remove_knn = results_remove['accuracy'][1]
    else:
        acc_remove_nb = 0
        acc_remove_knn = 0

    best_score_keep = max(acc_keep_nb, acc_keep_knn)
    best_score_remove = max(acc_remove_nb, acc_remove_knn)

    print(f"\nRESULTS (Best Accuracy): Keep={best_score_keep:.4f} vs Remove={best_score_remove:.4f}")
    
    best_df = None
    best_approach_name = ""
    best_model_type = ""
    
    if best_score_remove > best_score_keep:
        print("🏆 WINNER: REMOVE Outliers (Z-Score)")
        best_df = train_clean
        best_approach_name = "Remove Outliers (Z-Score)"
        best_model_type = "NB" if acc_remove_nb > acc_remove_knn else "KNN"
    else:
        print("🏆 WINNER: KEEP Outliers")
        best_df = train_raw
        best_approach_name = "Keep Outliers"
        best_model_type = "NB" if acc_keep_nb > acc_keep_knn else "KNN"

    # Save Best Dataset
    best_df.to_csv(OUTPUT_TRAIN, index=False)
    test_raw.to_csv(OUTPUT_TEST, index=False)
    print(f"   Best dataset saved to {OUTPUT_TRAIN}")

    # _________________________________ CONFUSION MATRIX (Best Model) _________________________________
    print(f"\nGenerating Confusion Matrix for the Winner: {best_model_type} ({best_approach_name})...")
    
    cnf_mtx, labels = get_confusion_matrix(best_df.copy(), test_raw.copy(), TARGET, best_model_type)
    
    plt.figure()
    # Corrected call without 'title' argument inside
    plot_confusion_matrix(cnf_mtx, labels)
    plt.title(f"Confusion Matrix - {best_model_type} ({best_approach_name})")
    
    plt.tight_layout()
    plt.savefig("images/traffic_outliers_zscore_best_cm.png")
    print("   CM saved to images/traffic_outliers_zscore_best_cm.png")
    
    plt.show()

if __name__ == "__main__":
    run_zscore_analysis()