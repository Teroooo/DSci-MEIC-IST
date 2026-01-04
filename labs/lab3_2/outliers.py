import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from dslabs_functions import plot_multibar_chart, run_NB, run_KNN, plot_confusion_matrix

# _________________________________ CONFIGURATION _________________________________
TRAIN_PATH = "traffic_accidents_mvi_train.csv"
TEST_PATH = "traffic_accidents_mvi_test.csv"

# Output files
OUTPUT_TRAIN = "traffic_accidents_outliers_train.csv"
OUTPUT_TEST = "traffic_accidents_outliers_test.csv"

TARGET = "crash_type"
CLASS_EVAL_METRICS = ["accuracy", "recall", "precision", "auc", "f1"]

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
        # Default dslabs params for consistency
        clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean') 
        
    clf.fit(trn_x, trn_y)
    prd_y = clf.predict(tst_x)
    labels = list(pd.unique(trn_y))
    labels.sort()
    
    return confusion_matrix(tst_y, prd_y, labels=labels), labels

def run_outlier_analysis():
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
        plt.savefig("images/traffic_outliers_keep_eval.png")
        print("   Chart saved: images/traffic_outliers_keep_eval.png")

    # _________________________________ APPROACH 2: REMOVE OUTLIERS (IsolationForest) ________________________
    print("\n--- APPROACH 2: REMOVE OUTLIERS (Isolation Forest) ---")
    
    # Detect outliers ONLY on Training data
    # Contamination=0.01 means we expect ~1% of data to be outliers
    iso = IsolationForest(contamination=0.01, random_state=42)
    
    # We fit on the whole train set
    outlier_pred = iso.fit_predict(train_raw)
    
    # -1 are outliers, 1 are inliers. We keep only 1.
    mask = outlier_pred == 1
    train_clean = train_raw[mask].copy()
    
    print(f"   Original Rows: {len(train_raw)}, Rows after cleanup: {len(train_clean)}")
    print(f"   Dropped {len(train_raw) - len(train_clean)} outliers.")

    results_remove = evaluate_approach(train_clean.copy(), test_raw.copy(), target=TARGET)

    if results_remove:
        plt.figure()
        plot_multibar_chart(
            ["NB", "KNN"], results_remove, title="Remove Outliers Evaluation", percentage=True
        )
        plt.savefig("images/traffic_outliers_remove_eval.png")
        print("   Chart saved: images/traffic_outliers_remove_eval.png")

    # _________________________________ COMPARISON & SELECTION _________________________________
    
    # Retrieve Accuracies
    acc_keep_nb = results_keep['accuracy'][0]
    acc_keep_knn = results_keep['accuracy'][1]
    acc_remove_nb = results_remove['accuracy'][0]
    acc_remove_knn = results_remove['accuracy'][1]
    
    # Determine best score for each approach
    best_score_keep = max(acc_keep_nb, acc_keep_knn)
    best_score_remove = max(acc_remove_nb, acc_remove_knn)

    print(f"\nRESULTS (Best Accuracy): Keep={best_score_keep:.4f} vs Remove={best_score_remove:.4f}")
    
    best_df = None
    best_approach_name = ""
    best_model_type = ""
    
    if best_score_remove > best_score_keep:
        print("🏆 WINNER: REMOVE Outliers")
        best_df = train_clean
        best_approach_name = "Remove Outliers"
        # Determine if NB or KNN was better in this approach
        best_model_type = "NB" if acc_remove_nb > acc_remove_knn else "KNN"
    else:
        print("🏆 WINNER: KEEP Outliers")
        best_df = train_raw
        best_approach_name = "Keep Outliers"
        # Determine if NB or KNN was better in this approach
        best_model_type = "NB" if acc_keep_nb > acc_keep_knn else "KNN"

    # Save Best Dataset
    best_df.to_csv(OUTPUT_TRAIN, index=False)
    test_raw.to_csv(OUTPUT_TEST, index=False) # Test set is never touched by outlier removal
    print(f"   Best dataset saved to {OUTPUT_TRAIN}")

    # _________________________________ CONFUSION MATRIX (Best Model) _________________________________
    print(f"\nGenerating Confusion Matrix for the Winner: {best_model_type} ({best_approach_name})...")
    
    # Generate matrix
    cnf_mtx, labels = get_confusion_matrix(best_df.copy(), test_raw.copy(), TARGET, best_model_type)
    
    # Plot
    plt.figure()
    # FIX: Remove 'title' argument. We set title manually below.
    plot_confusion_matrix(cnf_mtx, labels)
    plt.gca().set_title(f"Confusion Matrix ({best_approach_name} - {best_model_type})")
    
    plt.savefig("images/traffic_outliers_best_cm.png")
    print("   CM saved to images/traffic_outliers_best_cm.png")
    
    plt.show()

if __name__ == "__main__":
    run_outlier_analysis()