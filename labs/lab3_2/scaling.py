import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from dslabs_functions import plot_multibar_chart, run_NB, run_KNN, plot_confusion_matrix

# _________________________________ CONFIGURATION _________________________________
# Input files (Output from Outliers Analysis - Keep Outliers won)
TRAIN_PATH = "traffic_accidents_outliers_zscore_train.csv"
TEST_PATH = "traffic_accidents_outliers_zscore_test.csv"

# Output files
OUTPUT_TRAIN = "traffic_accidents_scaled_train.csv"
OUTPUT_TEST = "traffic_accidents_scaled_test.csv"

TARGET = "crash_type"
CLASS_EVAL_METRICS = ["accuracy", "recall", "precision", "auc", "f1"]

def evaluate_approach(train: pd.DataFrame, test: pd.DataFrame, target: str = "class") -> dict[str, list]:
    """
    Evaluates NB and KNN. Returns dictionary for plotting.
    """
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    
    eval_dict: dict[str, list] = {}

    # Run Models
    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric="accuracy")
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric="accuracy")

    # Structure data for dslabs plot
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            nb_score = eval_NB.get(met, 0.0)
            knn_score = eval_KNN.get(met, 0.0)
            eval_dict[met] = [nb_score, knn_score]
            
    return eval_dict

def get_confusion_matrix(train, test, target, model_type="KNN"):
    """
    Helper to generate the confusion matrix for the winner
    """
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

def scale_data(train, test, scaler):
    """
    Fits scaler on train, transforms train and test.
    Returns reconstructed DataFrames with column names and target.
    """
    # Separate Target
    trn_y = train[TARGET]
    tst_y = test[TARGET]
    trn_x = train.drop(columns=[TARGET])
    tst_x = test.drop(columns=[TARGET])
    
    # Fit & Transform
    # CRITICAL: Fit ONLY on Train to avoid data leakage
    scaler.fit(trn_x)
    
    trn_x_scaled = scaler.transform(trn_x)
    tst_x_scaled = scaler.transform(tst_x)
    
    # Reconstruct DataFrames
    train_scaled = pd.DataFrame(trn_x_scaled, columns=trn_x.columns)
    train_scaled[TARGET] = trn_y.values # Add target back
    
    test_scaled = pd.DataFrame(tst_x_scaled, columns=tst_x.columns)
    test_scaled[TARGET] = tst_y.values
    
    return train_scaled, test_scaled

def run_scaling_analysis():
    print("Loading Data (from Outliers Step)...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    # _________________________________ APPROACH 1: MinMax Scaler _________________________________
    print("\n--- APPROACH 1: MinMax Scaler ---")
    scaler_mm = MinMaxScaler()
    train_mm, test_mm = scale_data(train_raw.copy(), test_raw.copy(), scaler_mm)
    
    results_mm = evaluate_approach(train_mm.copy(), test_mm.copy(), target=TARGET)
    
    if results_mm:
        plt.figure()
        plot_multibar_chart(
            ["NB", "KNN"], results_mm, title="MinMax Scaler Evaluation", percentage=True
        )
        plt.savefig("images/traffic_scaling_minmax_eval.png")
        print("   Chart saved: images/traffic_scaling_minmax_eval.png")

    # _________________________________ APPROACH 2: Standard Scaler _________________________________
    print("\n--- APPROACH 2: Standard Scaler ---")
    scaler_std = StandardScaler()
    train_std, test_std = scale_data(train_raw.copy(), test_raw.copy(), scaler_std)
    
    results_std = evaluate_approach(train_std.copy(), test_std.copy(), target=TARGET)
    
    if results_std:
        plt.figure()
        plot_multibar_chart(
            ["NB", "KNN"], results_std, title="Standard Scaler Evaluation", percentage=True
        )
        plt.savefig("images/traffic_scaling_standard_eval.png")
        print("   Chart saved: images/traffic_scaling_standard_eval.png")

    # _________________________________ COMPARISON & SELECTION _________________________________
    
    # Usually we compare KNN performance, as NB is less affected by scaling
    acc_mm_knn = results_mm['accuracy'][1]
    acc_std_knn = results_std['accuracy'][1]
    
    # Also track NB just in case
    acc_mm_nb = results_mm['accuracy'][0]
    acc_std_nb = results_std['accuracy'][0]

    print(f"\nRESULTS (KNN Accuracy): MinMax={acc_mm_knn:.4f} vs Standard={acc_std_knn:.4f}")
    
    best_df_train = None
    best_df_test = None
    best_approach_name = ""
    best_model_type = "" # Will store if NB or KNN is the overall winner
    
    # Determine winner between scalers based on KNN (since scaling is mostly for KNN)
    if acc_std_knn > acc_mm_knn:
        print("🏆 WINNER: Standard Scaler")
        best_df_train = train_std
        best_df_test = test_std
        best_approach_name = "Standard Scaler"
        
        # Who is better inside Standard Scaler? NB or KNN?
        if acc_std_nb > acc_std_knn:
            best_model_type = "NB"
        else:
            best_model_type = "KNN"
    else:
        print("🏆 WINNER: MinMax Scaler")
        best_df_train = train_mm
        best_df_test = test_mm
        best_approach_name = "MinMax Scaler"
        
        # Who is better inside MinMax?
        if acc_mm_nb > acc_mm_knn:
            best_model_type = "NB"
        else:
            best_model_type = "KNN"

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
    plt.savefig("images/traffic_scaling_best_cm.png")
    print("   CM saved to images/traffic_scaling_best_cm.png")
    
    plt.show()

if __name__ == "__main__":
    run_scaling_analysis()