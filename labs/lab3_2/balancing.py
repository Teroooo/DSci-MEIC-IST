import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from dslabs_functions import plot_multibar_chart, run_NB, run_KNN, plot_confusion_matrix

# _________________________________ CONFIGURATION _________________________________
# Input files (Output from Scaling Step - Standard Scaler won)
TRAIN_PATH = "traffic_accidents_scaled_train.csv"
TEST_PATH = "traffic_accidents_scaled_test.csv"

# Output files
OUTPUT_TRAIN = "traffic_accidents_balanced_train.csv"
OUTPUT_TEST = "traffic_accidents_balanced_test.csv"

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

def run_balancing_analysis():
    print("Loading Data (from Scaling Step)...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    
    # Separate Target for Sampling
    target_col = train_raw[TARGET]
    data_cols = train_raw.drop(columns=[TARGET])

    # _________________________________ APPROACH 1: Random Undersampling ________________________
    print("\n--- APPROACH 1: Random Undersampling ---")
    
    # Define sampler
    # sampling_strategy='majority' means resample only the majority class
    undersampler = RandomUnderSampler(sampling_strategy='not minority', random_state=42)
    
    trnX_under, trnY_under = undersampler.fit_resample(data_cols, target_col)
    
    # Reconstruct DataFrame
    train_under = pd.concat([pd.DataFrame(trnX_under, columns=data_cols.columns), 
                             pd.Series(trnY_under, name=TARGET)], axis=1)
    
    print(f"   Original shape: {train_raw.shape}")
    print(f"   Under-sampled shape: {train_under.shape}")
    
    results_under = evaluate_approach(train_under.copy(), test_raw.copy(), target=TARGET)
    
    if results_under:
        plt.figure()
        plot_multibar_chart(
            ["NB", "KNN"], results_under, title="Undersampling Evaluation", percentage=True
        )
        plt.savefig("images/traffic_balancing_under_eval.png")
        print("   Chart saved: images/traffic_balancing_under_eval.png")

    # _________________________________ APPROACH 2: SMOTE (Over-sampling) _______________________
    print("\n--- APPROACH 2: SMOTE (Over-sampling) ---")
    
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    
    trnX_smote, trnY_smote = smote.fit_resample(data_cols, target_col)
    
    # Reconstruct DataFrame
    train_smote = pd.concat([pd.DataFrame(trnX_smote, columns=data_cols.columns), 
                             pd.Series(trnY_smote, name=TARGET)], axis=1)
    
    print(f"   SMOTE shape: {train_smote.shape}")
    
    results_smote = evaluate_approach(train_smote.copy(), test_raw.copy(), target=TARGET)
    
    if results_smote:
        plt.figure()
        plot_multibar_chart(
            ["NB", "KNN"], results_smote, title="SMOTE Evaluation", percentage=True
        )
        plt.savefig("images/traffic_balancing_smote_eval.png")
        print("   Chart saved: images/traffic_balancing_smote_eval.png")

    # _________________________________ COMPARISON & SELECTION _________________________________
    # We compare based on Recall or F1 usually, but let's look at Accuracy/F1
    # Often in balancing, Accuracy drops but Recall increases.
    
    # Retrieve Scores (metrics: 0=Accuracy, 1=Recall, 2=Precision, 3=AUC, 4=F1 in CLASS_EVAL_METRICS list logic? 
    # Actually results_under['accuracy'] gives [nb, knn].
    
    # Let's pick the best model based on Recall (often preferred in accidents) or pure Accuracy?
    # Lab guidelines usually ask for "best performance". Let's stick to Accuracy or F1 if Acc is similar.
    
    acc_under_knn = results_under['accuracy'][1]
    acc_smote_knn = results_smote['accuracy'][1]
    
    print(f"\nRESULTS (KNN Accuracy): Under={acc_under_knn:.4f} vs SMOTE={acc_smote_knn:.4f}")
    
    best_df = None
    best_approach_name = ""
    best_model_type = ""
    
    # Determine Winner
    if acc_smote_knn > acc_under_knn:
        print("🏆 WINNER: SMOTE")
        best_df = train_smote
        best_approach_name = "SMOTE"
        # Check if NB beat KNN inside SMOTE
        if results_smote['accuracy'][0] > results_smote['accuracy'][1]:
            best_model_type = "NB"
        else:
            best_model_type = "KNN"
    else:
        print("🏆 WINNER: Undersampling")
        best_df = train_under
        best_approach_name = "Undersampling"
        # Check if NB beat KNN inside Under
        if results_under['accuracy'][0] > results_under['accuracy'][1]:
            best_model_type = "NB"
        else:
            best_model_type = "KNN"

    # Save Best Dataset
    best_df.to_csv(OUTPUT_TRAIN, index=False)
    test_raw.to_csv(OUTPUT_TEST, index=False) # Test data is NEVER balanced
    print(f"   Best dataset saved to {OUTPUT_TRAIN}")

    # _________________________________ CONFUSION MATRIX (Best Model) _________________________________
    print(f"\nGenerating Confusion Matrix for the Winner: {best_model_type} ({best_approach_name})...")
    
    cnf_mtx, labels = get_confusion_matrix(best_df.copy(), test_raw.copy(), TARGET, best_model_type)
    
    plt.figure()
    plot_confusion_matrix(cnf_mtx, labels)
    plt.title(f"Confusion Matrix - {best_model_type} ({best_approach_name})")
    
    plt.tight_layout()
    plt.savefig("images/traffic_balancing_best_cm.png")
    print("   CM saved to images/traffic_balancing_best_cm.png")
    
    plt.show()

if __name__ == "__main__":
    run_balancing_analysis()