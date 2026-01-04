import pandas as pd
from numpy import ndarray
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_multibar_chart, run_NB, run_KNN

# Define the metrics we want to compare
CLASS_EVAL_METRICS = ["accuracy", "recall", "precision", "auc", "f1"]

# _________________________________ CONFIGURATION _________________________________
TRAIN_PATH = "traffic_accidents_train.csv"
TEST_PATH = "traffic_accidents_test.csv"
TARGET = "crash_type"
OUTPUT_TRAIN = "traffic_accidents_mvi_train.csv"
OUTPUT_TEST = "traffic_accidents_mvi_test.csv"

def evaluate_approach(train: pd.DataFrame, test: pd.DataFrame, target: str = "class") -> dict[str, list]:
    """
    Evaluates NB and KNN models and returns a dictionary with all metrics 
    comparing both models: {'accuracy': [nb_acc, knn_acc], 'recall': ...}
    """
    # Prepare data
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    
    eval_dict: dict[str, list] = {}

    # Run Models
    # Note: We run with 'accuracy' as a default driver, but we extract all metrics below
    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric="accuracy")
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric="accuracy")

    # Collect all metrics into the format required for plot_multibar_chart
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            # We treat NB as the first bar, KNN as the second
            nb_score = eval_NB.get(met, 0.0)
            knn_score = eval_KNN.get(met, 0.0)
            eval_dict[met] = [nb_score, knn_score]
            
    return eval_dict

def run_mvi_analysis():
    # 1. Load Data
    print("Loading data...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    # _________________________________ APPROACH 1: DROP MISSING _________________________________
    print("\n--- APPROACH 1: DROP MISSING ---")
    train_drop = train_raw.dropna().copy()
    test_drop = test_raw.dropna().copy()
    
    # Evaluate
    print("   Evaluating Drop Approach...")
    results_drop = evaluate_approach(train_drop.copy(), test_drop.copy(), target=TARGET)
    
    # Plot
    if results_drop:
        figure()
        plot_multibar_chart(
            ["NB", "KNN"], 
            results_drop, 
            title="Drop Missing Evaluation", 
            percentage=True
        )
        savefig("images/traffic_mvi_drop_eval.png")
        print("   Graph saved to images/traffic_mvi_drop_eval.png")

    # _________________________________ APPROACH 2: IMPUTE (FREQUENT) _____________________________
    print("\n--- APPROACH 2: IMPUTE MISSING (Mode) ---")
    # Using pandas filling for simplicity and consistency
    train_imp = train_raw.fillna(train_raw.mode().iloc[0]).copy()
    test_imp = test_raw.fillna(train_raw.mode().iloc[0]).copy() # Important: use TRAIN mode for TEST
    
    # Evaluate
    print("   Evaluating Imputation Approach...")
    results_imp = evaluate_approach(train_imp.copy(), test_imp.copy(), target=TARGET)
    
    # Plot
    if results_imp:
        figure()
        plot_multibar_chart(
            ["NB", "KNN"], 
            results_imp, 
            title="Imputation (Mode) Evaluation", 
            percentage=True
        )
        savefig("images/traffic_mvi_impute_eval.png")
        print("   Graph saved to images/traffic_mvi_impute_eval.png")

    # _________________________________ COMPARISON _________________________________
    # We compare based on Accuracy (or your preferred metric)
    # results_drop['accuracy'] is a list [NB_acc, KNN_acc]
    acc_drop = results_drop['accuracy'][1] if results_drop else 0 # KNN accuracy
    acc_imp = results_imp['accuracy'][1] if results_imp else 0    # KNN accuracy
    
    print(f"\nKNN Accuracy (Drop):   {acc_drop:.4f}")
    print(f"KNN Accuracy (Impute): {acc_imp:.4f}")
    
    if acc_imp > acc_drop:
        print("\n🏆 WINNER: Imputation Approach")
        train_imp.to_csv(OUTPUT_TRAIN, index=False)
        test_imp.to_csv(OUTPUT_TEST, index=False)
    else:
        print("\n🏆 WINNER: Drop Approach")
        train_drop.to_csv(OUTPUT_TRAIN, index=False)
        test_drop.to_csv(OUTPUT_TEST, index=False)
        
    # Note on F1 Score
    print("\nNOTE: If F1-Score for KNN is 0, it is likely because the data is unscaled.")
    print("KNN requires Scaling (normalization) which is the next step in data preparation.")
    
    show()

if __name__ == "__main__":
    run_mvi_analysis()