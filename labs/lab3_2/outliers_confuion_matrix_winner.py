import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from dslabs_functions import plot_confusion_matrix

# _________________________________ CONFIGURATION _________________________________
# Como "Keep Outliers" venceu, usamos os ficheiros da fase anterior (MVI)
TRAIN_PATH = "traffic_accidents_mvi_train.csv"
TEST_PATH = "traffic_accidents_mvi_test.csv"
TARGET = "crash_type"

def run_cm_only():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Prepare Data
    trnY = train.pop(TARGET).values
    trnX = train.values
    tstY = test.pop(TARGET).values
    tstX = test.values
    
    labels = list(pd.unique(trnY))
    labels.sort()

    # _________________________________ MODEL: NAIVE BAYES _________________________________
    print("Training Naive Bayes (Winner of Keep Outliers)...")
    clf = GaussianNB()
    clf.fit(trnX, trnY)
    
    # Predict
    prdY = clf.predict(tstX)
    
    # Generate Matrix
    cnf_mtx = confusion_matrix(tstY, prdY, labels=labels)

    # _________________________________ PLOT _________________________________
    print("Plotting Confusion Matrix...")
    plt.figure()
    
    # CORREÇÃO: Removemos o argumento 'title' de dentro da função
    plot_confusion_matrix(cnf_mtx, labels)
    
    # Adicionamos o título externamente
    plt.title("Confusion Matrix - NB (Keep Outliers)")
    
    plt.tight_layout()
    plt.savefig("images/traffic_outliers_best_cm.png")
    print("Graph saved to images/traffic_outliers_best_cm.png")
    plt.show()

if __name__ == "__main__":
    run_cm_only()
