from numpy import array, ndarray
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from matplotlib.pyplot import figure, savefig, show
from sklearn.model_selection import train_test_split
from pandas import read_csv

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart
from utils.dslabs_functions import plot_evaluation_results


file_tag = "Combined_flight_v1"
train_filename = "../../classification/Combined_flight_v1.csv"
target = "Cancelled"
eval_metric = "accuracy"

data = read_csv(train_filename)
labels = list(data[target].unique())
labels.sort()

X = data.drop(columns=[target])
y = data[target]
vars = data.columns.to_list()

trnX, tstX, trnY, tstY = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")


def naive_Bayes_study(
    trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, metric: str = "accuracy"
) -> tuple:
    estimators: dict = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
    }

    xvalues: list = []
    yvalues: list = []
    best_model = None
    best_params: dict = {"name": "", "metric": metric, "params": ()}
    best_performance = 0
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY: array = estimators[clf].predict(tstX)
        eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if eval - best_performance > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["name"] = clf
            best_params[metric] = eval
            best_model = estimators[clf]
        yvalues.append(eval)
        # print(f'NB {clf}')
    plot_bar_chart(
        xvalues,
        yvalues,
        title=f"Naive Bayes Models ({metric})",
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params


figure()
best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, eval_metric)
savefig(f"images/Combined_flight_v1/{file_tag}_nb_{eval_metric}_study.png")
show()

figure()
best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, "recall")
savefig(f"images/Combined_flight_v1/{file_tag}_nb_recall_study.png")
show()


prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'images/Combined_flight_v1/{file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')
show()