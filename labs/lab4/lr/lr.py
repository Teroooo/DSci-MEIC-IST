import sys
import os
from numpy import array, ndarray
from matplotlib.pyplot import figure, savefig, show
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, read_train_test_from_files, \
	plot_evaluation_results, plot_multiline_chart

def logistic_regression_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
) -> tuple[LogisticRegression | None, dict]:
    nr_iterations: list[int] = [lag] + [
        i for i in range(2 * lag, nr_max_iterations + 1, lag)
    ]

    penalty_types: list[str] = ["l1", "l2"]  # only available if optimizer='liblinear'

    best_model = None
    best_params: dict = {"name": "LR", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    for type in penalty_types:
        warm_start = False
        y_tst_values: list[float] = []
        for j in range(len(nr_iterations)):
            clf = LogisticRegression(
                penalty=type,
                max_iter=lag,
                warm_start=warm_start,
                solver="liblinear",
                verbose=False,
            )
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            warm_start = True
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params["params"] = (type, nr_iterations[j])
                best_model: LogisticRegression = clf
            # print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
        values[type] = y_tst_values
    plot_multiline_chart(
        nr_iterations,
        values,
        title=f"LR models ({metric})",
        xlabel="nr iterations",
        ylabel=metric,
        percentage=True,
    )
    print(
        f'LR best for {best_params["params"][1]} iterations (penalty={best_params["params"][0]})'
    )

    return best_model, best_params


file_tag = "traffic"
train_filename = '../../../../traffic_final_train.csv'
test_filename = '../../../../traffic_final_test.csv'
target = "crash_type"
#file_tag = "cflights"
#train_filename = '../../../../cflights_train.csv'
#test_filename = '../../../../cflights_test.csv'
#target = "Cancelled"
eval_metric = "accuracy"
sample_frac: float = 0.2  # fraction of data to use (0 < sample_frac <= 1]


trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
    train_filename, test_filename, target
)

# Optional subsampling of train and test sets
if not (0 < sample_frac <= 1.0):
    raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")
if sample_frac < 1.0:
    trnX, _, trnY, _ = train_test_split(
        trnX, trnY, train_size=sample_frac, stratify=trnY, random_state=42
    )
    tstX, _, tstY, _ = train_test_split(
        tstX, tstY, train_size=sample_frac, stratify=tstY, random_state=42
    )

print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")

figure()
best_model, params = logistic_regression_study(
    trnX,
    trnY,
    tstX,
    tstY,
    nr_max_iterations=5000,
    lag=500,
    metric=eval_metric,
)
savefig(f"images/{file_tag}_lr_{eval_metric}_study.png")
show()

prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'images/{file_tag}_lr_{params["name"]}_best_{params["metric"]}_eval.png')
show()

type: str = params["params"][0]
nr_iterations: list[int] = [i for i in range(100, 1001, 100)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric = "accuracy"

warm_start = False
for n in nr_iterations:
    clf = LogisticRegression(
        warm_start=warm_start,
        penalty=type,
        max_iter=n,
        solver="liblinear",
        verbose=False,
    )
    clf.fit(trnX, trnY)
    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))
    warm_start = True

figure()
plot_multiline_chart(
    nr_iterations,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"LR overfitting study for penalty={type}",
    xlabel="nr_iterations",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"images/{file_tag}_lr_{eval_metric}_overfitting.png")

# --- Variables' importance (coefficients magnitude) ---
if best_model is not None:
    # coef_ shape: (n_classes, n_features) or (1, n_features)
    import numpy as _np

    coef = best_model.coef_
    # Aggregate importance across classes by mean absolute value
    importance = _np.mean(_np.abs(coef), axis=0)
    feature_names = vars if isinstance(vars, list) and len(vars) == len(importance) else [f"f{i}" for i in range(len(importance))]

    # Sort for a cleaner plot
    order = _np.argsort(importance)
    sorted_importance = importance[order]
    sorted_features = [feature_names[i] for i in order]

    figure(figsize=(8, max(4, len(sorted_features) * 0.25)))
    plt.barh(sorted_features, sorted_importance)
    plt.title("LR Feature Importance (|coef|, mean across classes)")
    plt.xlabel("Importance")
    plt.tight_layout()
    savefig(f"images/{file_tag}_lr_feature_importance.png")