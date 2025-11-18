from matplotlib.font_manager import FontProperties
from typing import Callable

# Color configuration for plots
LINE_COLOR = '#1f77b4'      # Blue
FILL_COLOR = '#aec7e8'      # Light blue
ACTIVE_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
]

# Font configuration
FONT_TEXT = FontProperties(size='xx-small')

# Chart configuration
HEIGHT: int = 4
NR_COLUMNS: int = 3

# Outlier detection configuration
NR_STDEV: int = 2
IQR_FACTOR: float = 1.5

# Classification evaluation metrics
CLASS_EVAL_METRICS: dict[str, Callable] = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "auc": roc_auc_score,
    "f1": f1_score,
}