# DSci - Data Science Course

Repository for Data Science course labs and projects.

## Project Structure

```
DSci/
├── classification/           # Dataset folder (sibling to labs/)
│   ├── traffic_accidents.csv
│   └── Combined_Flights_2022.csv
├── labs/
│   ├── lab1/
│   │   ├── train.py         # Main training script
│   │   ├── results/         # Generated results (auto-created)
│   │   │   ├── traffic_accidents/
│   │   │   └── Combined_Flights_2022/
│   │   └── report/          # LaTeX report files
│   └── venv/                # Python virtual environment
├── forecasting/             # Forecasting datasets
└── requirements.txt         # Python dependencies
```

## Setup Instructions

### 1. Create Dataset Folder

Create a `classification` folder as a sibling to `labs`:

```bash
mkdir classification
```

Place your CSV dataset files in the `classification` folder:
- `traffic_accidents.csv`
- `Combined_Flights_2022.csv`

### 2. Create Virtual Environment

Navigate to the `labs` folder and create a virtual environment:

**On Linux/Mac/WSL:**
```bash
cd labs
python3 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
cd labs
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Lab 1: Baseline Models

### Goal
Train baseline classification models on raw data to assess performance and establish comparison benchmarks.

### Datasets

1. **Traffic Accidents** - Predict injury severity (`most_severe_injury`)
   - Multi-class classification (5 classes: FATAL, INCAPACITATING, NO INDICATION, NONINCAPACITATING, REPORTED NOT EVIDENT)
   
2. **Combined Flights 2022** - Predict flight cancellation (`Cancelled`)
   - Binary classification (True/False)

### Running the Training Script

Navigate to `labs/lab1` and run:

**Train on Traffic Accidents dataset:**
```bash
python train.py 1
```

**Train on Combined Flights dataset:**
```bash
python train.py 2
```

### Models Trained

The script trains and evaluates 5 classification techniques:
- Naïve Bayes
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Multi-layer Perceptron (MLP)

Each model undergoes hyperparameter tuning using GridSearchCV with 5-fold cross-validation.

### Output Files

Results are saved to `labs/lab1/results/[dataset_name]/`:

**Text Files:**
- `model_results.csv` - Performance metrics (Accuracy, Precision, Recall, F1-Score)
- `best_hyperparameters.txt` - Best hyperparameters found for each model

**Charts (10 total):**

*Hyperparameter Studies (5):*
- `nb_hyperparameter_study.png` - Naïve Bayes var_smoothing
- `lr_hyperparameter_study.png` - Logistic Regression C and solver
- `knn_hyperparameter_study.png` - KNN neighbors, weights, and metrics
- `dt_hyperparameter_study.png` - Decision Tree depth and split parameters
- `mlp_hyperparameter_study.png` - MLP hidden layers and alpha

*Performance Charts (5):*
- `naive_bayes_performance.png` - Confusion matrix + per-class metrics
- `logistic_regression_performance.png` - Confusion matrix + per-class metrics
- `knn_performance.png` - Confusion matrix + per-class metrics
- `decision_tree_performance.png` - Confusion matrix + per-class metrics
- `multi_layer_perceptron_performance.png` - Confusion matrix + per-class metrics

*Overall Comparison (1):*
- `model_performance_comparison.png` - All models comparison

### Data Processing Pipeline

1. **Load data** from CSV
2. **Drop missing values**:
   - Remove completely empty columns
   - Remove rows with any missing values
3. **Encode categorical target** to numeric (if needed)
4. **Keep only numeric columns** (discard non-numeric features)
5. **Split data** into train (70%) and test (30%) sets
6. **Standardize features** using StandardScaler
7. **Train models** with hyperparameter tuning
8. **Evaluate** on test set
9. **Generate charts and reports**

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

See `requirements.txt` for specific versions.

## Notes

- Training uses `n_jobs=1` for stability in WSL environments
- Progress is shown with `verbose=1` during GridSearchCV
- All random states are set to 42 for reproducibility
- Results are automatically organized by dataset name
