import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_clean_data(filepath, target_column, dataset_name=None):
    """Load data and perform cleaning operations"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Original shape: {df.shape}")
    
    # Sample large datasets for faster training (only for flights)
    if dataset_name == 'Combined_Flights_2022' and len(df) > 500000:
        print(f"Sampling 500,000 records from {len(df)} for faster training...")
        df = df.sample(n=500000, random_state=42)
        print(f"Sampled shape: {df.shape}")
    
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Drop columns that are completely empty
    df = df.dropna(axis=1, how='all')
    
    # PREPROCESSING: Encode categorical target to numeric BEFORE dropping rows
    # This converts text labels to numbers, making the target "numeric"
    if target_column in df.columns:
        # Handle boolean columns
        if df[target_column].dtype == 'bool':
            print(f"\nConverting boolean target '{target_column}' to numeric...")
            df[target_column] = df[target_column].astype(int)
            print(f"  False -> 0, True -> 1")
        # Handle string/object columns
        elif df[target_column].dtype == 'object':
            print(f"\nEncoding categorical target '{target_column}' to numeric...")
            le = LabelEncoder()
            df[target_column] = le.fit_transform(df[target_column])
            print(f"  Classes: {list(le.classes_)}")
            print(f"  Encoded as: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Keep only numeric columns (now includes the encoded target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    
    # Dataset-specific cleaning strategy
    if dataset_name == 'Combined_Flights_2022':
        # For flights dataset: drop columns with ANY missing to preserve Cancelled=1 rows
        print(f"\n[Flights Dataset] Dropping columns with missing values...")
        print(f"Columns before: {df.shape[1]}")
        df = df.dropna(axis=1, how='any')
        print(f"Columns after: {df.shape[1]}")
    else:
        # For other datasets: follow baseline requirements (drop rows with missing)
        print(f"\n[Baseline] Dropping all records with missing values...")
        df = df.dropna(axis=0, how='any')
    
    print(f"\nFinal cleaned shape: {df.shape}")
    print(f"Features: {[col for col in df.columns if col != target_column]}")
    print(f"Target: {target_column}")
    
    # Check target distribution
    if target_column in df.columns:
        print(f"\nTarget distribution:")
        print(df[target_column].value_counts().sort_index())
        n_classes = df[target_column].nunique()
        print(f"Number of classes: {n_classes}")
        
        if n_classes < 2:
            raise ValueError(
                f"ERROR: Target '{target_column}' has only {n_classes} class after cleaning!\n"
                f"This usually means one class was completely removed when dropping missing values.\n"
                f"Try a different target column or modify the cleaning process."
            )
    
    return df

def prepare_data(df, target_column):
    """Prepare features and target, split data"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics, y_pred

def plot_model_performance(y_test, y_pred, model_name, output_dir):
    """Plot confusion matrix and per-class performance metrics for a model"""
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=True)
    axes[0].set_title(f'{model_name}: Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # 2. Per-class Performance Metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    if classes:
        metrics_data = {
            'Precision': [report[c]['precision'] for c in classes],
            'Recall': [report[c]['recall'] for c in classes],
            'F1-Score': [report[c]['f1-score'] for c in classes]
        }
        
        x = np.arange(len(classes))
        width = 0.25
        
        axes[1].bar(x - width, metrics_data['Precision'], width, label='Precision', color='steelblue')
        axes[1].bar(x, metrics_data['Recall'], width, label='Recall', color='lightcoral')
        axes[1].bar(x + width, metrics_data['F1-Score'], width, label='F1-Score', color='lightgreen')
        
        axes[1].set_ylabel('Score')
        axes[1].set_title(f'{model_name}: Per-Class Performance')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes, rotation=45, ha='right')
        axes[1].legend()
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save with model-specific filename
    filename = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_performance.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def train_naive_bayes(X_train, X_test, y_train, y_test, output_dir):
    """Train Naïve Bayes with hyperparameter tuning"""
    print("\n=== Training Naïve Bayes ===")
    
    param_grid = {
        'var_smoothing': np.logspace(-10, -1, 10)  # Reduced from 20 to 10
    }
    
    grid_search = GridSearchCV(
        GaussianNB(), param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Plot hyperparameter study
    results = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=(10, 6))
    plt.plot(results['param_var_smoothing'], results['mean_test_score'], 'b-o')
    plt.xscale('log')
    plt.xlabel('var_smoothing')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Naïve Bayes: Hyperparameter Study')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nb_hyperparameter_study.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    metrics, y_pred = evaluate_model(best_model, X_test, y_test, 'Naïve Bayes')
    
    # Plot performance
    plot_model_performance(y_test, y_pred, 'Naïve Bayes', output_dir)
    
    return best_model, metrics, grid_search.best_params_

def train_logistic_regression(X_train, X_test, y_train, y_test, output_dir):
    """Train Logistic Regression with hyperparameter tuning"""
    print("\n=== Training Logistic Regression ===")
    
    param_grid = {
        'C': np.logspace(-2, 2, 5),  # Reduced from 10 to 5 values
        'solver': ['lbfgs'],  # Use only lbfgs for speed
        'max_iter': [500]  # Reduced from 1000
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Plot hyperparameter study
    results = pd.DataFrame(grid_search.cv_results_)
    for solver in ['lbfgs', 'liblinear']:
        mask = results['param_solver'] == solver
        plt.plot(results[mask]['param_C'], results[mask]['mean_test_score'], 
                 marker='o', label=solver)
    plt.xscale('log')
    plt.xlabel('C (Regularization)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Logistic Regression: Hyperparameter Study')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_hyperparameter_study.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    metrics, y_pred = evaluate_model(best_model, X_test, y_test, 'Logistic Regression')
    
    # Plot performance
    plot_model_performance(y_test, y_pred, 'Logistic Regression', output_dir)
    
    return best_model, metrics, grid_search.best_params_

def train_knn(X_train, X_test, y_train, y_test, output_dir):
    """Train KNN with hyperparameter tuning"""
    print("\n=== Training KNN ===")
    
    param_grid = {
        'n_neighbors': [3, 7],  # 2 k values for graph
        'weights': ['uniform', 'distance'],  # Both weighting schemes
        'metric': ['euclidean']  # Just euclidean
    }
    
    grid_search = GridSearchCV(
        KNeighborsClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Plot hyperparameter study
    results = pd.DataFrame(grid_search.cv_results_)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    for weight in ['uniform', 'distance']:
        mask = results['param_weights'] == weight
        ax[0 if weight == 'uniform' else 1].plot(
            results[mask]['param_n_neighbors'], 
            results[mask]['mean_test_score'],
            marker='o', label='euclidean'
        )
        ax[0 if weight == 'uniform' else 1].set_xlabel('Number of Neighbors')
        ax[0 if weight == 'uniform' else 1].set_ylabel('Cross-Validation Accuracy')
        ax[0 if weight == 'uniform' else 1].set_title(f'KNN: {weight} weights')
        ax[0 if weight == 'uniform' else 1].legend()
        ax[0 if weight == 'uniform' else 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'knn_hyperparameter_study.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    metrics, y_pred = evaluate_model(best_model, X_test, y_test, 'KNN')
    
    # Plot performance
    plot_model_performance(y_test, y_pred, 'KNN', output_dir)
    
    return best_model, metrics, grid_search.best_params_

def train_decision_tree(X_train, X_test, y_train, y_test, output_dir):
    """Train Decision Tree with hyperparameter tuning"""
    print("\n=== Training Decision Tree ===")
    
    param_grid = {
        'max_depth': [5, 10, 15],  # Reduced options
        'min_samples_split': [2, 10],  # Reduced from 4 to 2 options
        'min_samples_leaf': [1, 4],  # Reduced from 4 to 2 options
        'criterion': ['gini']  # Use only gini for speed
    }
    
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Plot hyperparameter study
    results = pd.DataFrame(grid_search.cv_results_)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    for criterion in ['gini', 'entropy']:
        mask = results['param_criterion'] == criterion
        depth_scores = results[mask].groupby('param_max_depth')['mean_test_score'].mean()
        ax[0].plot(range(len(depth_scores)), depth_scores.values, marker='o', label=criterion)
    
    ax[0].set_xlabel('Max Depth')
    ax[0].set_xticklabels([str(d) for d in depth_scores.index])
    ax[0].set_ylabel('Cross-Validation Accuracy')
    ax[0].set_title('Decision Tree: Max Depth vs Accuracy')
    ax[0].legend()
    ax[0].grid(True)
    
    split_scores = results.groupby('param_min_samples_split')['mean_test_score'].mean()
    ax[1].plot(split_scores.index, split_scores.values, marker='o', color='green')
    ax[1].set_xlabel('Min Samples Split')
    ax[1].set_ylabel('Cross-Validation Accuracy')
    ax[1].set_title('Decision Tree: Min Samples Split vs Accuracy')
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dt_hyperparameter_study.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    metrics, y_pred = evaluate_model(best_model, X_test, y_test, 'Decision Tree')
    
    # Plot performance
    plot_model_performance(y_test, y_pred, 'Decision Tree', output_dir)
    
    return best_model, metrics, grid_search.best_params_

def train_mlp(X_train, X_test, y_train, y_test, output_dir):
    """Train Multi-layer Perceptron with hyperparameter tuning"""
    print("\n=== Training Multi-layer Perceptron ===")
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],  # Reduced from 4 to 2 options
        'activation': ['relu'],  # Use only relu
        'alpha': [0.0001, 0.001],  # Reduced from 3 to 2 values
        'learning_rate': ['adaptive'],  # Use only adaptive
        'max_iter': [500]  # Reduced from 1000
    }
    
    grid_search = GridSearchCV(
        MLPClassifier(random_state=42, early_stopping=True), 
        param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Plot hyperparameter study
    results = pd.DataFrame(grid_search.cv_results_)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hidden layer sizes
    layer_scores = results.groupby('param_hidden_layer_sizes')['mean_test_score'].mean().sort_values(ascending=False)
    ax[0].barh(range(len(layer_scores)), layer_scores.values)
    ax[0].set_yticks(range(len(layer_scores)))
    ax[0].set_yticklabels([str(l) for l in layer_scores.index])
    ax[0].set_xlabel('Cross-Validation Accuracy')
    ax[0].set_title('MLP: Hidden Layer Sizes')
    ax[0].grid(True, axis='x')
    
    # Alpha (regularization)
    for activation in ['relu', 'tanh']:
        mask = results['param_activation'] == activation
        alpha_scores = results[mask].groupby('param_alpha')['mean_test_score'].mean()
        ax[1].plot(alpha_scores.index, alpha_scores.values, marker='o', label=activation)
    
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Alpha (Regularization)')
    ax[1].set_ylabel('Cross-Validation Accuracy')
    ax[1].set_title('MLP: Alpha vs Accuracy')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mlp_hyperparameter_study.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    metrics, y_pred = evaluate_model(best_model, X_test, y_test, 'Multi-layer Perceptron')
    
    # Plot performance
    plot_model_performance(y_test, y_pred, 'Multi-layer Perceptron', output_dir)
    
    return best_model, metrics, grid_search.best_params_

def plot_performance_comparison(results_df, output_dir):
    """Plot performance comparison across all models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.set_xlabel('')
        ax.set_ylim([0, 1.1])
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def get_dataset_config(dataset_choice):
    """Get dataset configuration based on user choice"""
    datasets = {
        '1': {
            'path': '../../classification/traffic_accidents.csv',
            'name': 'traffic_accidents',
            'target': 'crash_type'  # Predict type of crash
        },
        '2': {
            'path': '../../classification/Combined_Flights_2022.csv',
            'name': 'Combined_Flights_2022',
            'target': 'Cancelled'  # Predict if flight will be cancelled
        }
    }
    
    return datasets.get(dataset_choice)

def auto_detect_target(df):
    """Auto-detect suitable target column for classification"""
    # Look for categorical columns with reasonable number of classes
    for col in df.columns:
        if df[col].dtype == 'object':
            n_unique = df[col].nunique()
            # Good target: categorical with 2-20 unique values
            if 2 <= n_unique <= 20:
                return col
        elif df[col].dtype in ['int64', 'float64']:
            n_unique = df[col].nunique()
            # Could be a categorical target encoded as numbers
            if 2 <= n_unique <= 20:
                return col
    
    # Default to last column if nothing found
    return df.columns[-1]

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python train.py <dataset_choice>")
        print("  1 - Traffic Accidents dataset")
        print("  2 - Combined Flights 2022 dataset")
        sys.exit(1)
    
    dataset_choice = sys.argv[1]
    config = get_dataset_config(dataset_choice)
    
    if config is None:
        print(f"Invalid dataset choice: {dataset_choice}")
        print("Valid options: 1 (Traffic Accidents) or 2 (Combined Flights)")
        sys.exit(1)
    
    dataset_path = config['path']
    dataset_name = config['name']
    target_column = config['target']
    
    # Create output directory
    output_dir = os.path.join('results', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"TRAINING ON DATASET: {dataset_name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Load and clean data
    df = load_and_clean_data(dataset_path, target_column, dataset_name)
    
    # Auto-detect target if not specified
    if target_column is None or target_column not in df.columns:
        print(f"\nTarget column not in the dataframe: {target_column}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, target_column)
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train all models
    results = []
    best_params = {}
    
    # Naïve Bayes
    model, metrics, params = train_naive_bayes(X_train, X_test, y_train, y_test, output_dir)
    results.append(metrics)
    best_params['Naïve Bayes'] = params
    
    # Logistic Regression
    model, metrics, params = train_logistic_regression(X_train, X_test, y_train, y_test, output_dir)
    results.append(metrics)
    best_params['Logistic Regression'] = params
    
    # KNN
    model, metrics, params = train_knn(X_train, X_test, y_train, y_test, output_dir)
    results.append(metrics)
    best_params['KNN'] = params
    
    # Decision Tree
    model, metrics, params = train_decision_tree(X_train, X_test, y_train, y_test, output_dir)
    results.append(metrics)
    best_params['Decision Tree'] = params
    
    # MLP
    model, metrics, params = train_mlp(X_train, X_test, y_train, y_test, output_dir)
    results.append(metrics)
    best_params['Multi-layer Perceptron'] = params
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print("\nPerformance Metrics:")
    print(results_df.to_string(index=False))
    
    print("\n\nBest Hyperparameters:")
    for model_name, params in best_params.items():
        print(f"\n{model_name}:")
        for param, value in params.items():
            print(f"  {param}: {value}")
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)
    
    # Save best parameters
    with open(os.path.join(output_dir, 'best_hyperparameters.txt'), 'w') as f:
        f.write("BEST HYPERPARAMETERS\n")
        f.write("="*80 + "\n\n")
        for model_name, params in best_params.items():
            f.write(f"{model_name}:\n")
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
    
    # Plot performance comparison
    plot_performance_comparison(results_df, output_dir)
    
    print("\n" + "="*80)
    print(f"Training completed! All results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - model_results.csv")
    print("  - best_hyperparameters.txt")
    print("  - *_hyperparameter_study.png (5 files)")
    print("  - *_performance.png (5 files)")
    print("  - model_performance_comparison.png")
    print(f"\nTotal: 12 files (2 text + 10 charts)")
    print("="*80)

if __name__ == "__main__":
    main()
