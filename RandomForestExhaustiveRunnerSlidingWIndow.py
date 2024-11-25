import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from numpy.typing import NDArray
import pickle
from sklearn.metrics  import accuracy_score
import optuna

# Convert one-hot to categorical
def onehot2cat(y: NDArray) -> NDArray:
    return np.argmax(y, axis=1)

# Load dataset
def loadDataset(fn: str, toCat: bool = False) -> NDArray:
    with open(fn, 'rb') as f:
        data = pickle.load(f)
        
    X = data['X'] 
    y = onehot2cat(data['Y']) if toCat else data['Y']
    
    return X, y

# Load the dataset
fnt = 'wtdt-part.pickle'
X, y = loadDataset(fnt, toCat=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def create_advanced_windows(data, labels, window_size, step=1):
    X_windows = []
    y_windows = []
    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i + window_size]
        features = np.hstack([
            window.flatten(),               # Flattened features
            window.mean(axis=0),            # Mean of each feature
            window.std(axis=0),             # Standard deviation
            (window[-1] - window[0]).mean() # Trend
        ])
        X_windows.append(features)
        y_windows.append(labels[i + window_size - 1])  # Use last label
    return np.array(X_windows), np.array(y_windows) 

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 10, 500)          # Number of trees
    max_depth = trial.suggest_int("max_depth", 2, 50)                 # Max depth
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None, 10, 50, 100])
    min_samples_split = trial.suggest_int("min_samples_split", 2, 100)  # Min samples split
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)  # Min samples in a leaf

    # Create RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

    window_size = trial.suggest_int("window_size", 10, 50)
    
    # Create windows with advanced features
    X_train_windows, y_train_windows = create_advanced_windows(X, y, window_size)
    X_test_windows, y_test_windows = create_advanced_windows(X_test, y_test, window_size)

    # Train model
    model.fit(X_train_windows, y_train_windows)

    # Predict and evaluate
    y_pred_windows = model.predict(X_test_windows)
    accuracy = accuracy_score(y_test_windows, y_pred_windows)
    print("Improved Accuracy with Sliding Window:", accuracy)
    
    return accuracy  # Optuna minimizes by default

# Persistent storage
storage_name = "sqlite:///forest_optimization_sliding_window.db"

# Create or load the study
study = optuna.create_study(
    study_name="forest_optimization_sliding_window",  # Name of the study
    direction="maximize",  # Minimize the negative accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

# Run the optimization
study.optimize(objective, n_jobs=2)

# Display the best results
print("Best hyperparameters:", study.best_params)
print("Best accuracy:", -study.best_value)
