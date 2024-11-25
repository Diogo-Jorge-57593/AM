import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from numpy.typing import NDArray
import pickle
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

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 10, 200)          # Number of trees
    max_depth = trial.suggest_int("max_depth", 2, 50)                 # Max depth
    max_features = trial.suggest_float("max_features", 0.1, 1.0)      # Proportion of features
    min_samples_split = trial.suggest_float("min_samples_split", 1e-6, 1.0)  # Min samples split

    # Create RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        n_jobs=-1
    )
    
    # Perform cross-validation and calculate the mean accuracy
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    mean_accuracy = np.mean(scores)
    
    return -mean_accuracy  # Optuna minimizes by default

# Persistent storage
storage_name = "sqlite:///random_forest_study.db"

# Create or load the study
study = optuna.create_study(
    study_name="random_forest_optimization",  # Name of the study
    direction="minimize",  # Minimize the negative accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

# Run the optimization
study.optimize(objective, n_trials=100, n_jobs=-1)

# Display the best results
print("Best hyperparameters:", study.best_params)
print("Best accuracy:", -study.best_value)
