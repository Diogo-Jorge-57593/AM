import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    C = trial.suggest_loguniform("C", 1e-3, 1e3)  # Regularization parameter
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    gamma = trial.suggest_loguniform("gamma", 1e-4, 1e1) if kernel in ["rbf", "poly", "sigmoid"] else "scale"
    degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3

    # Create SVC model with a pipeline for scaling
    model = Pipeline([
        ("scaler", StandardScaler()),  # Normalize features for better SVM performance
        ("svm", SVC(C=C, kernel=kernel, gamma=gamma, degree=degree))
    ])
    
    # Perform cross-validation and calculate the mean accuracy
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    mean_accuracy = np.mean(scores)
    
    return mean_accuracy  # Optuna minimizes by default

# Persistent storage
storage_name = "sqlite:///svm_study.db"

# Create or load the study
study = optuna.create_study(
    study_name="svm_optimization",  # Name of the study
    direction="maximize",  # Minimize the negative accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

# Run the optimization
study.optimize(objective, n_jobs=-1)

# Display the best results
print("Best hyperparameters:", study.best_params)
print("Best accuracy:", -study.best_value)
