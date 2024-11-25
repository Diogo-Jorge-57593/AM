import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import pickle
import optuna
from sklearn.utils import compute_class_weight

# Convert one-hot to categorical
def onehot2cat(y: np.ndarray) -> np.ndarray:
    return np.argmax(y, axis=1)

# Load dataset
def loadDataset(fn: str, toCat: bool = False) -> np.ndarray:
    with open(fn, 'rb') as f:
        data = pickle.load(f)
        
    X = data['X'] 
    y = onehot2cat(data['Y']) if toCat else data['Y']
    
    return X, y

# Objective function that balances accuracy, stability, and class-specific penalties
def objective_function(model, X, y, weight_accuracy=0.6, weight_stability=0.2, weight_class_penalty=0.2, target_class=None):
    """
    Objective function that balances accuracy with feature importance stability and class-specific penalties.
    """
    accuracies = []
    importances_list = []
    class_performance = []

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X, y):
        # Train and validate the model
        model.fit(X[train_idx], y[train_idx])
        accuracy = model.score(X[val_idx], y[val_idx])  # Calculate accuracy on validation set
        accuracies.append(accuracy)
        
        # Store feature importances
        importances_list.append(model.feature_importances_)
        
        # Calculate performance on each class
        if target_class is not None:
            class_accuracy = np.sum(y[val_idx] == target_class) / len(val_idx)  # Accuracy for target class
            class_performance.append(class_accuracy)

    # Calculate mean and variance of feature importances
    importances_array = np.array(importances_list)
    variance_importances = np.var(importances_array, axis=0)
    mean_variance = np.mean(variance_importances)  # Average variance across features

    # Calculate the mean accuracy
    mean_accuracy = np.mean(accuracies)

    # Penalize the performance on the specific class
    if target_class is not None:
        mean_class_accuracy = np.mean(class_performance)
        class_penalty = 1 - mean_class_accuracy  # Penalize lower performance on the specific class
    else:
        class_penalty = 0

    # Normalize metrics to balance their scales
    normalized_accuracy = mean_accuracy  # Accuracy is already on a normalized scale (0 to 1)
    normalized_stability = 1 / (1 + mean_variance)  # Lower variance means better stability

    # Combined score
    combined_score = (weight_accuracy * normalized_accuracy) + (weight_stability * normalized_stability) - (weight_class_penalty * class_penalty)

    return combined_score


fnt = 'wtdt-part.pickle'
X, y = loadDataset(fnt, toCat=True)

def optuna_objective(trial):

    n_estimators = trial.suggest_int("n_estimators", 1, 500)  # Number of trees
    max_depth = trial.suggest_int("max_depth", 2, 50)  # Max depth of trees
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.1, 0.5, 0.7, 1.0])  # Max features to consider
    min_samples_split = trial.suggest_int("min_samples_split", 2, 100)  # Minimum samples required to split an internal node
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)  # Minimum samples required at a leaf node
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])  # Whether to use bootstrap sampling for building trees
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])  # Function to measure quality of a split

    # Class weight options for imbalanced classes
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])  # Use balanced class weights if needed

    # warm_start is useful for incrementally adding trees to the model, but only relevant with n_estimators > 1
    

    # oob_score should only be True if bootstrap=True
    oob_score = trial.suggest_categorical("oob_score", [True, False]) if bootstrap else False

    # If class_weight is "balanced" and warm_start is used, calculate class weights manually
    if class_weight == "balanced":
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight = dict(zip(np.unique(y), class_weights))


    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        criterion=criterion,
        class_weight=class_weight,
        oob_score=oob_score 
    )

    # Evaluate the model with the custom objective function
    combined_score = objective_function(model, X, y)
    
    return combined_score

# Persistent storage
storage_name = "sqlite:///forest_study_weighted.db"

# Create or load the study
study = optuna.create_study(
    study_name="forest_optimization_weighted",  # Name of the study
    direction="maximize",  # Minimize the combined score
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

# Run the optimization
study.optimize(optuna_objective, n_jobs=-1)

# Display the best results
print("Best hyperparameters:", study.best_params)
print("Best combined score:", study.best_value)
