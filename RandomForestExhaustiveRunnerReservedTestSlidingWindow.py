import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from numpy.typing import NDArray
import pickle
import optuna

def create_sliding_windows(data, labels, window_size, step=1):
    X_windows = []
    y_windows = []
    for i in range(0, len(data) - window_size + 1, step):
        X_windows.append(data[i:i + window_size].flatten())  # Flatten for RF compatibility
        y_windows.append(labels[i + window_size - 1])  # Label is the last element in the window
    return np.array(X_windows), np.array(y_windows)

# Convert one-hot to categorical
def onehot2cat(y: NDArray) -> NDArray:
    return np.argmax(y, axis=1)

def loadDataset(fn:str, toCat:bool=False) -> NDArray:
    '''load dataset'''
    with open(fn, 'rb') as f:
        data = pickle.load(f)
        
    X = data['X'] 
    if toCat: y = onehot2cat(data['Y'])
    else:     y = data['Y'] 
    
    return X, y

fnt = 'wtdt-part.pickle'
X, y = loadDataset(fnt, toCat=True)

class_0_indices = np.where(y == 0)[0]  # Get all indices of class 0
class_0_samples = X[class_0_indices]  # Extract all samples of class 0
perturbed_samples = class_0_samples + np.random.normal(0, 0.1, class_0_samples.shape)  # Add noise
X[class_0_indices] = perturbed_samples  # Replace original samples with perturbed versions


print(X.shape)

# Objective function for Optuna
def objective(trial):
    window_size = trial.suggest_int("window_size", 2, 50)
    step = trial.suggest_int("step", 1, 10)
    n_estimators = trial.suggest_int("n_estimators", 1, 500)  # Number of trees
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])  # Criterion for splitting
    max_depth = trial.suggest_int("max_depth", 1, 500, log=True)  # Max depth
    min_samples_split = trial.suggest_int("min_samples_split", 2, 100)  # Min samples to split
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)  # Min samples in a leaf
    min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5)  # Min weight fraction of a leaf
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])  # Max features
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 1000, log=True)  # Max leaf nodes
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.5)  # Min impurity decrease
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])  # Bootstrap samples
    oob_score = trial.suggest_categorical("oob_score", [False, True])  # Out-of-bag score
    weight_0 = trial.suggest_int("weight_0", 1, 500)  # Class weights
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.1)  # Complexity parameter for pruning
    max_samples = trial.suggest_float("max_samples", 0.1, 1.0) if bootstrap else None  # Max samples for bootstrap

    if oob_score != None and bootstrap == False:
        raise optuna.exceptions.TrialPruned() 
    
    # Create RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        oob_score=oob_score,
        class_weight = {0: weight_0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
        ccp_alpha=ccp_alpha,
        max_samples=max_samples,
        random_state=42
    )

    # Generate sliding windows for training and testing
    X_windows, y_windows = create_sliding_windows(X, y, window_size, step=step)

    f1_scorer = make_scorer(f1_score, average='weighted')
    scores = cross_val_score(model, X_windows, y_windows, cv=5, scoring=f1_scorer)

    return scores.mean() 

# Persistent storage
storage_name = "sqlite:///forest_study_perturbed_window_weighted.db"

# Create or load the study
study = optuna.create_study(
    study_name="forest_optimization_perturbed_window",  # Name of the study
    direction="maximize",  # Maximize accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

# Run the optimization
study.optimize(objective, n_jobs=-1)