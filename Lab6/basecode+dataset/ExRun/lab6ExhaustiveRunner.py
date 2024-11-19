import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from numpy.typing import NDArray
import pickle

# Save the optimization result to a file
def save_state(res, filename="optimization_state_forest.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(res, f)
    print(f"Optimization state saved to {filename}")

# Load the optimization result from a file
def load_state(filename="optimization_state_forest.pkl"):
    try:
        with open(filename, "rb") as f:
            res = pickle.load(f)
        print(f"Optimization state loaded from {filename}")
        return res
    except FileNotFoundError:
        print(f"No saved state found. Starting fresh.")
        return None

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

# Hyperparameter space
space = [
    Integer(10, 200, name='n_estimators'),  # Number of trees
    Integer(2, 50, name='max_depth'),       # Max depth
    Real(0.1, 1.0, name='max_features'),    # Proportion of features to consider
    Real(1e-6, 1.0, name='min_samples_split')  # Min samples split
]

# Logger for optimization results
def test_logger(res):
    if not hasattr(test_logger, "best_score"):
        test_logger.best_score = float("inf")
        test_logger.best_params = None

    if res.fun < test_logger.best_score:
        test_logger.best_score = res.fun
        test_logger.best_params = res.x

    print(f"Iteration {len(res.x_iters)}:")
    print(f" - Current Parameters: {res.x_iters[-1]}")
    print(f" - Current Score: {-res.func_vals[-1]:.4f}")
    print(f" - Best Score So Far: {-test_logger.best_score:.4f}")
    print(f" - Best Parameters So Far: {test_logger.best_params}")
    print("-" * 40)

@use_named_args(space)
def objective(n_estimators, max_depth, max_features, min_samples_split):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        n_jobs=-1
    )
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    mean_accuracy = np.mean(scores)
    return -mean_accuracy

# Attempt to load previous state
result = load_state()

if result is None:
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=30,
        random_state=42,
        callback=[test_logger]
    )
    save_state(result)

# Display best results
print("Best hyperparameters:", result.x)
print("Best accuracy:", -result.fun)
