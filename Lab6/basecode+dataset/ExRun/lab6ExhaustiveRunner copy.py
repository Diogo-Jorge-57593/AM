import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from numpy.typing import NDArray
import pickle

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

# Define the study name and database file (SQLite)
study_name = 'hyperparameter_optimization_study2'
db_url = f'sqlite:///{study_name}.db'

def objective(trial):
    # Model 1: Logistic Regression (Regularized)
    model_name = trial.suggest_categorical('model', ['logistic_regression', 'naive_bayes', 'cart', 'knn', 'svm', 'random_forest'])
    
    if model_name == 'logistic_regression':
        C = trial.suggest_float('logistic_C', 1e-5, 1e5, log=True)
        solver = trial.suggest_categorical('logistic_solver', ['liblinear'])
        model = LogisticRegression(C=C, solver=solver, max_iter=10000)
        features = f"C={C}, solver={solver}"
    
    # Model 2: Naive Bayes
    elif model_name == 'naive_bayes':
        var_smoothing = trial.suggest_float('nb_var_smoothing', 1e-12, 1e-2, log=True)
        model = GaussianNB(var_smoothing=var_smoothing)
        features = f"var_smoothing={var_smoothing}"

    # Model 3: CART (Decision Tree)
    elif model_name == 'cart':
        max_depth = trial.suggest_int('cart_max_depth', 1, 20)
        min_samples_split = trial.suggest_int('cart_min_samples_split', 2, 10)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        features = f"max_depth={max_depth}, min_samples_split={min_samples_split}"

    # Model 4: K-Nearest Neighbors
    elif model_name == 'knn':
        n_neighbors = trial.suggest_int('knn_n_neighbors', 1, 30)
        weights = trial.suggest_categorical('knn_weights', ['uniform', 'distance'])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        features = f"n_neighbors={n_neighbors}, weights={weights}"

    # Model 5: Support Vector Machine (SVM)
    elif model_name == 'svm':
        C = trial.suggest_float('svm_C', 1e-5, 1e5, log=True)
        kernel = trial.suggest_categorical('svm_kernel', ['linear', 'rbf'])
        gamma = trial.suggest_categorical('svm_gamma', ['scale', 'auto'])
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        features = f"C={C}, kernel={kernel}, gamma={gamma}"

    # Model 6: Random Forest
    elif model_name == 'random_forest':
        n_estimators = trial.suggest_int('rf_n_estimators', 10, 200)
        max_depth = trial.suggest_int('rf_max_depth', 3, 30)
        min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        features = f"n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}"

    # Use cross-validation to evaluate the model
    accuracy = cross_val_score(model, X, y, cv=5, scoring=make_scorer(accuracy_score)).mean()
    
    # Print model, features, and accuracy for this trial
    print(f"Model: {model_name}")
    print(f"Features: {features}")
    print(f"Accuracy: {accuracy:.4f}")
    print('-' * 50)
    
    return accuracy

# Create an Optuna study to optimize the objective, specifying the storage location
study = optuna.create_study(direction='maximize', storage=db_url, study_name=study_name, load_if_exists=True)

# Optimize the hyperparameters with Optuna for 100 trials (or continue from previous)
study.optimize(objective)

# Print the best hyperparameters found
print("Best hyperparameters found: ", study.best_params)
