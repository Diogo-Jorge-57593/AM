import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from numpy.typing import NDArray
import pickle
import optuna

# Load the original training dataset
fnt_train = 'wdt-train.pickle'
with open(fnt_train, "rb") as f:
    test_data = pickle.load(f)
X_train = test_data['X']
y_train = test_data['y']

# Load the perturbed test dataset
fnt_test_perturbed = 'wdt-test-perturbed.pickle'
with open(fnt_test_perturbed, "rb") as f:
    test_data = pickle.load(f)
X_test_perturbed = test_data['X']
y_test_perturbed = test_data['y']

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    C = trial.suggest_float("C", 1e-5, 1e10, log=True)  # Regularization strength
    max_iter = trial.suggest_int("max_iter", 1, 500)  # Maximum number of iterations
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None])  # Regularization type
    solver = trial.suggest_categorical(
        "solver", ["lbfgs", "liblinear", "saga", "newton-cg"]
    ) 
    tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)  # Tolerance for stopping
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])  # Fit intercept
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])  # Class balancing
    warm_start = trial.suggest_categorical("warm_start", [True, False])  # Reuse solution of previous fit
    
    # Ensure valid configurations
    if solver == "saga":
        raise optuna.TrialPruned()
    if penalty == "l1" and solver not in ["liblinear", "saga"]:
        raise optuna.TrialPruned()  # Invalid configuration for l1
    if penalty == "elasticnet" and solver != "saga":
        raise optuna.TrialPruned()  # Elasticnet only works with saga
    if penalty == None and solver not in ["lbfgs", "saga"]:
        raise optuna.TrialPruned()  # None only works with lbfgs and saga
    if penalty == "l2" and solver not in ["lbfgs", "liblinear", "saga", "newton-cg"]:
        raise optuna.TrialPruned()  # Invalid configuration for l2

    # Add l1_ratio only if elasticnet is selected
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)  # Elasticnet mixing parameter
    else:
        l1_ratio = None

    # Create Logistic Regression model
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol,
        fit_intercept=fit_intercept,
        class_weight=class_weight,
        warm_start=warm_start,
        random_state=42
    )


    model.fit(X_train, y_train)
    
    # Predict using the trained model
    y_pred = model.predict(X_test_perturbed)

    # Overall accuracy
    overall_accuracy = accuracy_score(y_test_perturbed, y_pred)
    print(f"Overall Test Accuracy: {overall_accuracy:.4f}")

    # Accuracy for each fault class
    print("\nAccuracy per Fault Class:")
    for fault_class in range(7):  # Assuming 7 fault classes (0–6)
        # Extract samples belonging to the current fault class
        class_indices = np.where(y_test_perturbed == fault_class)[0]
        y_true_class = y_test_perturbed[class_indices]
        y_pred_class = y_pred[class_indices]

        # Compute accuracy for the current fault class
        class_accuracy = accuracy_score(y_true_class, y_pred_class)
        print(f"Fault Class {fault_class}: {class_accuracy:.4f}")
    
    return overall_accuracy 

# Persistent storage
storage_name = "sqlite:///lg_study_perturbed.db"

# Create or load the study
study = optuna.create_study(
    study_name="lg_optimization_perturbed",  # Name of the study
    direction="maximize",  # Maximize accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

study.optimize(objective, n_jobs=-1)