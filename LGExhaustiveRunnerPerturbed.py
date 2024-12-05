import datetime
import json
import os
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from numpy.typing import NDArray
import pickle
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

def saveSKLModel(fn:str, model) -> None:
    '''save SKLearn model as pickle'''
    with open(fn, 'wb') as f:
        pickle.dump(model, f)

def stop_after_3am():
    current_time = datetime.datetime.now()
    three_am_today = current_time.replace(hour=3, minute=37, second=0, microsecond=0)
    
    # Check if the current time is after 3 AM
    if current_time > three_am_today:
        raise optuna.exceptions.OptunaError("Optimization stopped because it is after 3 AM.")
    else:
        print("Current time is before 3 AM. Continuing optimization...")

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

def create_sliding_windows(data, labels, window_size, step=1):
    X_windows = []
    y_windows = []
    for i in range(0, len(data) - window_size + 1, step):
        X_windows.append(data[i:i + window_size].flatten())
        y_windows.append(labels[i + window_size - 1]) 
    return np.array(X_windows), np.array(y_windows)

def on_better_model(study: optuna.Study, trial):
    print("Checking for better model...")
    if len(study.trials) > 0:
        should_run = random.random() < 0 # 10% chance of running
        try:
            # Check if the current trial is the best
            if study.best_trial == trial or should_run:
                print(f"New best model found! Trial {trial.number}: {trial.value:.4f}")
                # Save the model
                print("Saving model...")

                current_params = trial.params
                print(f"Parameters: {current_params}")

                scaler = current_params.get("scaler", "StandardScaler")  # Default to 'StandardScaler'

                steps = []  
                if scaler == "StandardScaler":
                    steps.append(("scaler", StandardScaler()))
                if scaler == "MinMaxScaler":
                    steps.append(("scaler", MinMaxScaler()))    

                # Get hyperparameters
                C = current_params.get("C", 1.0)  # Default to 1.0 if `None`
                l1_ratio = current_params.get("l1_ratio", None)  # Default to None
                max_iter = current_params.get("max_iter", 100)  # Default to 100
                penalty = current_params.get("penalty", "l2")  # Default to 'l2'
                solver = current_params.get("solver", "lbfgs")  # Default to 'lbfgs'
                tol = current_params.get("tol", 1e-4)  # Default to 1e-4
                fit_intercept = current_params.get("fit_intercept", True)  # Default to True
                class_weight = current_params.get("class_weight", None) 
                warm_start = current_params.get("warm_start", False)  # Default to False

                # Create the SVM model
                steps.append(("LG", LogisticRegression(
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
                )))

                pipeline = Pipeline(steps)

                pipeline.fit(X, y)

                saveSKLModel("T1-LogisticRegressionReal.pickle", pipeline)
                try:
                    print("Sending model to Dropzone...")
                    with open("autoSender.py") as file:
                        exec(file.read())

                    log_entry = {
                        "model_name": "Logistic Regression",
                        "best_params": current_params,  # Ensure this variable is correctly populated
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    log_file = "upload_results.json"

                    # Debug: Check if log_file exists
                    print(f"Checking if {log_file} exists...")

                    # Check if the file exists
                    if os.path.exists(log_file):
                        try:
                            # Read existing logs
                            print(f"{log_file} exists. Reading the file...")
                            with open(log_file, "r") as f:
                                logs = json.load(f)
                                print("Existing logs loaded successfully.")
                        except json.JSONDecodeError:
                            # Handle corrupted or malformed JSON file
                            print(f"Error reading {log_file}. File may be corrupted. Initializing fresh logs.")
                            logs = []
                    else:
                        # Initialize an empty log list if the file does not exist
                        print(f"{log_file} does not exist. Initializing fresh logs.")
                        logs = []

                    # Append the new log entry
                    print(f"Appending new log entry: {log_entry}")
                    logs.append(log_entry)

                    # Write the updated logs back to the file
                    try:
                        with open(log_file, "w") as f:
                            json.dump(logs, f, indent=4)
                        print(f"Logs successfully written to {log_file}.")
                    except Exception as e:
                        print(f"Error writing to {log_file}: {e}")

                except FileNotFoundError:
                    print("The autoSender.py file was not found.")
        except ValueError:
            # Handle cases where best_trial is unavailable
            print("No best trial found yet.")


# Objective function for Optuna
def objective(trial):

    # Stop optimization after 3 AM
    # stop_after_3am()

    # Suggest hyperparameters
    scaler = trial.suggest_categorical("scaler", ["StandardScaler", "MinMaxScaler"])  # Scaler type

    # Hyperparameters
    C = trial.suggest_float("C", 1e-5, 1e10, log=True)  # Regularization strength
    max_iter = trial.suggest_int("max_iter", 1, 500)  # Maximum number of iterations
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None])  # Regularization type
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "newton-cg"])
    tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)  # Tolerance for stopping
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])  # Fit intercept
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])  # Class balancing
    warm_start = trial.suggest_categorical("warm_start", [True, False])  # Reuse solution of previous fit

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

    steps = []
    if scaler == "StandardScaler":
        steps.append(("scaler", StandardScaler()))
    if scaler == "MinMaxScaler":
        steps.append(("scaler", MinMaxScaler()))    

    # Create the SVM model
    steps.append(("LG", LogisticRegression(
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
    )))

    pipeline = Pipeline(steps)

    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test_perturbed)
    overall_accuracy = accuracy_score(y_test_perturbed, y_pred)
    
    return overall_accuracy 

# Persistent storage
storage_name = "sqlite:///LG_study.db"

# Create or load the study
study = optuna.create_study(
    study_name="study",  # Name of the study
    direction="maximize",  # Maximize accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

# Run the optimization
study.optimize(objective, n_jobs=1, callbacks=[on_better_model])