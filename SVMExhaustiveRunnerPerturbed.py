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

def on_better_model(study: optuna.Study, trial):
    print("Checking for better model...")
    if len(study.trials) > 0:
        should_run = random.random() < 0.001
        try:
            # Check if the current trial is the best
            if study.best_trial == trial or should_run:
                print(f"New best model found! Trial {trial.number}: {trial.value:.4f}")
                # Save the model
                print("Saving model...")

                current_params = trial.params
                print(f"Parameters: {current_params}")
            
                # Get hyperparameters
                C = current_params.get("C", 1.0)  # Default to 1.0 if `None`
                kernel = current_params.get("kernel", "rbf")  # Default to 'rbf'
                degree = current_params.get("degree", 3)  # Default to 3
                gamma = current_params.get("gamma", "scale")  # Default to 'scale'
                coef0 = current_params.get("coef0", 0.0)  # Default to 0.0
                max_iter = current_params.get("max_iter", -1)  # Default to -1 (no limit in scikit-learn)

                scaler = current_params.get("scaler", "StandardScaler")  # Default to 'StandardScaler'

                steps = []
                if scaler == "StandardScaler":
                    steps.append(("scaler", StandardScaler()))
                if scaler == "MinMaxScaler":
                    steps.append(("scaler", MinMaxScaler()))    

                # Create the SVM model
                steps.append(("svc", SVC(
                    C=C,
                    kernel=kernel,
                    degree=degree,
                    gamma=gamma,
                    coef0=coef0,
                    max_iter=max_iter,
                    random_state=42
                )))

                pipeline = Pipeline(steps)

                # Train the pipeline on the entire training set
                pipeline.fit(X, y)

                saveSKLModel("T1-SVM.pickle", pipeline)
                try:
                    print("Sending model to Dropzone...")
                    with open("autoSender.py") as file:
                        exec(file.read())

                    log_entry = {
                        "model_name": "SVM",
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
    
    # Suggest hyperparameters
    scaler = trial.suggest_categorical("scaler", ["StandardScaler", "MinMaxScaler"])  # Scaler type
    C = trial.suggest_float("C", 1e-5, 1e2, log=True)  # Regularization strength
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])  # Kernel type
    degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3  # Only for poly kernel
    #gamma = trial.suggest_categorical("gamma", ["scale", "auto"]) if kernel in ["rbf", "poly", "sigmoid"] else None
    coef0 = trial.suggest_float("coef0", 0.0, 1.0) if kernel in ["poly", "sigmoid"] else 0.0  # Only for poly and sigmoid
    max_iter = trial.suggest_int("max_iter", 1, 500)  # Maximum number of iterations
    if kernel in ["rbf", "poly", "sigmoid"]:
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    else:
        gamma = "scale"  # Default value for linear kernel

    steps = []
    if scaler == "StandardScaler":
        steps.append(("scaler", StandardScaler()))
    if scaler == "MinMaxScaler":
        steps.append(("scaler", MinMaxScaler()))    

    # Create the SVM model
    steps.append(("svc", SVC(
    C=C,
    kernel=kernel,
    degree=degree,
    gamma=gamma,
    coef0=coef0,
    max_iter=max_iter,
    random_state=42
)))

    pipeline = Pipeline(steps)

    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test_perturbed)
    overall_accuracy = accuracy_score(y_test_perturbed, y_pred)
    
    return overall_accuracy 

# Persistent storage
storage_name = "sqlite:///SVM_study.db"

# Create or load the study
study = optuna.create_study(
    study_name="study",  # Name of the study
    direction="maximize",  # Maximize accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

# Run the optimization
study.optimize(objective, n_jobs=1, callbacks=[on_better_model])