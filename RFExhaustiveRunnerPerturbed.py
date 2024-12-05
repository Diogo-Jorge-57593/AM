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
        should_run = random.random() < 0.1 # 10% chance of running
        try:
            # Check if the current trial is the best
            if study.best_trial == trial or should_run:
                print(f"New best model found! Trial {trial.number}: {trial.value:.4f}")
                # Save the model
                print("Saving model...")

                current_params = trial.params
                print(f"Parameters: {current_params}")


                # Get window parameters
                window_size = current_params.get("window_size", 10)
                step = current_params.get("step", 1)

                # Create sliding windows
                X_windows, y_windows = create_sliding_windows(X, y, window_size, step)

                # get hyperparameters
                n_estimators = current_params.get("n_estimators", 100)
                max_depth = current_params.get("max_depth", 10)
                max_features = current_params.get("max_features", "sqrt")
                min_samples_split = current_params.get("min_samples_split", 2)
                min_samples_leaf = current_params.get("min_samples_leaf", 1)

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )

                # Train the pipeline on the entire training set
                model.fit(X_windows, y_windows)

                saveSKLModel("T1-randomForest.pickle", model)
                try:
                    print("Sending model to Dropzone...")
                    with open("autoSender.py") as file:
                        exec(file.read())

                    log_entry = {
                        "model_name": "RandomForest",
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

    # suggest sliding window parameters
    window_size = trial.suggest_int("window_size", 10, 20)
    step = trial.suggest_int("step", 1, 10)

    # Create sliding windows
    X_windows, y_windows = create_sliding_windows(X_train, y_train, window_size, step)
    X_windows_perturbed, y_windows_perturbed = create_sliding_windows(X_test_perturbed, y_test_perturbed, window_size, step)  

    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 1, 500)  # Number of trees
    max_depth = trial.suggest_int("max_depth", 1, 500, log=True)  # Max depth
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])  # Max features
    min_samples_split = trial.suggest_int("min_samples_split", 2, 100)  # Min samples to split
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)  # Min samples in a leaf

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42 # Ensure reproducibility
    )

    model.fit(X_windows, y_windows)

    # Predict and evaluate
    y_pred = model.predict(X_windows_perturbed)
    overall_accuracy = accuracy_score(y_windows_perturbed, y_pred)
    
    return overall_accuracy 

# Persistent storage
storage_name = "sqlite:///RF_study.db"

# Create or load the study
study = optuna.create_study(
    study_name="study",  # Name of the study
    direction="maximize",  # Maximize accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

# Run the optimization
study.optimize(objective, n_jobs=2, callbacks=[on_better_model],n_trials=1)