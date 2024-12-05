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
from sklearn.neural_network import MLPClassifier
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
    if len(study.trials) > 15:
        should_run = random.random() < 0
        try:
            # Check if the current trial is the best
            if study.best_trial == trial or should_run:
                print(f"New best model found! Trial {trial.number}: {trial.value:.4f}")
                # Save the model
                print("Saving model...")

                current_params = trial.params
                print(f"Parameters: {current_params}")

                scaler = current_params.get('scaler', 'StandardScaler')

                steps = []
                if scaler == "StandardScaler":
                    steps.append(("scaler", StandardScaler()))
                if scaler == "MinMaxScaler":
                    steps.append(("scaler", MinMaxScaler()))

                feature_selector = current_params.get('feature_selector', 'None')
                if feature_selector == "SelectKBest":
                    k = current_params.get('k', 10)
                    steps.append(("select", SelectKBest(f_classif, k=k)))
                elif feature_selector == "PCA":
                    n_components = current_params.get('n_components', 10)
                    steps.append(("pca", PCA(n_components=n_components)))

                hidden_layer_sizes = current_params.get('hidden_layer_sizes', [(100,)])
                activation = current_params.get('activation', 'relu')
                solver = current_params.get('solver', 'adam')
                learning_rate = current_params.get('learning_rate', 'constant')
                alpha = current_params.get('alpha', 1e-4)
                max_iter = current_params.get('max_iter', 200)
                if solver == "sgd":
                    momentum = current_params.get("momentum", 0.9)
                    nesterovs_momentum = current_params.get("nesterovs_momentum", False)
                    learning_rate_init = current_params.get("learning_rate_init", 0.001)

                # Initialize the classifier with current hyperparameters
                mlp_params = {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': activation,
                    'solver': solver,
                    'learning_rate': learning_rate,
                    'alpha': alpha,
                    'max_iter': max_iter,
                    'random_state': 42
                }

                # Add SGD-specific parameters if 'sgd' solver is selected
                if solver == "sgd":
                    mlp_params.update({
                        'momentum': momentum,
                        'nesterovs_momentum': nesterovs_momentum,
                        'learning_rate_init': learning_rate_init
                    })

                steps.append(("MLP", MLPClassifier(**mlp_params)))

                pipeline = Pipeline(steps)

                pipeline.fit(X, y)

                saveSKLModel("T1-Neural.pickle", pipeline)

                try:
                    print("Sending model to Dropzone...")
                    with open("autoSender.py") as file:
                        exec(file.read())

                    log_entry = {
                        "model_name": "Neural",
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

    stop_after_3am()

    scaler = trial.suggest_categorical("scaler", ["StandardScaler", "MinMaxScaler"])  # Scaler type

    steps = []
    if scaler == "StandardScaler":
        steps.append(("scaler", StandardScaler()))
    if scaler == "MinMaxScaler":
        steps.append(("scaler", MinMaxScaler()))   

    feature_selector = trial.suggest_categorical("feature_selector", ["None", "SelectKBest", "PCA"])
    if feature_selector == "SelectKBest":
        k = trial.suggest_int("k", 10, X_train.shape[1])  # Tune the number of features to select
        steps.append(("select", SelectKBest(f_classif, k=k)))
    elif feature_selector == "PCA":
        n_components = trial.suggest_int("n_components", 10, X_train.shape[1])
        steps.append(("pca", PCA(n_components=n_components)))

    # Define the hyperparameter search space
    hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(100,), (50, 50), (100, 50, 50)])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
    learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
    alpha = trial.suggest_float('alpha', 1e-4, 1e-2)
    max_iter = trial.suggest_int('max_iter', 10, 500)
    if solver == "sgd":
        momentum = trial.suggest_float("momentum", 0.0, 0.9)
        nesterovs_momentum = trial.suggest_categorical("nesterovs_momentum", [True, False])
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2)

    # Initialize the classifier with current hyperparameters
    mlp_params = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
        'learning_rate': learning_rate,
        'alpha': alpha,
        'max_iter': max_iter,
        'random_state': 42
    }

    # Add SGD-specific parameters if 'sgd' solver is selected
    if solver == "sgd":
        mlp_params.update({
            'momentum': momentum,
            'nesterovs_momentum': nesterovs_momentum,
            'learning_rate_init': learning_rate_init
        })

    steps.append(("MLP", MLPClassifier(**mlp_params)))

    pipeline = Pipeline(steps)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test_perturbed)
    overall_accuracy = accuracy_score(y_test_perturbed, y_pred)
    
    return overall_accuracy 

# Persistent storage
storage_name = "sqlite:///Neural_study.db"

# Create or load the study
study = optuna.create_study(
    study_name="study",  # Name of the study
    direction="maximize",  # Maximize accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

# Run the optimization
study.optimize(objective, n_jobs=3, callbacks=[on_better_model])