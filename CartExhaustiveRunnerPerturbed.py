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
from sklearn.tree import DecisionTreeClassifier

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
        should_run = random.random() < 0
        try:
            # Check if the current trial is the best
            if study.best_trial == trial or should_run:
                print(f"New best model found! Trial {trial.number}: {trial.value:.4f}")
                # Save the model
                print("Saving model...")

                current_params = trial.params
                print(f"Parameters: {current_params}")

                criterion = current_params.get("criterion", "gini")
                splitter = current_params.get("splitter", "best")
                max_depth = current_params.get("max_depth", None)
                min_samples_split = current_params.get("min_samples_split", 2)
                min_samples_leaf = current_params.get("min_samples_leaf", 1)
                min_weight_fraction_leaf = current_params.get("min_weight_fraction_leaf", 0.0)
                max_features = current_params.get("max_features", None)
                max_leaf_nodes = current_params.get("max_leaf_nodes", None)
                min_impurity_decrease = current_params.get("min_impurity_decrease", 0.0)
                class_weight = current_params.get("class_weight", None)
                ccp_alpha = current_params.get("ccp_alpha", 0.0)
            
                model = DecisionTreeClassifier(
                    criterion=criterion,
                    splitter=splitter,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                    max_features=max_features,
                    max_leaf_nodes=max_leaf_nodes,
                    min_impurity_decrease=min_impurity_decrease,
                    class_weight=class_weight,
                    ccp_alpha=ccp_alpha,
                    random_state=42
                )

                model.fit(X, y)

                saveSKLModel("T1-CART.pickle", model)

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
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])  # Criterion for splitting
    splitter = trial.suggest_categorical("splitter", ["best", "random"])  # Strategy for splitting at each node
    max_depth = trial.suggest_int("max_depth", 1, 500, log=True)  # Maximum depth of the tree (None = no limit)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)  
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20) 
    min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5)  
    max_features = trial.suggest_categorical("max_features", [None, "sqrt", "log2"]) 
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 500) 
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.5) 
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"]) 
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.5) 

    model = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_perturbed)
    overall_accuracy = accuracy_score(y_test_perturbed, y_pred)
    
    return overall_accuracy 

# Persistent storage
storage_name = "sqlite:///CART_study.db"

# Create or load the study
study = optuna.create_study(
    study_name="study",  # Name of the study
    direction="maximize",  # Maximize accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

# Run the optimization
study.optimize(objective, n_jobs=3, callbacks=[on_better_model])