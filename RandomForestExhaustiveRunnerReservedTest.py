import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from numpy.typing import NDArray
import pickle
import optuna

# Convert one-hot to categorical
def onehot2cat(y: NDArray) -> NDArray:
    return np.argmax(y, axis=1)

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
    class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])  # Class weights
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
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        max_samples=max_samples,
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
storage_name = "sqlite:///forest_study_perturbed_new.db"

# Create or load the study
study = optuna.create_study(
    study_name="forest_optimization_perturbed_new",  # Name of the study
    direction="maximize",  # Maximize accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

# Run the optimization
study.optimize(objective, n_jobs=-1)