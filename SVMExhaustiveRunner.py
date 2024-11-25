import numpy as np
from sklearn.metrics import accuracy_score
import pickle
import optuna
from sklearn.svm import SVC

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
    C = trial.suggest_float("C", 1e-5, 1e2, log=True)  # Regularization strength
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])  # Kernel type
    degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3  # Only for poly kernel
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"]) if kernel in ["rbf", "poly", "sigmoid"] else None
    coef0 = trial.suggest_float("coef0", 0.0, 1.0) if kernel in ["poly", "sigmoid"] else 0.0  # Only for poly and sigmoid

    # Create the SVM model
    model = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        max_iter=5000,  # Increase max iterations for convergence
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
storage_name = "sqlite:///SVM_study_perturbed.db"

# Create or load the study
study = optuna.create_study(
    study_name="SVM_optimization_perturbed",  # Name of the study
    direction="maximize",  # Maximize accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

study.optimize(objective, n_jobs=-1)