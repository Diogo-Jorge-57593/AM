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

def saveSKLModel(fn: str, model) -> None:
    '''Save SKLearn model as pickle'''
    with open(fn, 'wb') as f:
        pickle.dump(model, f)

def onehot2cat(y: NDArray) -> NDArray:
    '''Convert one-hot encoded labels to categorical'''
    return np.argmax(y, axis=1)

def loadDataset(fn: str, toCat: bool = False) -> NDArray:
    '''Load dataset'''
    with open(fn, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    if toCat:
        y = onehot2cat(data['Y'])
    else:
        y = data['Y']

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
        try:
            # Check if the current trial is the best
            if study.best_trial == trial:
                print(f"New best model found! Trial {trial.number}: {trial.value:.4f}")
                # Save the model
                print("Saving model...")

                best_params = study.best_params
                print(f"Best parameters: {best_params}")

                use_scaler = best_params["use_scaler"]
                use_pca = best_params["use_pca"]
                use_feature_selection = best_params["use_feature_selection"]

                # Pipeline
                steps = []
                if use_scaler == "StandardScaler":
                    steps.append(("scaler", StandardScaler()))
                elif use_scaler == "MinMaxScaler":
                    steps.append(("normalizer", MinMaxScaler()))
                if use_pca:
                    steps.append(("pca", PCA(n_components=trial.suggest_int("n_components", 2, 24))))
                else:
                    trial.suggest_int("n_components", 0, 1)
                if use_feature_selection:
                    steps.append(("feature_selection", SelectKBest(f_classif, k=trial.suggest_int("n_features", 10, 100))))
                else:
                    trial.suggest_int("n_features", 0, 1)

                # Add classifier to the pipeline
                steps.append(("classifier", LogisticRegression(
                    C=best_params["C"],
                    penalty=best_params["penalty"],
                    solver=best_params["solver"],
                    l1_ratio=best_params.get("l1_ratio", None),
                    max_iter=best_params["max_iter"],
                    tol=best_params["tol"],
                    fit_intercept=best_params["fit_intercept"],
                    class_weight=best_params["class_weight"],
                    warm_start=best_params["warm_start"],
                    random_state=42
                )))
                pipeline = Pipeline(steps)

                # Train the pipeline on the entire training set
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test_perturbed)
                overall_accuracy = accuracy_score(y_test_perturbed, y_pred)
                print(f"Overall accuracy: {overall_accuracy:.4f}")
                saveSKLModel("T1-LogisticRegression.pickle", pipeline)
                try:
                    print("Sending model to Dropzone...")
                    with open("autoSender.py") as file:
                        exec(file.read())
                except FileNotFoundError:
                    print("The autoSender.py file was not found.")
        except ValueError:
            # Handle cases where best_trial is unavailable
            print("No best trial found yet.")


# Objective function for Optuna
def objective(trial):
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

    # Create the pipeline
    use_scaler = trial.suggest_categorical("use_scaler", ["StandardScaler", "MinMaxScaler", None])
    use_pca = trial.suggest_categorical("use_pca", [True, False])
    use_feature_selection = trial.suggest_categorical("use_feature_selection", [True, False])

    # Pipeline
    steps = []
    if use_scaler == "StandardScaler":
        steps.append(("scaler", StandardScaler()))
    elif use_scaler == "MinMaxScaler":
        steps.append(("normalizer", MinMaxScaler()))
    if use_pca:
        steps.append(("pca", PCA(n_components=trial.suggest_int("n_components", 2, 24))))
    if use_feature_selection:
        steps.append(("feature_selection", SelectKBest(f_classif, k=trial.suggest_int("n_features", 10, 500))))

    # Add classifier to the pipeline
    steps.append(("classifier", LogisticRegression(
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
    
    # Accuracy per fault class (use y_test_windows instead of y_test)
    for fault_class in range(7):  # Assuming 7 fault classes (0–6)
        class_indices = np.where(y_test_perturbed == fault_class)[0]
        y_true_class = y_test_perturbed[class_indices]
        y_pred_class = y_pred[class_indices]

        # Compute accuracy for the current fault class
        if len(y_true_class) > 0:  # Avoid division by zero
            class_accuracy = accuracy_score(y_true_class, y_pred_class)
            return class_accuracy
        else:
            print(f"Fault Class {fault_class}: No samples in this class")




# Persistent storage
storage_name = "sqlite:///logistic_study.db"

# Create or load the study
study = optuna.create_study(
    study_name="study",  # Name of the study
    direction="maximize",  # Maximize accuracy
    storage=storage_name,  # Use SQLite for storage
    load_if_exists=True  # Load the study if it already exists
)

# Run the optimization
study.optimize(objective, n_jobs=4)
