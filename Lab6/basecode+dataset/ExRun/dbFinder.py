from collections import defaultdict
import optuna

study_name = 'hyperparameter_optimization_study2'
db_url = f'sqlite:///{study_name}.db'
study = optuna.create_study(direction='maximize', storage=db_url, study_name=study_name, load_if_exists=True)

trials = study.trials  # Returns a list of all trials

best_trial = study.best_trial
print("Best trial number:", best_trial.number)
print("Best value:", best_trial.value)
print("Best params:", best_trial.params)

best_trials_per_model = defaultdict(lambda: None)

# Iterate over all completed trials
for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        model_name = trial.params.get('model')  # Get the model type
        if model_name:
            current_best = best_trials_per_model[model_name]
            # Update if it's the first trial for this model or if it has a better value
            if current_best is None or (trial.value is not None and trial.value > current_best.value):
                best_trials_per_model[model_name] = trial

# Print the best trial for each model
for model_name, best_trial in best_trials_per_model.items():
    if best_trial:
        print(f"Model: {model_name}")
        print(f"Best Trial Number: {best_trial.number}")
        print(f"Accuracy: {best_trial.value:.4f}")
        print(f"Parameters: {best_trial.params}")
        print("-" * 50)