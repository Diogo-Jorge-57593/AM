from collections import defaultdict
import optuna

# Setup study
study_name = 'forest_optimization'
db_url = f'sqlite:///forest_study.db'
study = optuna.create_study(direction='maximize', storage=db_url, study_name=study_name, load_if_exists=True)

# Fetch all completed trials
completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

# Sort trials by value (higher is better in 'maximize')
sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)

# Get the top 10 trials (highest values)
top_10_trials = sorted_trials[:10]

# Print the details of the top 10 trials
print("Top 10 Best Trials (Maximize):")
for rank, trial in enumerate(top_10_trials, start=1):
    print(f"Rank {rank}")
    print(f"Trial Number: {trial.number}")
    print(f"Value: {trial.value:.4f}")
    print(f"Parameters: {trial.params}")
    print("-" * 50)

# Optionally, group the top 10 by model name
best_trials_per_model = defaultdict(list)

for trial in top_10_trials:
    model_name = trial.params.get('model')  # Get the model type
    if model_name:
        best_trials_per_model[model_name].append(trial)

# Print the best trials grouped by model
print("Top 10 Trials Grouped by Model:")
for model_name, trials in best_trials_per_model.items():
    print(f"Model: {model_name}")
    for trial in trials:
        print(f"  Trial Number: {trial.number}, Value: {trial.value:.4f}, Parameters: {trial.params}")
    print("-" * 50)
