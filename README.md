# RL-HVAC

# Environment Creation
Virtual environment is used to setup the environment

First install virtual environment using the command
```
pip3 install virtualenv
```
Create a virtual environment using the command
```
python3 -m venv sinergym-env
```
Activate the virtual environment using the command
```
source sinergym-env/bin/activate
```
Install the required packages using the command
```
pip3 install -r requirements.txt
```

# Custom environment usage

To be able to use custom environments, you need to change some files

Inside the sinergym-env dictionary, go to sinergym-env/lib/sinergym/data/buildings and add your epJSON file

Then add your default json file to sinergym-env/lib/sinergym/data/default_configuration

Also you need to add your custom discretization process into sinergym-env/lib/sinergym/utils/constants.py


# SSH Screen

screeen -S my_session
ctrl+a d detach
screen -r my_session

screen -list 
screein -r id

# Standalone Training Plots (Local)

Both standalone trainers now save:
- `training_summary.json`
- `metrics.json`
- `training_core_metrics.png`

## Generate plots during training

These are generated automatically at the end of:
- `python train_standalone.py ...`
- `python train_standalone_madqn_pmv.py ...`

## Rebuild or compare plots offline

Single run:
```
python plot_standalone_training.py --run-dir ./standalone_results/<run_folder>
```

Compare multiple runs:
```
python plot_standalone_training.py \
  --run-dir ./standalone_results/<run1> \
  --run-dir ./standalone_results/<run2> \
  --label dqn \
  --label madqn_pmv
```

# Optuna Tuning

Use Optuna for short-budget hyperparameter search, then retrain best configs longer.

Setup:
```
cd /storage/Master/ParallelRL/RL-HVAC
source ~/.bashrc
conda activate hvac-rl
```

Install dependency (if needed):
```
pip install optuna
```

Tune DQN (single-agent):
```
python optuna_tune_dqn.py \
  --n-trials 20 \
  --episodes 3 \
  --season hot \
  --env-id A403mediumfanger \
  --study-dir ./optuna_runs
```

Tune MADQN-PMV (multi-agent):
```
python optuna_tune_madqn_pmv.py \
  --n-trials 20 \
  --episodes 3 \
  --season hot \
  --env-id A403mediumfanger \
  --study-dir ./optuna_runs
```

Outputs:
- `best_params_dqn.json` / `best_params_madqn_pmv.json`
- `optuna_study_*.db` (sqlite)
- `trials_*.csv`

After tuning:
1. Take top 3-5 parameter sets.
2. Retrain each with longer episodes and multiple seeds.
3. Select by mean and std, not only best single run.