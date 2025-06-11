#!/usr/bin/env python3
import argparse
import os
import json
import socket
from datetime import datetime
import wandb
# import warnings
# warnings.filterwarnings("ignore", message=".*Casting input x to numpy array.*")

from experiments.madqn import madqn_train
from experiments.dqn import dqn_train 
from utils.experiment_utils import create_experiment_name
def parse_args():
    p = argparse.ArgumentParser(description="Main entrypoint for RL-HVAC sweep/agent")
    p.add_argument("--workdir",required=True,help="Base folder logs")
    p.add_argument("--sweep-id",default=None,help="Join this existing sweep")
    p.add_argument("--project",default=os.environ.get("WANDB_PROJECT", "A403-Train"))
    p.add_argument("--entity",default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--count",type=int,default=None)
    p.add_argument("--algorithm", required=True, choices=["madqn", "dqn"], help="RL algorithm to use")
    return p.parse_args()


def main():
    args = parse_args()

    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.workdir, f"{hostname}_{timestamp}_{os.urandom(2).hex()}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Writing outputs to: {run_dir}")
    
    # Create experiment save dir
    train_season = "hot"
    ENV_ID ="A403medium"
    NUM_EPISODES = 10             
    unique_experiment_name = f"{train_season}_{ENV_ID}_train_{timestamp}"
   
    experiment_save_dir_name = os.path.join(run_dir, "results", args.algorithm, unique_experiment_name)

    if not os.path.exists(experiment_save_dir_name):
        os.makedirs(experiment_save_dir_name)

    ENV_NAME = f"{ENV_ID}_{train_season}_MULTISPEED_FAN"
    ALGORITHM_NAME = args.algorithm.upper()
    
    name = create_experiment_name(env_name=ENV_NAME, episodes=NUM_EPISODES,algorithm_name=ALGORITHM_NAME)
    if args.algorithm == "madqn":
        layer_sizes = [[64, 64],[128,64],[128,128]]
    elif args.algorithm == "dqn":
        layer_sizes = [[128, 64]]
    else:
        raise ValueError("Unsupported algorithm")
    if args.sweep_id is None:
        # Build your sweep config dict however you like:
        sweep_config = {
            "method": "random",
            "project": "A403-Train",
            "name": name,
            "metric": {"name": "final_val_power_kWh_mean", "goal": "minimize"},
            "parameters": {
                    'learning_rate': {'min': 1e-3,'max': 3e-3},
                    'lambda_energy': {'values': [1/1_600_000]},
                    'gamma': {'min': 0.8,'max': 0.99},
                    'co2_weight': {'min': 0.20,'max': 0.40},
                    'temp_weight': {'min': 0.10,'max': 0.50},
                    'experiment_save_dir': {'value': experiment_save_dir_name},
                    'train_season': {'value': train_season},
                    'agent_count': {'value': 40},
                    'num_episodes': {'value': NUM_EPISODES},
                    'layer_sizes': {'values': layer_sizes},
                    'env_id': {'value': ENV_ID},
                }
        }
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        agent_count = args.count
    else:
        print("Joining existing sweep")
        sweep_id = args.sweep_id
        agent_count = args.count 

    print(f"Using sweep ID: {sweep_id}, launching {agent_count} agents")

    if agent_count == 0:
        return
    
    if args.algorithm == "madqn":
        train_func = madqn_train.train
    elif args.algorithm == "dqn":
        train_func = dqn_train.train
    else:
        raise ValueError("Unsupported algorithm")
    # 5) Launch the specified number of agents (in this container it's usually 1)
    wandb.agent(
        sweep_id,
        function=train_func,
        count=agent_count
    )

    wandb.finish()


if __name__ == "__main__":
    main()