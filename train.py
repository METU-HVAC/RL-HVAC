#!/usr/bin/env python3
import argparse
import os
import json
import socket
from datetime import datetime
import wandb
# import warnings
# warnings.filterwarnings("ignore", message=".*Casting input x to numpy array.*")

from experiments.madqn.madqn_train import * 

def parse_args():
    p = argparse.ArgumentParser(description="Main entrypoint for RL-HVAC sweep/agent")
    p.add_argument(
        "--workdir",
        required=True,
        help="Base folder under which each container gets its own subfolder"
    )
    p.add_argument(
        "--sweep-id",
        default=None,
        help="If set, join this existing sweep instead of creating a new one"
    )
    p.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "A403-Train"),
        help="W&B project name"
    )
    p.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity (user or team)"
    )
    p.add_argument(
        "--count",
        type=int,
        default=None,
        help="How many agents to launch (only relevant when creating the sweep)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Create a unique sub‐folder per container, e.g. by hostname + timestamp
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #get a random 2 digit hex number
    #timestamp = f"{timestamp}_{os.urandom(2).hex()}"
    run_dir = os.path.join(args.workdir, f"{hostname}_{timestamp}_{os.urandom(2).hex()}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Writing outputs to: {run_dir}")
    
    # Create experiment save dir
    train_season = "hot"
    ENV_ID ="A403medium"             
    unique_experiment_name = f"{train_season}_{ENV_ID}_train_{timestamp}"
    experiment_save_dir_name = run_dir+"/results/madqn/" + unique_experiment_name
    if not os.path.exists(experiment_save_dir_name):
        os.makedirs(experiment_save_dir_name)


    ENV_NAME = f"{ENV_ID}_{train_season}_64_64_MULTISPEED_FAN" 
    ALGORITHM_NAME = "MADQN"
    NUM_EPISODES = 10
    name = create_experiment_name(env_name=ENV_NAME, episodes=NUM_EPISODES,algorithm_name=ALGORITHM_NAME)

    # 3) Initialize W&B (no project/sweep yet)
    # wandb.init(
    #     project=args.project,
    #     entity=args.entity,
    #     dir=run_dir,
    #     reinit=True,
    #     )

    # 4) Create the sweep only if --sweep-id wasn't provided
    if args.sweep_id is None:
        # Build your sweep config dict however you like:
        sweep_config = {
            "method": "random",
            "project": "A403-Train",
            "name": name,
            "metric": {"name": "final_val_power_kWh_mean", "goal": "minimize"},
            "parameters": {
                    'learning_rate': {
                        #'values': [3e-4,1e-3,3e-3]
                        #'values': [3e-4]
                        'min': 1e-4,
                        'max': 3e-3,
                    },
                    'lambda_energy': {
                        'values': [1/1_600_000]
                    },
                    'gamma': {
                        'min': 0.8,
                        'max': 0.99,
                    },
                    'co2_weight': {
                        'min': 0.10,
                        'max': 0.90,
                    },
                    'temp_weight': {
                        ## When temp is 1 energy be from 1 to 3. Which means temp weight can be from 0.25 to 0.50
                        'min': 0.10,
                        'max': 0.90,
                    },
                    'experiment_save_dir': {
                        'value': experiment_save_dir_name
                    },
                    'train_season': {
                        'value': train_season
                    },
                    'agent_count': {
                        'value': 40
                    },
                    'num_episodes': {
                        'value': NUM_EPISODES
                    },
                    'env_id': {
                        'value': ENV_ID
                    },
                }
        }
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        agent_count = args.count
    else:
        print("Joining existing sweep")
        sweep_id = args.sweep_id
        agent_count = args.count 

    print(f"Using sweep ID: {sweep_id}, launching {agent_count} agents")

    # If count == 0, do not start any agents: just exit
    if agent_count == 0:
        return
    # 5) Launch the specified number of agents (in this container it's usually 1)
    wandb.agent(
        sweep_id,
        function=train,
        count=agent_count
    )

    wandb.finish()


if __name__ == "__main__":
    main()