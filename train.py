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
from experiments.dqn import dqn_train_pmv 
from experiments.madqn import ac_only_train
from experiments.madqn import madqn_train_fully_competetive
from experiments.madqn import madqn_train_fully_competetive
from experiments.madqn import madqn_train_part_competetive
from experiments.madqn import madqn_train_part_cooperative
from experiments.dqn import dqn_train_pmv_fan_only
from experiments.dqn import dqn_train_pmv_ac_only
from experiments.sac import sac_train_pmv
from experiments.sac import masac_train_fully_competetive
from experiments.sac import masac_train_fully_cooperative
from utils.experiment_utils import create_experiment_name
def parse_args():
    p = argparse.ArgumentParser(description="Main entrypoint for RL-HVAC sweep/agent")
    p.add_argument("--workdir",required=True,help="Base folder logs")
    p.add_argument("--sweep-id",default=None,help="Join this existing sweep")
    p.add_argument("--project",default=os.environ.get("WANDB_PROJECT", "A403-Train"))
    p.add_argument("--entity",default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--count",type=int,default=None)
    p.add_argument("--algorithm", required=True, choices=["madqn", "dqn" , "dqn_pmv","ac_only",
                                                          "madqn_fully_competitive",
                                                          "madqn_fully_cooperative",
                                                          "madqn_part_competitive",
                                                          "madqn_part_cooperative",
                                                          "dqn_pmv_fan_only","dqn_pmv_ac_only",
                                                          "sac","masac_fully_competetive", "masac_fully_cooperative"
                                                          ], help="RL algorithm to use")
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
    ENV_ID ="A403mediumfanger"
    NUM_EPISODES = 10          
    unique_experiment_name = f"{train_season}_{ENV_ID}_train_{timestamp}"
   
    experiment_save_dir_name = os.path.join(run_dir, "results", args.algorithm, unique_experiment_name)

    if not os.path.exists(experiment_save_dir_name):
        os.makedirs(experiment_save_dir_name)

    ENV_NAME = f"{ENV_ID}_{train_season}_MULTISPEED_FAN"
    ALGORITHM_NAME = args.algorithm.upper()
    
    name = create_experiment_name(env_name=ENV_NAME, episodes=NUM_EPISODES,algorithm_name=ALGORITHM_NAME)

    if args.sweep_id is None:
        # Build your sweep config dict however you like:
        sweep_config = {
            "method": "grid",
            "project": "A403-Train",
            "name": name,
            "metric": {"name": "final_val_power_kWh_mean", "goal": "minimize"},
            "parameters": {
                    'learning_rate':{'value': 3e-3}, #{'values': [3e-4,1e-3,3e-3]},
                    'lambda_energy': {'value': 1/1_600_000}, # [1/2_000_000,1/1_600_000,1/1_200_000]
                    'gamma': {'value':0.95}, # [0.90,0.95,0.99]
                    'co2_weight':{'values':[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]},#{'values':[0.30,0.40,0.50]},#{'min':0.30,'max':0.60},# {'min':0.30,'max':0.60}, # 0.2 yapma 
                    #'temp_weight': {'min':0.20,'max':0.80}, #[0.40,0.50,0.60]
                    'pmv_weight': {'values':[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]},#{'values':[0.40,0.50,0.60,0.70]},#{'values':[0.50,0.60,0.70]},# {'min':0.40,'max':0.70},
                    'switching_penalty': {'value': 0},#{'values':[0.00,0.05,0.10]},
                    'experiment_save_dir': {'value': experiment_save_dir_name},
                    'train_season': {'value': train_season},
                    'agent_count': {'value': 81},
                    'num_episodes': {'value': NUM_EPISODES},
                    'layer_sizes': {'value': [256,256,256]},
                    'env_id': {'value': ENV_ID},
                    'memory_capacity': {'value': 2*52600}, # [52600,2*52600,4*52600]
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
    elif args.algorithm == "ac_only":
        train_func = ac_only_train.train
    # elif args.algorithm == "five_zone_train":
    #     train_func = five_zone_train.train
    elif args.algorithm == "dqn_pmv":
        train_func = dqn_train_pmv.train
    elif args.algorithm == "madqn_fully_competitive":
        train_func = madqn_train_fully_competetive.train
    elif args.algorithm == "madqn_fully_cooperative":
        train_func = madqn_train_fully_competetive.train
    elif args.algorithm == "madqn_part_competitive":
        train_func = madqn_train_part_competetive.train
    elif args.algorithm == "madqn_part_cooperative":
        train_func = madqn_train_part_cooperative.train
    elif args.algorithm == "dqn_pmv_fan_only":
        train_func = dqn_train_pmv_fan_only.train
    elif args.algorithm == "dqn_pmv_ac_only":
        train_func = dqn_train_pmv_ac_only.train
    elif args.algorithm == "sac":
        train_func = sac_train_pmv.train
    elif args.algorithm == "masac_fully_competetive":
        train_func = masac_train_fully_competetive.train
    elif args.algorithm == "masac_fully_cooperative":
        train_func = masac_train_fully_cooperative.train
    
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