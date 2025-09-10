#!/usr/bin/env python3
import argparse
import os
import socket
from datetime import datetime
import wandb

# Import evaluation functions for all algorithms
from experiments.madqn import madqn_train
from experiments.dqn import dqn_train, dqn_train_pmv
from experiments.madqn import ac_only_train
from experiments.madqn import madqn_train_fully_competetive
from experiments.madqn import madqn_train_part_competetive
from experiments.madqn import madqn_train_part_cooperative
from experiments.madqn import madqn_train_fully_cooperative
from experiments.dqn import dqn_train_pmv_fan_only, dqn_train_pmv_ac_only
from experiments.sac import sac_train_pmv
from experiments.sac import masac_train_fully_competetive, masac_train_fully_cooperative

from utils.experiment_utils import create_experiment_name


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate pre-trained RL-HVAC models")
    p.add_argument("--workdir", required=True, help="Base folder for evaluation logs")
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "A403-Eval"))
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--algorithm", required=True, choices=[
        "madqn", "dqn", "dqn_pmv", "ac_only",
        "madqn_fully_competitive", "madqn_fully_cooperative",
        "madqn_part_competitive", "madqn_part_cooperative",
        "dqn_pmv_fan_only", "dqn_pmv_ac_only",
        "sac", "masac_fully_competetive", "masac_fully_cooperative"
    ])
    p.add_argument("--model-path", required=True, help="Path to pre-trained AC model (.pth)")
    p.add_argument("--fan-model-path", default=os.environ.get("MODEL_PATH_FAN", None),
                   help="Optional path to pre-trained Fan model (.pth) for multi-agent")
    p.add_argument("--env-id", required=True, help="Environment ID, e.g., A403small")
    p.add_argument("--weather", required=True, choices=["hot", "cool", "mixed"])
    # Model hyperparameters for logging
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--co2-weight", type=float, default=None)
    p.add_argument("--pmv-weight", type=float, default=None)
    p.add_argument("--switching-penalty", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    # === Setup directories ===
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.workdir, f"eval_{hostname}_{timestamp}_{os.urandom(2).hex()}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Writing evaluation outputs to: {run_dir}")

    ENV_NAME = f"{args.env_id}_{args.weather}_MULTISPEED_FAN"
    ALGORITHM_NAME = args.algorithm.upper()

    # Create experiment name and save dir
    eval_experiment_name = create_experiment_name(env_name=ENV_NAME, episodes=1, algorithm_name=ALGORITHM_NAME)
    experiment_save_dir_name = os.path.join(run_dir, "results", args.algorithm, eval_experiment_name)
    os.makedirs(experiment_save_dir_name, exist_ok=True)

    print(f"Evaluating model: {args.model_path}")
    print(f"Environment: {ENV_NAME}")
    print(f"Algorithm: {ALGORITHM_NAME}")
    print(f"Results will be stored in: {experiment_save_dir_name}")

    # === WandB setup ===
    config = {
            "method": "grid",
            "project": "A403-Eval",
            "name": eval_experiment_name,
            "metric": {"name": "final_val_power_kWh_mean", "goal": "minimize"},
            "parameters": {
                    'learning_rate':{'value': 3e-3}, #{'values': [3e-4,1e-3,3e-3]},
                    'lambda_energy': {'value': 1/1_600_000}, # [1/2_000_000,1/1_600_000,1/1_200_000]
                    'gamma': {'value':0.95}, # [0.90,0.95,0.99]
                    'co2_weight':{'value': 0.9},#{'values':[0.30,0.40,0.50]},#{'min':0.30,'max':0.60},# {'min':0.30,'max':0.60}, # 0.2 yapma 
                    #'temp_weight': {'min':0.20,'max':0.80}, #[0.40,0.50,0.60]
                    'pmv_weight': {'value': 0.3},#{'values':[0.40,0.50,0.60,0.70]},#{'values':[0.50,0.60,0.70]},# {'min':0.40,'max':0.70},
                    'switching_penalty': {'value': 0},#{'values':[0.00,0.05,0.10]},
                    'experiment_save_dir': {'value': experiment_save_dir_name},
                    'model_path_ac': {'value': args.model_path},
                    'model_path_fan': {'value': args.fan_model_path},
                    'train_season': {'value': args.weather},
                    'agent_count': {'value': 81},
                    'layer_sizes': {'value': [256,256,256]},
                    'env_id': {'value': args.env_id},
                    'memory_capacity': {'value': 2*52600}, # [52600,2*52600,4*52600]
                    'num_episodes': {'value': 1},  # Evaluation typically runs for 1 episode
                }
        }
    sweep_id = wandb.sweep(config, project=args.project, entity=args.entity)
    

    # === Map algorithm to evaluation function ===

    if args.algorithm == "dqn_pmv":
        eval_func = dqn_train_pmv.evaluate
    elif args.algorithm == "madqn_part_competitive":
        eval_func = madqn_train_part_competetive.evaluate
    elif args.algorithm == "madqn_fully_cooperative":
        eval_func = madqn_train_fully_cooperative.evaluate
    elif args.algorithm == "madqn_part_cooperative":
        eval_func = madqn_train_part_cooperative.evaluate
    elif args.algorithm == "madqn_fully_competitive":
        eval_func = madqn_train_fully_competetive.evaluate
    else:
        raise ValueError(f"Unsupported algorithm for evaluation: {args.algorithm}")

    wandb.agent(
        sweep_id,
        function=eval_func,
        count=1
    )

    wandb.finish()


if __name__ == "__main__":
    main()
