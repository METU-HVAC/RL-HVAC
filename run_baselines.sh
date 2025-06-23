#!/bin/bash
# run_baselines.sh
# This script runs baseline_test.py for multiple combinations of room, season, and algorithm.
# It sets PYTHONPATH so that the 'algorithms' module is available.
#
# Usage: ./run_baselines.sh

# Set the project root directory so that the "algorithms" module is found.
export PYTHONPATH="/home/mehmetbh/workspace/RL-Paper:$PYTHONPATH"

# Path to the Python interpreter and your test script.
PYTHON="/home/mehmetbh/workspace/RL-Paper/venv/bin/python"
SCRIPT="/home/mehmetbh/workspace/RL-Paper/experiments/baselines/baseline_test.py"
# please use FULL WINDOW FAN CONTROL
# Define parameter lists.
rooms=("A403mediumwindow")
# seasons=("hot" "cool" "mixed")
seasons=("hot")
# algorithms=("setpoint05" "setpoint075" "setpoint1"
#             "on_off05" "on_off075" "on_off1"
#             "window_on_off" "window_schedule"
#             "multispeed_setpoint")

algorithms=(
            "window_on_off" "window_schedule"
            )
#algorithms=("multispeed_setpoint")
# Loop through each combination and run the test sequentially.
for room in "${rooms[@]}"; do
    for season in "${seasons[@]}"; do
        for algo in "${algorithms[@]}"; do
            echo "Running baseline_test.py with room: ${room}, season: ${season}, algorithm: ${algo}"
            $PYTHON $SCRIPT --room "$room" --season "$season" --algorithm "$algo"
        done
    done
done
