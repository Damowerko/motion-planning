#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

N_TRIALS=100
MAX_STEPS=1000
# POLICIES=("c" "c_sq" "d0_sq" "d1_sq" "d2_sq" "d3_sq" "d4_sq" "d5_sq" "d6_sq" "d7_sq" "d8_sq" "d9_sq")
POLICIES=("d2_sq" "d6_sq")
for policy in "${POLICIES[@]}"; do
    ./cluster/run_local.sh scripts/baseline.py --policy $policy --n_trials $N_TRIALS --n_workers 10 --max_steps $MAX_STEPS --no-scenarios --no-delay
done
# Define an array of checkpoint IDs
# CHECKPOINTS=("s3mmghbq" "lltq0n3w" "yrja7uof" "b6f30ap9" "ka8rfwv5" "n3mq95uy")
# CHECKPOINTS=("yrja7uof" "b6f30ap9" "ka8rfwv5" "n3mq95uy")
# CHECKPOINTS=("yfuzvibe" "1g3cd75x" "bzcwzx15")
# # Loop through the checkpoints
# for checkpoint in "${CHECKPOINTS[@]}"; do
#     ./cluster/run.sh scripts/test.py --n_trials 100 --n_workers 10 --max_steps $MAX_STEPS --checkpoint wandb://damowerko-academic/motion-planning/$checkpoint --best
# done
# # Delay experiments
# CHECKPOINTS=("xdbf9fux" "o5tb680f")
# for checkpoint in "${CHECKPOINTS[@]}"; do
#     ./cluster/run.sh ./scripts/delay.py --checkpoint wandb://damowerko-academic/motion-planning/$checkpoint --n_trials 100 --n_workers 10 --max_steps $MAX_STEPS
# done
