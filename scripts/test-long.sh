#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

MAX_STEPS=800
POLICIES=("c_sq" "d0_sq" "d1_sq" "d2_sq" "d3_sq" "d4_sq" "d5_sq" "d6_sq" "d7_sq" "d8_sq" "d9_sq")
for policy in "${POLICIES[@]}"; do
    ./cluster/run.sh scripts/baseline.py --policy $policy --n_trials 100 --n_workers 10 --max_steps $MAX_STEPS
done
# Define an array of checkpoint IDs
CHECKPOINTS=("8hlpz45j" "xdbf9fux" "o5tb680f")
# Loop through the checkpoints
for checkpoint in "${CHECKPOINTS[@]}"; do
    ./cluster/run.sh scripts/test.py --n_trials 100 --n_workers 20 --max_steps $MAX_STEPS --checkpoint wandb://damowerko-academic/motion-planning/$checkpoint
done
# Delay experiments
CHECKPOINTS=("xdbf9fux" "o5tb680f")
for checkpoint in "${CHECKPOINTS[@]}"; do
    ./cluster/run.sh ./scripts/delay.py --checkpoint wandb://damowerko-academic/motion-planning/$checkpoint --n_trials 100 --n_workers 10 --max_steps 800
done
