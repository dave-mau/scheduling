#!/bin/bash

# Get repo root directory
INIT_DIR=$(pwd)
REPO_DIR=$(git rev-parse --show-toplevel)
DATA_DIR="$REPO_DIR/code/notebooks/02_three_stage_system/hparam_optimization"

ALGO="ppo"
ENV="ParsedHierarchicalSystem-v0"
SYSTEM_CONFIG_FILE="$REPO_DIR/code/notebooks/02_three_stage_system/system_config.json"

cd $REPO_DIR/code/rl-baselines3-zoo
python3 train.py \
    --algo $ALGO \
    --env $ENV \
    --env-kwargs system_config_file:\"${SYSTEM_CONFIG_FILE}\" episode_length:1000 render_mode:None \
    --eval-env-kwargs system_config_file:\"${SYSTEM_CONFIG_FILE}\" episode_length:1000 render_mode:None \
    --optimization-log-path "$DATA_DIR/logs/hparam/" \
    --log-folder "$DATA_DIR/logs/train/" \
    --save-freq -1 \
    --device "cpu" \
    --optimize-hyperparameters \
    --n-jobs 25 \
    --sampler "tpe" \
    --pruner "median" \
    --n-evaluations 100000 \
    --eval-episodes 30 \
    --n-eval-envs 1 \
    --progress
cd $INIT_DIR

