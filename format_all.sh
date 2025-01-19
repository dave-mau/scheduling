#! /usr/bin/bash
black . --workers 8 --quiet -l 119 --exclude 'rl-baselines3-zoo'
isort --profile black --skip 'rl-baselines3-zoo' . 
