#! /usr/bin/bash
EXCLUDE_DIRS="rl-baselines3-zoo"
black . --workers 8 --quiet -l 119 --exclude $EXCLUDE_DIRS
isort --profile black --skip $EXCLUDE_DIRS . 
