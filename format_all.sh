#! /usr/bin/bash
black . --workers 8 --quiet -l 119
isort --profile black .
