#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh


# Run a prepared experiment (all configs overriden in a .yaml folder, no CLI)
python python src/train.py experiment=x_cvtfl
