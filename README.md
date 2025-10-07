# SENSEI parse

This repository contains code for the experiments carried out in our submitted paper **"SENSEI-ASG: A Challenging Dataset for Argument Summary Graph Parsing"**

## Installation

1. **Create a Conda environment:**

conda create --name sensei_parse python=3
conda activate sensei_parse

2. **Install PyTorch**
Follow the instructions here: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

3. **Install other requirements:**

pip install -r requirements.txt

## Run Experiments

The file `run_experiments.sh` takes one or more training dataset names as command line arguments. Dataset names should be separated by an underscore (`_`).

To reproduce all experiments, run all of the following commands in turn:

run_experiments.sh sensei
run_experiments.sh debatabase
run_experiments.sh argessays
run_experiments.sh sensei_debatabase
run_experiments.sh sensei_argessays
run_experiments.sh debatabase_argessays
run_experiments.sh sensei_debatabase_argessays
