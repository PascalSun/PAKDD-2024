#!/bin/bash

# take a input params for dataset variable
dataset=${1:-Cora}

travel_dataset=${2:-Miami_FL}

# feature centrality vs performance
python3 -m run --model feature_centrality --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random

# feature 1433 vs performance
python3 -m run --model feature_1433 --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random

# random input vs performance
python3 -m run --model random_input --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random
