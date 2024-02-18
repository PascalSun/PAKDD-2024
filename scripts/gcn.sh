#!/bin/bash

# take a input params for dataset variable
dataset=${1:-Cora}
travel_dataset=${2:-Miami_FL}

# gcn with centrality features, supervised learning
python3 -m src.iid.run --model gcn --gcn_type supervised_centrality --gcn_hidden_dim "16" --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random

# gcn with 1433 features, supervised learning, seems like layers of this also not matter
python3 -m src.iid.run --model gcn --gcn_type supervised_feature --gcn_hidden_dim "16" --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random

# gcn unsupervised, with centrality
python3 -m src.iid.run --model gcn --gcn_type unsupervised --gcn_dataset_type centrality --gcn_hidden_dim "16" --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random

# gcn unsupervised, with 1433 features
python3 -m src.iid.run --model gcn --gcn_type unsupervised --gcn_dataset_type feature_1433 --gcn_hidden_dim "16" --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random
