#!/bin/bash

# take a input params for dataset variable
dataset=${1:-Cora}
travel_dataset=${2:-Miami_FL}
# graph sage with centrality features, supervised learning
python3 -m run --model graph_sage --graph_sage_type supervised_centrality --graph_sage_aggr max --graph_sage_hidden_dim "32,16" --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random
#
# graph sage with 1433 features, supervised learning, seems like layers of this also not matter
python3 -m run --model graph_sage --graph_sage_type supervised_feature --graph_sage_aggr max --graph_sage_hidden_dim "300,200,100" --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random

# graph sage unsupervised, with centrality
python3 -m run --model graph_sage --graph_sage_type unsupervised --graph_sage_dataset_type centrality --graph_sage_hidden_dim "300" --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random

# graph sage unsupervised, with 1433 features
python3 -m run --model graph_sage --graph_sage_type unsupervised --graph_sage_dataset_type feature_1433 --graph_sage_aggr max --graph_sage_hidden_dim "300" --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random

