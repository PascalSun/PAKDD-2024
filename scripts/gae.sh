#!/bin/bash

# take a input params for dataset variable
dataset=${1:-Cora}
travel_dataset=${2:-Miami_FL}
# GAE: GCN as encoder
# centrality
python3 -m src.iid.run --model gae --gae_encoder gcn --gae_feature centrality --gcn_hidden_dim "32,16" --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random

# feature
python3 -m src.iid.run --model gae --gae_encoder gcn --gae_feature feature_1433 --gcn_hidden_dim "32,16" --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random

# GAE: GraphSAGE as encoder
# centrality
python3 -m src.iid.run --model gae --gae_encoder graph_sage --gae_feature centrality --graph_sage_aggr max --graph_sage_hidden_dim "32,16" --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random

# feature
python3 -m src.iid.run --model gae --gae_encoder graph_sage --gae_feature feature_1433 --graph_sage_aggr max --graph_sage_hidden_dim "32,16" --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random
