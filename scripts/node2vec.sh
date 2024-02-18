#!/bin/bash

# take a input params for dataset variable
dataset=${1:-Travel}
travel_dataset=${2:-Miami_FL}
# default dim is 10
# for the IID experiments, check the dim vs performance
python3 -m src.iid.run --model node2vec --node2vec_params_mode dim --start_dim 8 --end_dim 20 --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random

## walk length and context size vs performance
#python3 -m src.iid.run --model node2vec --node2vec_params_mode walk_length --node2vec_walk_length 30 --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random
#
## walks per node vs performance
#python3 -m src.iid.run --model node2vec --node2vec_params_mode walk_per_node --node2vec_walks_per_node 30 --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random
#
## q vs performance
#python3 -m src.iid.run --model node2vec --node2vec_params_mode p --node2vec_pq_step 0.1 --node2vec_p 3 --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random
#
## p vs performance
#python3 -m src.iid.run --model node2vec --node2vec_params_mode q --node2vec_pq_step 0.1 --node2vec_q 3 --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random
#
## num negative samples vs performance
#python3 -m src.iid.run --model node2vec --node2vec_params_mode num_negative_samples --node2vec_num_negative_samples 500 --dataset $dataset --balanced_ac --travel_dataset $travel_dataset --oversampling random
