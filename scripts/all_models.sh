#!/bin/bash

# take a input params for dataset variable
dataset=${1:-Cora}

travel_dataset=${2:-Miami_FL}




# Function to check if a dataset and script combination has been processed
has_been_processed() {
    local dataset="$1"
    local script="$2"
    grep -q "${dataset}-${script}" "reports/iid/summary/model_checkpoints.txt"
}

# Function to mark a dataset and script combination as processed
mark_as_processed() {
    local dataset="$1"
    local script="$2"
    echo "${dataset}-${script}" >> reports/iid/summary/model_checkpoints.txt
}

scripts=("basic.sh" "node2vec.sh" "gae.sh" "graph_sage.sh" "gcn.sh")
for script in "${scripts[@]}"; do
        if has_been_processed "$dataset" "$script"; then
            echo "Script $script for dataset $dataset already processed. Skipping."
            continue
        fi

        bash "scripts/bash/iid/$script" $dataset
        # echo it is done
        echo "Script $script for dataset $dataset done."
        mark_as_processed "$dataset" "$script"
  done

    # Optional: Wait for all background processes to finish before moving to the next dataset
wait

## run all other scripts inside this folder
#bash scripts/bash/iid/basic.sh $dataset $travel_dataset
#bash scripts/bash/iid/gae.sh $dataset $travel_dataset
#bash scripts/bash/iid/graph_sage.sh $dataset $travel_dataset
#bash scripts/bash/iid/node2vec.sh $dataset $travel_dataset
#bash scripts/bash/iid/gcn.sh $dataset $travel_dataset
