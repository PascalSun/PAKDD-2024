#!/bin/bash

datasets=(
    "Actor"
    "AMAZON_COMPUTERS"
    "AMAZON_PHOTO"
    "AttributedGraphDataset_BlogCatalog"
    "AttributedGraphDataset_CiteSeer"
    "AttributedGraphDataset_Cora"
    "AttributedGraphDataset_Flickr"
    "AttributedGraphDataset_Pubmed"
    "AttributedGraphDataset_Wiki"
    "CitationsFull_CiteSeer"
    "CitationsFull_Cora"
    "CitationsFull_Cora_ML"
    "CitationsFull_DBLP"
    "CitationsFull_PubMed"
    "CiteSeer"
    "Coauther_CS"
    "Coauther_Physics"
    "Cora"
    "Cora_Full"
    "GitHub"
    "HeterophilousGraphDataset_Amazon_ratings"
    "HeterophilousGraphDataset_Minesweeper"
    "HeterophilousGraphDataset_Questions"
    "HeterophilousGraphDataset_Roman_empire"
    "HeterophilousGraphDataset_Tolokers"
    "PubMed"
    "TWITCH_DE"
    "TWITCH_EN"
    "TWITCH_ES"
    "TWITCH_FR"
    "TWITCH_PT"
    "TWITCH_RU"
    "WEBKB_Cornell"
    "WEBKB_Texas"
    "WEBKB_Wisconsin"
)



# Function to check if a dataset and script combination has been processed
has_been_processed() {
    local dataset="$1"
    local script="$2"
    grep -q "${dataset}-${script}" "reports/model_checkpoints.txt"
}

# Function to mark a dataset and script combination as processed
mark_as_processed() {
    local dataset="$1"
    local script="$2"
    echo "${dataset}-${script}" >> reports/model_checkpoints.txt
}

# Loop through each dataset and call other scripts
for dataset in "${datasets[@]}"; do
    echo "Processing $dataset"

    # Example scripts
    scripts=("basic.sh" "node2vec.sh" "gae.sh" "graph_sage.sh" "gcn.sh")

    for script in "${scripts[@]}"; do
        if has_been_processed "$dataset" "$script"; then
            echo "Script $script for dataset $dataset already processed. Skipping."
            continue
        fi

        bash "scripts/$script" $dataset &
        # echo it is done
        echo "Script $script for dataset $dataset done."
        mark_as_processed "$dataset" "$script"
    done

    # Optional: Wait for all background processes to finish before moving to the next dataset
    wait
done

# Wait for all background processes to finish before exiting the script
wait
echo "All datasets processed."
