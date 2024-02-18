#!/bin/bash

datasets=(
#    "KarateClub"
#    "Mitcham"
    "PubMed"
    "CiteSeer"
    "Cora"
    "AMAZON_COMPUTERS"
    "AMAZON_PHOTO"
#    "NELL"
    "CitationsFull_Cora"
    "CitationsFull_CiteSeer"
    "CitationsFull_PubMed"
    "CitationsFull_Cora_ML"
    "CitationsFull_DBLP"
    "Cora_Full"
    "Coauther_CS"
    "Coauther_Physics"
    "Flickr"
#    "Yelp"
    "AttributedGraphDataset_Wiki"
    "AttributedGraphDataset_Cora"
    "AttributedGraphDataset_CiteSeer"
    "AttributedGraphDataset_Pubmed"
    "AttributedGraphDataset_BlogCatalog"
#    "AttributedGraphDataset_PPI"
    "AttributedGraphDataset_Flickr"
#    "AttributedGraphDataset_Facebook"
    "WEBKB_Cornell"
    "WEBKB_Texas"
    "WEBKB_Wisconsin"
    "HeterophilousGraphDataset_Roman_empire"
    "HeterophilousGraphDataset_Amazon_ratings"
    "HeterophilousGraphDataset_Minesweeper"
    "HeterophilousGraphDataset_Tolokers"
    "HeterophilousGraphDataset_Questions"
    "Actor"
    "GitHub"
    "TWITCH_DE"
    "TWITCH_EN"
    "TWITCH_ES"
    "TWITCH_FR"
    "TWITCH_PT"
    "TWITCH_RU"
    "PolBlogs"
    "EllipticBitcoinDataset"
    "DGraphFin"
    "Reddit"
    "AMAZON_PRODUCTS"
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
    scripts=("graph_sage.sh")

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
