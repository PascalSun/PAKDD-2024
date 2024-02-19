# PAKDD

This is the Code and Datasets for the paper

**"Are Graph Embeddings the Panacea? – an Empirical Survey from the Data Fitness Perspective"**

accepted by PAKDD 2024.

## Datasets

We have used the following datasets in our experiments:

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

## Experiment Design

![](./imgs/experiment-design.drawio_page-0001.jpg)

## Results

Similarity matrices for: F1 score (left), Accuracy score (middle), Network Parameters (|E|, K, L) (right).



<p align="center">
  <img src="imgs/similarity_matrix_macro_f_beta_score.jpg" alt="Similarity matrices for F1 score" width="30%" />
  <img src="imgs/similarity_matrix_accuracy_score.jpg" alt="Similarity matrices for Accuracy score" width="30%" />
  <img src="imgs/similarity_matrix_network_ln_num_edges_ln_num_classes_ln_average_shortest_path_length.jpg" alt="Similarity matrices for Network Parameters (|E|, K, L)" width="30%" />
</p>

## Reproduce the Results

### Setup

1. Clone the repository
2. Setup the conda environment
   ```bash
   conda create -n PAKDD-481 python=3.8
   conda activate PAKDD-481
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # install pytorch
   pip install git+https://github.com/pyg-team/pytorch_geometric.git  # install pytorch geometric
   pip install pyg_lib torch_scatter==1.4.0 torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html  
   pip install -r requirements.txt
   ```

3. Calculate the network characteristics
   First start the docker container with the following command:
   ```bash
   docker-compose up
   ```
   Then run the following command to calculate the network characteristics:
   ```bash
   python3 -m graph_metrics.dataset_network_metrics
   ```

4. Run the experiments
   ```bash
   bash ./scripts/all_datasets.sh
   ```

## Citation
