# PAKDD

This is the Code and Datasets for the paper "Are Graph Embeddings the Panacea? – an Empirical Survey from the Data
Fitness Perspective" accepted by PAKDD 2024.

## Datasets

We have used the following datasets in our experiments:

- `cora`
- `citeseer`

## Algorithms

## Results

## Reproduce the Results

### Setup

1. Clone the repository
2. Setup the conda environment

```bash
conda create -n PAKDD-481 python=3.8
conda activate PAKDD-481
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # install pytorch
conda install pyg -c pyg # install pytorch geometric
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
