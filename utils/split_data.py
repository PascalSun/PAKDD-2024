import torch


def split_data(dataset):
    num_nodes = dataset[0].num_nodes
    num_train = int(num_nodes * 0.5)
    num_val = int(num_nodes * 0.2)

    # Shuffle nodes and create masks
    permuted_nodes = torch.randperm(num_nodes)
    train_nodes = permuted_nodes[:num_train]
    val_nodes = permuted_nodes[num_train : num_train + num_val]
    test_nodes = permuted_nodes[num_train + num_val :]

    dataset.data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    dataset.data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    dataset.data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    dataset.data.train_mask[train_nodes] = True
    dataset.data.val_mask[val_nodes] = True
    dataset.data.test_mask[test_nodes] = True
    return dataset
