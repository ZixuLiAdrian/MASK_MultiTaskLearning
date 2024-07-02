import torch
import os
import random
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
import numpy as np


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """
    Splits the data into training, validation, and test sets.

    Parameters:
    - data: The dataset to be split.
    - train_ratio: The proportion of the dataset to include in the train split.
    - val_ratio: The proportion of the dataset to include in the validation split.

    Returns:
    - train_data, val_data, test_data: Split datasets.
    """
    num_nodes = data.num_nodes
    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def save_model(model, path="model.pth"):
    """
    Saves the model to the specified path.

    Parameters:
    - model: The PyTorch model to be saved.
    - path: The path where the model should be saved.
    """
    torch.save(model.state_dict(), path)


def load_model(model, path="model.pth", device="cuda:0"):
    """
    Loads the model from the specified path.

    Parameters:
    - model: The PyTorch model structure.
    - path: The path from where the model should be loaded.
    - device: The device to which the model should be moved.

    Returns:
    - model: The model with loaded weights.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def mask_node_features(data, mask_rate=0.15):
    """
    Masks a fraction of node features for the given data.

    Parameters:
    - data: Input data with node features.
    - mask_rate: Fraction of node features to mask.

    Returns:
    - data with masked node features and the original node features.
    """
    num_nodes = data.num_nodes
    num_mask = int(mask_rate * num_nodes)

    mask_indices = random.sample(range(num_nodes), num_mask)
    masked_node_features = data.x[mask_indices].clone()
    data.x[mask_indices] = 0  # or other masking strategy

    return data, masked_node_features, mask_indices


def mask_edges(data, mask_rate=0.15):
    """
    Masks a fraction of edges for the given data.

    Parameters:
    - data: Input data with edges.
    - mask_rate: Fraction of edges to mask.

    Returns:
    - data with masked edges and the original edges.
    """
    num_edges = data.edge_index.shape[1]
    num_mask = int(mask_rate * num_edges)

    mask_indices = random.sample(range(num_edges), num_mask)
    masked_edges = data.edge_index[:, mask_indices].clone()
    data.edge_index = torch.cat([data.edge_index[:, :mask_indices[0]], data.edge_index[:, mask_indices[-1] + 1:]],
                                dim=1)

    return data, masked_edges


def sample_subgraph(data, sample_rate=0.8):
    """
    Samples a subgraph from the given data.

    Parameters:
    - data: Input data from which subgraph is to be sampled.
    - sample_rate: Fraction of nodes to be included in the subgraph.

    Returns:
    - Sampled subgraph.
    """
    # This is a placeholder and might need a more sophisticated sampling mechanism
    num_nodes = data.num_nodes
    num_sample = int(sample_rate * num_nodes)

    sample_indices = random.sample(range(num_nodes), num_sample)

    subgraph = data.subgraph(torch.LongTensor(sample_indices))

    return subgraph


def inject_anomalies(data, anomaly_rate=0.05):
    """
    Injects anomalies into some of the nodes.

    Parameters:
    - data: Input data where anomalies are to be injected.
    - anomaly_rate: Fraction of nodes to be labeled as anomalies.

    Returns:
    - Data with injected anomalies.
    """
    num_nodes = data.num_nodes
    num_anomalies = int(anomaly_rate * num_nodes)

    anomaly_indices = random.sample(range(num_nodes), num_anomalies)
    data.y[anomaly_indices] = 1  # Assuming y is the label and 1 indicates anomaly

    return data


def generate_negative_edges(data, num_neg_samples):
    num_nodes = data.num_nodes
    existing_edges = {tuple(edge) for edge in data.edge_index.t().tolist()}

    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
            neg_edges.append([u, v])
            existing_edges.add((u, v))
    neg_edge_index = torch.tensor(neg_edges).t().contiguous()
    return neg_edge_index



