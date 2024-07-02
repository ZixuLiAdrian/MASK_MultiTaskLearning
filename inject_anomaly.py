import torch
import numpy as np


def inject_anomalies(data, num_anomalies, anomaly_magnitude=1.0):
    """
    Inject anomalies into the dataset.

    Parameters:
    - data: The graph data (assumed to be in PyG format)
    - num_anomalies: Number of nodes to inject with anomalies
    - anomaly_magnitude: The magnitude of the anomaly (how much to perturb the node features)

    Returns:
    - data: Modified data with anomalies
    """

    # 随机选一些node
    chosen_nodes = np.random.choice(data.num_nodes, size=num_anomalies, replace=False)

    # 把选到的幸运小node改一改
    for node in chosen_nodes:
        perturbation = torch.randn_like(data.x[node]) * anomaly_magnitude
        data.x[node] += perturbation

    return data
