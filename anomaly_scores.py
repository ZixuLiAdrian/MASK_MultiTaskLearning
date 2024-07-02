import torch


def compute_anomaly_scores_from_embeddings(original_embeddings, reconstructed_embeddings):
    """
    Compute anomaly scores based on the difference between original and reconstructed embeddings.

    Parameters:
    - original_embeddings: The original embeddings of nodes (from the encoder)
    - reconstructed_embeddings: The embeddings obtained after decoding

    Returns:
    - scores: Anomaly scores for each node
    """

    diff = torch.norm(original_embeddings - reconstructed_embeddings, dim=1)

    return diff
