import torch
import torch.nn as nn
import torch.nn.functional as F


class Task1(nn.Module):
    """
    Decoder for node feature reconstruction.
    """

    def __init__(self, in_features, out_features):
        super(Task1, self).__init__()
        self.decoder = nn.Linear(in_features, out_features)

    def forward(self, embeddings, masked_indices):
        reconstructed_features = self.decoder(embeddings[masked_indices])
        return reconstructed_features


class Task2(nn.Module):
    """
    Decoder for edge prediction.
    """

    def __init__(self, in_features):
        super(Task2, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2 * in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings1, embeddings2):
        concat_embeddings = torch.cat([embeddings1, embeddings2], dim=1)
        return self.decoder(concat_embeddings)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Task 3.
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, positive_distance, negative_distance):
        loss = (positive_distance
                + F.relu(self.margin - negative_distance)).mean()
        return loss


class Task3:
    """
    Contrastive learning task. This doesn't involve a neural network directly,
    but rather computes the loss based on the embeddings provided.
    """

    def __init__(self):
        self.loss_fn = ContrastiveLoss()

    def compute_loss(self, positive_distance, negative_distance):
        return self.loss_fn(positive_distance, negative_distance)
