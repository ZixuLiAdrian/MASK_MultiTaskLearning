import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import optuna
from sklearn.model_selection import KFold
from model import GNNEncoder, MOELayer, Decoder
from tasks import Task1, Task2, Task3
from load_data import load_data
from utils import mask_node_features, mask_edges, inject_anomalies, sample_subgraph, generate_negative_edges
from pygod.utils.utility import load_data
from decoders import Decoder_Task1, EdgePredictor_Task2, drop_nodes, permute_edges, subgraph, mask_nodes, \
    local_global_loss_, adj_loss_, mask_edges
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from sklearn.metrics import *

# Adding for fine tuning
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MultiObjectiveOptimization.multi_task.min_norm_solvers import MinNormSolver, gradient_normalizers

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='model description: Updated for benchmark comparison')
parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
args = parser.parse_args()


def set_seed(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(torch.seed())
        np.random.seed(np.random.randint(1, 10000))
        torch.cuda.manual_seed_all(torch.seed())


set_seed()

# Load data
data = load_data(args.dataset)
data = inject_anomalies(data)
input_dim = data.x.shape[1]


def calculate_anomaly_score(data, encoder, decoder):
    embeddings = encoder(data.x, data.edge_index)
    reconstructed_features = decoder(embeddings)
    reconstruction_error = torch.sum((data.x - reconstructed_features) ** 2, dim=1)
    anomaly_scores = reconstruction_error
    labels = torch.zeros(data.num_nodes, dtype=torch.int)
    if hasattr(data, 'anomaly_indices'):
        labels[data.anomaly_indices] = 1
    return anomaly_scores, labels


def eval_recall_at_k(true_labels, scores, k):
    true_labels = np.array(true_labels)
    scores = np.array(scores)
    sorted_indices = np.argsort(scores)[::-1]
    k = int(k)
    top_k_indices = sorted_indices[:k]
    true_positives = true_labels[top_k_indices].sum()
    possible_positives = true_labels.sum()
    recall_at_k = true_positives / float(possible_positives)
    return recall_at_k


def evaluate_model(data, encoder, decoder_task1):
    anomaly_scores, _ = calculate_anomaly_score(data, encoder, decoder_task1)
    true_labels = data.y.cpu().numpy()
    auc = roc_auc_score(true_labels, anomaly_scores.cpu().detach().numpy())
    ap = average_precision_score(true_labels, anomaly_scores.cpu().detach().numpy())
    k = true_labels.sum()
    recall_at_k = eval_recall_at_k(true_labels, anomaly_scores.cpu().detach().numpy(), k)
    return auc, ap, recall_at_k


def gradient_normalizers(grads, losses, normalization_type, epsilon=1e-8):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]])) + epsilon
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t] + epsilon
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]])) + epsilon
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn


class GNNEncoderWithBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(GNNEncoderWithBN, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x.unsqueeze(2)).squeeze(2)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x.unsqueeze(2)).squeeze(2)
        x = self.bn2(x)
        return x


def focal_loss(inputs, targets, alpha=1, gamma=2):
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()


def mask_edges_with_dropout(data, dropout_rate=0.2):
    edge_index = data.edge_index
    mask = torch.rand(edge_index.size(1)) > dropout_rate
    return edge_index[:, mask]


def get_data_subset(data, idx):
    idx_tensor = torch.tensor(idx, device=data.x.device)
    edge_mask = torch.isin(data.edge_index[0], idx_tensor) & torch.isin(data.edge_index[1], idx_tensor)
    edge_index = data.edge_index[:, edge_mask]

    # Create a mapping from old indices to new indices
    idx_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(idx_tensor)}

    # Filter and reindex edges
    edge_index = torch.stack(
        [torch.tensor([idx_mapping[int(node)] for node in edge_index_row], device=edge_index.device) for edge_index_row
         in edge_index], dim=0)

    return Data(
        x=data.x[idx],
        edge_index=edge_index,
        y=data.y[idx],
        anomaly_indices=data.anomaly_indices if hasattr(data, 'anomaly_indices') else None
    )


def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 512, 2048)
    output_dim = trial.suggest_int('output_dim', 128, 512)
    lr_encoder = trial.suggest_float('lr_encoder', 1e-4, 1e-2, log=True)
    lr_task1 = trial.suggest_float('lr_task1', 1e-4, 1e-3, log=True)
    lr_task2 = trial.suggest_float('lr_task2', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.001, 0.005, log=True)
    dropout = trial.suggest_float('dropout', 0.3, 0.7)
    num_epochs = 100
    patience = 10

    # Cross-validation
    num_folds = 5
    kf = KFold(n_splits=num_folds)
    fold_results = []

    best_loss_task1_list = []
    best_loss_task2_list = []
    best_loss_task3_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data.x)):
        train_data = get_data_subset(data, train_idx)
        val_data = get_data_subset(data, val_idx)

        # Model Initialization
        encoder = GNNEncoderWithBN(in_channels=train_data.num_features, out_channels=output_dim, dropout=dropout).to(
            args.device)
        decoder_task1 = Decoder_Task1(output_dim, output_dim=train_data.num_features).to(args.device)
        edge_predictor_task2 = EdgePredictor_Task2(output_dim, hidden_dim=output_dim, nbaselayer=3, dropout=dropout).to(
            args.device)

        # Debug print for dimensions
        print(f"Fold {fold + 1}/{num_folds}:")
        print(f"Train Data: x {train_data.x.size()}, edge_index {train_data.edge_index.size()}")
        print(f"Validation Data: x {val_data.x.size()}, edge_index {val_data.edge_index.size()}")

        # Optimizers
        optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=lr_encoder, weight_decay=weight_decay)
        optimizer_task1 = torch.optim.Adam(decoder_task1.parameters(), lr=lr_task1, weight_decay=weight_decay)
        optimizer_task2 = torch.optim.Adam(edge_predictor_task2.parameters(), lr=lr_task2, weight_decay=weight_decay)

        best_loss = float('inf')
        best_loss_task1 = None
        best_loss_task2 = None
        best_loss_task3 = None
        epochs_no_improve = 0
        current_loss = None

        for epoch in range(num_epochs):
            encoder.train()
            decoder_task1.train()
            edge_predictor_task2.train()

            optimizer_encoder.zero_grad()
            optimizer_task1.zero_grad()
            optimizer_task2.zero_grad()

            # Task 1: Node Feature Recovery
            _, masked_features, masked_indices = mask_node_features(train_data)
            embeddings = encoder(train_data.x, train_data.edge_index)
            adj = to_dense_adj(train_data.edge_index, max_num_nodes=train_data.x.size(0))[0]

            print(f"Epoch {epoch + 1}: Embeddings {embeddings.shape}, Adjacency Matrix {adj.shape}")

            reconstructed_features = decoder_task1(embeddings)
            loss_task1 = F.mse_loss(reconstructed_features[masked_indices], masked_features)

            # Task 2: Edge Prediction
            masked_edge_index = mask_edges_with_dropout(train_data, dropout_rate=0.2)
            neg_edge_index = generate_negative_edges(train_data, num_neg_samples=masked_edge_index.size(1) * 2)

            pos_pred = edge_predictor_task2(embeddings, adj, masked_edge_index)
            neg_pred = edge_predictor_task2(embeddings, adj, neg_edge_index)

            pos_pred = torch.clamp(pos_pred, min=-10, max=10)
            neg_pred = torch.clamp(neg_pred, min=-10, max=10)

            pos_pred = torch.sigmoid(pos_pred)
            neg_pred = torch.sigmoid(neg_pred)

            if torch.isnan(pos_pred).any() or torch.isnan(neg_pred).any():
                continue

            pos_labels = torch.ones(masked_edge_index.size(1), device=args.device)
            neg_labels = torch.zeros(neg_edge_index.size(1), device=args.device)
            predictions = torch.cat([pos_pred.squeeze(), neg_pred.squeeze()], dim=0)
            true_labels = torch.cat([pos_labels, neg_labels], dim=0)

            loss_task2 = focal_loss(predictions, true_labels)

            # Task 3: Subgraph Sampling
            subgraph_data = sample_subgraph(train_data)
            original_embedding = encoder(train_data.x, train_data.edge_index)
            augmented_embedding = encoder(subgraph_data.x, subgraph_data.edge_index)
            if train_data.batch is None:
                train_data.batch = torch.zeros(train_data.num_nodes, dtype=torch.long, device=train_data.x.device)
            loss_task3 = local_global_loss_(original_embedding, augmented_embedding, train_data.edge_index,
                                            train_data.batch, 'JSD')

            # Normalize losses
            norm_loss_task1 = loss_task1 / loss_task1.detach().item() if loss_task1.detach().item() != 0 else 0
            norm_loss_task2 = loss_task2 / loss_task2.detach().item() if loss_task2.detach().item() != 0 else 0
            norm_loss_task3 = loss_task3 / loss_task3.detach().item() if loss_task3.detach().item() != 0 else 0

            # Combine normalized losses using Min-Norm Solver
            grads = {
                'task1': list(torch.autograd.grad(norm_loss_task1, encoder.parameters(), retain_graph=True)),
                'task2': list(torch.autograd.grad(norm_loss_task2, encoder.parameters(), retain_graph=True)),
                'task3': list(torch.autograd.grad(norm_loss_task3, encoder.parameters(), retain_graph=True))
            }
            losses = {'task1': norm_loss_task1, 'task2': norm_loss_task2, 'task3': norm_loss_task3}

            gn = gradient_normalizers(grads, losses, 'none')
            for t in grads:
                for gr_i in range(len(grads[t])):
                    grads[t][gr_i] = grads[t][gr_i] / gn[t]

            sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in grads])
            weight_task1, weight_task2, weight_task3 = sol

            loss = weight_task1 * norm_loss_task1 + weight_task2 * norm_loss_task2 + weight_task3 * norm_loss_task3

            loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder_task1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(edge_predictor_task2.parameters(), max_norm=1.0)

            optimizer_encoder.step()
            optimizer_task1.step()
            optimizer_task2.step()

            current_loss = loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {current_loss}")

            if current_loss < best_loss:
                best_loss = current_loss
                best_loss_task1 = loss_task1.item()
                best_loss_task2 = loss_task2.item()
                best_loss_task3 = loss_task3.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Append the final losses for each task to the lists
        if best_loss_task1 is not None and best_loss_task2 is not None and best_loss_task3 is not None:
            best_loss_task1_list.append(best_loss_task1)
            best_loss_task2_list.append(best_loss_task2)
            best_loss_task3_list.append(best_loss_task3)
        else:
            best_loss_task1_list.append(float('inf'))
            best_loss_task2_list.append(float('inf'))
            best_loss_task3_list.append(float('inf'))

        # Evaluate on validation data
        encoder.eval()
        decoder_task1.eval()
        auc, ap, recall_at_k = evaluate_model(val_data, encoder, decoder_task1)
        fold_results.append((auc, ap, recall_at_k))

    # Calculate average results over all folds
    mean_auc = np.mean([result[0] for result in fold_results])
    std_auc = np.std([result[0] for result in fold_results])
    mean_ap = np.mean([result[1] for result in fold_results])
    std_ap = np.std([result[1] for result in fold_results])
    mean_recall = np.mean([result[2] for result in fold_results])
    std_recall = np.std([result[2] for result in fold_results])

    trial.set_user_attr("mean_auc", mean_auc)
    trial.set_user_attr("std_auc", std_auc)
    trial.set_user_attr("mean_ap", mean_ap)
    trial.set_user_attr("std_ap", std_ap)
    trial.set_user_attr("mean_recall", mean_recall)
    trial.set_user_attr("std_recall", std_recall)

    trial.set_user_attr("best_loss_task1", np.mean(best_loss_task1_list))
    trial.set_user_attr("best_loss_task2", np.mean(best_loss_task2_list))
    trial.set_user_attr("best_loss_task3", np.mean(best_loss_task3_list))

    return -mean_auc  # Optuna minimizes the objective, so return negative AUC


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial
print(f"Value: {-trial.value}")  # Convert back to positive for interpretation
print("Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Retrieve and print the best results
mean_auc = trial.user_attrs.get("mean_auc")
std_auc = trial.user_attrs.get("std_auc")
mean_ap = trial.user_attrs.get("mean_ap")
std_ap = trial.user_attrs.get("std_ap")
mean_recall = trial.user_attrs.get("mean_recall")
std_recall = trial.user_attrs.get("std_recall")
print(f"Final AUC: {mean_auc:.4f} ± {std_auc:.4f}")
print(f"Final AP: {mean_ap:.4f} ± {std_ap:.4f}")
print(f"Final Recall@k: {mean_recall:.4f} ± {std_recall:.4f}")

# Print the final losses for each task
best_loss_task1 = trial.user_attrs.get("best_loss_task1")
best_loss_task2 = trial.user_attrs.get("best_loss_task2")
best_loss_task3 = trial.user_attrs.get("best_loss_task3")
print(f"Final Task 1 Loss: {best_loss_task1:.4f}")
print(f"Final Task 2 Loss: {best_loss_task2:.4f}")
print(f"Final Task 3 Loss: {best_loss_task3:.4f}")
