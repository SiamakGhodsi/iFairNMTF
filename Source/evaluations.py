import numpy as np
from itertools import permutations
import torch.nn.functional as F
import torch
import math


def get_membership_matrix(indices: np.ndarray) -> np.ndarray:
    """
    :param indices: (num_nodes,) Membership indices for nodes
    :return membership_matrix: (num_nodes, num_indices) Membership matrix
    """
    num_indices = np.max(indices) + 1
    return np.eye(num_indices)[indices, :]


def compute_group_balance(clusters: np.ndarray, groups: np.ndarray, normalize: bool = False) -> (np.ndarray, float):
    """
    :param clusters: (num_nodes,) Predicted cluster identities
    :param groups: (num_nodes,) Protected group memberships
    :param normalize: Whether to balance the groups with group size
    :return balances: (num_clusters,) Balance with each cluster
    :return avg_balance: Average balance of clusters
    """

    # Get memberships
    cluster_memberships = get_membership_matrix(clusters)
    group_memberships = get_membership_matrix(groups)
    num_clusters = cluster_memberships.shape[1]
    num_groups = group_memberships.shape[1]

    # Count number of elements in each cluster
    group_sizes = group_memberships.sum(axis=0).reshape((-1,))

    # Compute balance
    counts = np.matmul(cluster_memberships.T, group_memberships)
    balances = np.zeros((num_clusters,))
    for c in range(num_clusters):
        balance = float('inf')
        for g1 in range(num_groups):
            for g2 in range(num_groups):
                curr_balance = counts[c, g1] / (1e-6 + counts[c, g2])
                if normalize:
                    curr_balance = curr_balance * (group_sizes[g2] / (1e-6 + group_sizes[g1]))
                if curr_balance < balance:
                    balance = curr_balance
        balances[c] = balance

    return balances, balances.mean()


def reflow_clusters0(clusters: np.ndarray):
    """
    :param clusters: (num_nodes,) Cluster assignment
    :return reflow: (num_nodes,) Cluster number changed so that they are contiguous. If no point belongs to cluster 3
                    then cluster 4 will become cluster 3, cluster 5 will become cluster 4, and so on.
    """
    id_map = dict()
    idx = 0
    reflow = np.zeros(clusters.shape)
    for i in range(clusters.shape[0]):
        if clusters[i] not in id_map:
            id_map[clusters[i]] = idx
            idx += 1
        reflow[i] = id_map[clusters[i]]
    return reflow
    

def reflow_clusters(y):
    
    y=torch.tensor(y).type(torch.long)

    oh = F.one_hot(y)
    idx = torch.unique(y, sorted=False).type(torch.long)
    reflow = torch.argmax(oh[:,idx],dim=1)
    return reflow


def align_clusters(true_clusters: np.ndarray, pred_clusters: np.ndarray) -> (int, np.ndarray, np.ndarray):
    """
    :param true_clusters: (num_nodes,) Ground truth clusters
    :param pred_clusters: (num_nodes,) Predicted clusters
    :return num_mistakes: Number of mistakes incurred by the best alignment
    :return reflow_true: (num_nodes,) True clusters reflowed
    :return aligned_pred: (num_nodes,) Aligned predicted clusters
    """
    # Reflow clusters
    reflow_true = reflow_clusters0(true_clusters)
    reflow_pred = reflow_clusters0(pred_clusters)
    num_clusters_true = int(np.max(reflow_true) + 1)
    num_clusters_pred = int(np.max(reflow_pred) + 1)

    assert num_clusters_true == num_clusters_pred, 'Required num_clusters_true == num_clusters_pred.'

    # Find alignment with minimum error
    perms = set(permutations(list(range(num_clusters_true))))
    aligned_clusters = np.zeros(pred_clusters.shape)
    min_mistakes = float('inf')
    for perm in perms:
        temp_clusters = np.zeros(pred_clusters.shape)
        for i in range(pred_clusters.shape[0]):
            temp_clusters[i] = perm[pred_clusters[i]]

        mistakes = (temp_clusters != reflow_true).sum()
        if mistakes < min_mistakes:
            min_mistakes = mistakes
            aligned_clusters = temp_clusters

    return min_mistakes, reflow_true, aligned_clusters

## ------------------------------------------- Clustering metrics ----------------------------------------

def lab2com0(y):
    eths = np.unique(y)

    coms = []
    com = set()
    for i in range(eths.shape[0]):
        com0 = np.where(y == eths[i])[0]
        for j in range(com0.shape[0]):
            com.add(com0[j])
        coms.append(com)
        com = set()

    return coms
    
def lab2com(y):
    eths = torch.unique(y)
    coms = [set(torch.where(y == eths[i])[0].cpu().numpy()) for i in range(eths.shape[0])]
    return coms
