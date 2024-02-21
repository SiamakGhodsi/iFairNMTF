import numpy as np
from itertools import permutations
import torch.nn.functional as F
import torch
import math

def IGDC(a: np.ndarray, b:np.ndarray, s: str):
    igdc = 0
    if a.ndim == 1:
        igdc = math.dist(a, b)
    else:
        a = a.transpose()
        if (s=='avg'):
            cumul = 0
            for j in a:
                cumul+= math.dist(j,b)
            igdc = cumul/np.shape(a)[0]
        elif(s=='min'):
            dis = []
            for j in a:
                dis.append(math.dist(j,b))
            igdc = min(dis)
    return igdc

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


def compute_individual_balance(clusters: np.ndarray, fair_mat: np.ndarray, normalize: bool = False) -> \
        (np.ndarray, float):
    """
    :param clusters: (num_nodes,) Predicted clusters
    :param fair_mat: (num_nodes, num_nodes) Fairness graph under which balance must be computed
    :param normalize: Whether to normalize balance using cluster sizes
    :return balances: (num_nodes,) Balance for each individual
    :return avg_balance: Average balance of the individuals
    """

    # Get memberships
    cluster_memberships = get_membership_matrix(clusters)
    num_clusters = cluster_memberships.shape[1]
    num_nodes = fair_mat.shape[0]

    # Compute cluster sizes
    cluster_sizes = cluster_memberships.sum(axis=0).reshape((-1,))

    # Compute balance
    counts = np.matmul(fair_mat, cluster_memberships)
    balances = np.zeros((num_nodes,))
    for i in range(num_nodes):
        balance = float('inf')
        for c1 in range(num_clusters):
            for c2 in range(num_clusters):
                curr_balance = counts[i, c1] / (1e-6 + counts[i, c2])
                if normalize:
                    curr_balance = curr_balance * (cluster_sizes[c2] / (1e-6 + cluster_sizes[c1]))
                if curr_balance < balance:
                    balance = curr_balance
        balances[i] = balance

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

def modularity_density(adj, c, dict_vec=None):
    """Determines modularity_density of a set of communities using a metric
        that is free from bias and faster to compute.
        Parameters
        ----------
        adj : SciPy sparse matrix (csr or csc)
            The N x N Adjacency matrix of the graph of interest.
        c : Integer array
            Current array of community labels for the nodes in the graph as ordered
            by the adjacency matrix.
        dict_vec : dictionary, optional
            Tracks the nodes in each community, with cluster labels as dictionary-
            keys, and the corresponding boolean arrays (c == label) as values.
        Returns
        -------
        float
            Determines modularity_density of a set of communities
            in 'cluster_labels'.
        ------
        Modularity density in [1] is given as
        .. math::
           Q = \sum_{c \in C}\Bigg\{\frac{\sum_{i,j \in c}T_{ij}}{n_c}  - \sum_{c^{\prime} \in C-c}\Bigg( \frac{\sum_{{i \in c,}{j \in c^{\prime}}}T_{ij}}{\sqrt{n_c n_{c^{\prime}}}}\Bigg)\Bigg\}
           where:
           - each cluster ${c \in C}$ is represented by an indicator vector ${\vec{v}_c = [v{_{c_i}}] \in {\R}^{|V|} : v{_{c_i}}= 1}$ if ${i \in c}$, else $0$
           - \hat{n}_c = \frac{\vec{v}_c}{|\vec{v}_c|}
        References
        ----------
        .. [1] MULA S, VELTRI G. A new measure of modularity density for
               community detection. arXiv:1908.08452 2019.
        """
