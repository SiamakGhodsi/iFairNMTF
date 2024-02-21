import numpy as np

# -------------------------------------- generators -----------------------------------------------------

def sample_indep_edges(prob_mat: np.ndarray, diagonal: int = None) -> np.ndarray:
    """
    :param prob_mat: Matrix indicating the probability of edges
    :param diagonal: The element to put on diagonal. If None, then the diagonal will be sampled as usual
    :return sampled_mat: Symmetric matrix sampled using prob_mat
    """
    sampled_mat = np.random.random(prob_mat.shape) <= prob_mat
    diagonal_mat = np.diag(np.diag(sampled_mat)) if diagonal is None else diagonal * np.eye(sampled_mat.shape[0])
    sampled_mat = np.triu(sampled_mat, k=1)
    sampled_mat = sampled_mat + sampled_mat.T + diagonal_mat
    return sampled_mat


def conditioned_sbm(clusters: np.ndarray, fair_mat: np.ndarray, p: float = 0.2, q: float = 0.15, r: float = 0.1,
                    s: float = 0.05) -> np.ndarray:
    """
    :param clusters: (num_nodes,) Ground truth clusters
    :param fair_mat: (num_nodes, num_nodes) Binary symmetric fairness matrix
    :param p: Parameter used by SBM
    :param q: Parameter used by SBM
    :param r: Parameter used by SBM
    :param s: Parameter used by SBM
    :return prob_mat: (num_nodes, num_nodes) A matrix representing probability of each edge
    """
    num_clusters = np.max(clusters) + 1
    ones = np.ones(fair_mat.shape)
    comm_one_hot = np.eye(num_clusters)[clusters, :]
    same_comm = np.matmul(comm_one_hot, comm_one_hot.T)
    prob_mat = p * same_comm * fair_mat + q * (ones - same_comm) * fair_mat + r * same_comm * (ones - fair_mat) + \
               s * (ones - same_comm) * (ones - fair_mat)
    return prob_mat


def gen_random_reg_bipartite(d: int, n: int) -> np.ndarray:
    """
    :param d: Common degree of the nodes
    :param n: Number of nodes in each part of the bipartite graph
    :return graph: (n, n) One portion of the symmetric random regular bipartite graph between the nodes
    """
    # Initialize the graph
    graph = np.zeros((n, n))

    # Randomly permute indices on both sides
    indices_left = list(range(n))
    indices_right = list(range(n))
    np.random.shuffle(indices_left)
    np.random.shuffle(indices_right)

    # Populate the graph
    for i in range(n):
        for j in range(d):
            graph[indices_left[i], indices_right[(i + j) % n]] = 1

    return graph


def gen_kleindessner(num_nodes: int, num_clusters: int, num_groups: int, p: float = 0.2, q: float = 0.15,
                     r: float = 0.1, s: float = 0.05) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    :param num_nodes: Number of nodes in the graph
    :param num_clusters: Number of clusters in the graph
    :param num_groups: Number of protected groups
    :param p: Parameter used by SBM
    :param q: Parameter used by SBM
    :param r: Parameter used by SBM
    :param s: Parameter used by SBM
    :return adj_mat: (num_nodes, num_nodes) Binary symmetric adjacency matrix without self loops
    :return fair_mat: (num_nodes, num_nodes) Binary symmetric block-diagonal fairness matrix
    :return clusters: (num_nodes,) Ground truth clusters
    :return groups: (num_nodes,) Protected groups
    """
    assert p >= q >= r >= s, 'Need p >= q >= r >= s'
    assert num_nodes % (num_clusters * num_groups) == 0, '(num_clusters * num_groups) must divide num_nodes'

    # Assign the clusters
    clusters = np.concatenate([np.asarray([x] * (num_nodes // num_clusters)) for x in range(num_clusters)], axis=0)

    # Assign the groups
    groups = np.concatenate([np.concatenate([np.asarray([x] * (num_nodes // (num_clusters * num_groups)))
                                             for x in range(num_groups)], axis=0)
                             for _ in range(num_clusters)], axis=0)

    # Compute the fairness matrix
    group_one_hot = np.eye(num_groups)[groups, :]
    fair_mat = np.matmul(group_one_hot, group_one_hot.T)

    # Computed expected adjacency matrix
    expected_adj_mat = conditioned_sbm(clusters, fair_mat, p, q, r, s)

    # Sample the adjacency matrix
    adj_mat = sample_indep_edges(expected_adj_mat, diagonal=0)

    return adj_mat, fair_mat, clusters, groups