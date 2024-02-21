import numpy as np
from scipy.linalg import null_space, eigh, sqrtm
from sklearn.cluster import k_means


def kmeans(features: np.ndarray, num_clusters: int, normalize_rows: bool = False) -> np.ndarray:
    """
    :param features: (num_nodes, num_features) Feature matrix
    :param num_clusters: Number of clusters to discover
    :param normalize_rows: Whether to normalize rows of feature matrix or not
    :return clusters: (num_nodes,) Discovered clusters
    """
    if normalize_rows:
        features = features / np.linalg.norm(features, axis=1).reshape((-1, 1))
    _, clusters, _ = k_means(features, num_clusters)
    return clusters


def compute_top_eigen(mat: np.ndarray, k: int) -> np.ndarray:
    """
    :param mat: (num_nodes, num_nodes) Symmetric matrix for which eigenvectors must be computed
    :param k: Number of clusters to discover
    :return evec: (num_nodes, k) Top k eigenvectors of mat for the smallest k eigenvalues
    """
    assert mat.shape[0] >= k, 'Insufficient number of eigenvectors'
    _, vec = eigh(mat, subset_by_index=[0, k - 1])
    return vec


def compute_laplacian(adj_mat: np.ndarray, normalize_laplacian: bool = False) -> np.ndarray:
    """
    :param adj_mat: (num_nodes, num_nodes) Adjacency matrix of the observed graph
    :param normalize_laplacian: Whether to use normalized Laplacian or not
    :return laplacian: (num_nodes, num_nodes) The laplacian of the graph
    """
    degree_mat = np.diag(np.sum(adj_mat, axis=1))
    laplacian = degree_mat - adj_mat
    if normalize_laplacian:
        degree_mat_inv = np.sqrt(np.linalg.inv(1e-6 * np.eye(adj_mat.shape[0]) + degree_mat))
        laplacian = np.matmul(degree_mat_inv, np.matmul(laplacian, degree_mat_inv))
    return laplacian



