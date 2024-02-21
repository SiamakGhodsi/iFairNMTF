import torch
import numpy as np
import scipy as sp
from SC_helpers import kmeans, compute_top_eigen, compute_laplacian
from scipy.linalg import null_space, eigh, sqrtm
from scipy.sparse.linalg import spsolve
from utils import *
import copy

eps = torch.tensor(1.0e-10)

def iFairNMTF(A: torch.tensor, L: torch.tensor, Ln: torch.tensor, Lp: torch.tensor, k, lam, iter=500):
    
    """
    :param A: (num_nodes, num_nodes) Adjacency matrix of the observed graph
    :param L: (num_nodes, num_nodes) Laplacian of the contrastive individual fairness constraint i.e. representation graph constraint
    :param Ln: (num_nodes, num_nodes) Negative elements of laplacian used in Eq.(13). of the main paper 
    :param Lp: (num_nodes, num_nodes) Positive elements of laplacian used in Eq.(13). of the main paper
    :param k: required number of clusters
    :return clusters: (num_nodes, k) The cluster assignment for each node
    """

    n = A.shape[0]

    #RTR = R.T @ R
    H = torch.rand(n, k, dtype=torch.float)  # the membership degree matrix (H) initialization

    # Co-cluster factor initialization by eigen_values of the Adjacency matrix
    # W = torch.rand(k, k, dtype=torch.float)
    W = svd_init(A, k)

    #L, Ln, Lp = joint_Laplacian(groups) # for optimized computations this is moved to experiments

    err = torch.zeros(iter)

    for t in range(iter):
        Hn = (A.T @ H @ W + A @ H @ W.T + lam * (Ln @ H))
        Hd = H @ W.T @ H.T @ H @ W + H @ W @ H.T @ H @ W.T + lam * ((Lp) @ H)#(RTR @ H)
        H = H * (Hn / torch.maximum(Hd, eps)) ** 0.25

        Wn = H.T @ A @ H
        Wd = H.T @ H @ W @ H.T @ H
        W = W * (Wn / torch.maximum(Wd, eps))

        err[t] = torch.norm(A - H @ W @ H.T) ** 2 + lam * torch.trace(H.T @ L @ H)

    import matplotlib.pyplot as plt
    plt.plot(err)

    return H, W, err


def ind_fair_sc(A: np.ndarray, groups: np.ndarray, k: int,
                normalize_laplacian: bool = False, normalize_evec: bool = False) -> np.ndarray:
    """
    :param A: (num_nodes, num_nodes) Adjacency matrix of the observed graph
    :param groups: (num_nodes,) An array indicating sensitive group membership
    :param k: Number of clusters to discover
    :param normalize_laplacian: Whether to use normalized Laplacian or not
    :param normalize_evec: Whether to normalize the rows of eigenvector matrix before running k-means
    :return clusters: (num_nodes,) The cluster assignment for each node
    """

    # Compute the constraint matrix
    R = compute_RD(groups)         # Representation_graph adjacency mat: A (n x n) graph specifying
                                   # which node can represent which other nodes
    Z = null_space(R)              # null_space_basis

    #R = compute_R0(groups)
    #ones = np.ones(A.shape)
    #c_mat = np.matmul(R, np.eye(A.shape[0]) - ones / A.shape[0])
    #Z = null_space(c_mat) # null_space_basis
    assert Z.shape[1] >= k, 'Rank of c_mat is too high'

    # Compute the Laplacian
    L = compute_laplacian(A, normalize_laplacian=False)
    if normalize_laplacian:
        D = np.diag(A.sum(axis=1))
        Q = np.real(sqrtm(np.matmul(np.matmul(Z.T, D), Z)))
        Q_inv = np.linalg.inv(1e-6 * np.eye(Q.shape[0]) + Q)
        Z = np.matmul(Z, Q_inv)
    LL = np.matmul(Z.T, np.matmul(L, Z))

    # Compute eigenvectors
    Y = compute_top_eigen(LL, k)
    YY = np.matmul(Z, Y)

    # Run k-means
    clusters = kmeans(YY, k, normalize_evec)

    return clusters

def group_fair_sc(A: np.ndarray, groups: np.ndarray, k: int,
                normalize_laplacian: bool = False, normalize_evec: bool = False) -> np.ndarray:
    """
    :param A: (num_nodes, num_nodes) Adjacency matrix of the observed graph
    :param groups: (num_nodes,) An array indicating sensitive group membership
    :param k: Number of clusters to discover
    :param normalize_laplacian: Whether to use normalized Laplacian or not
    :param normalize_evec: Whether to normalize the rows of eigenvector matrix before running k-means
    :return clusters: (num_nodes,) The cluster assignment for each node
    """

    # Compute the constraint matrix
    Fair_Mat = compute_F(groups) # An (n x g) float matrix of size h-1 specifying membership to each
                            # protected group where h is the number of protected groups
    Z = null_space(Fair_Mat.T)  # null_space_basis
    assert Z.shape[1] >= k, 'Rank of c_mat is too high'

    # Compute the Laplacian
    L = compute_laplacian(A, normalize_laplacian=False)
    if normalize_laplacian:
        D = np.diag(A.sum(axis=1))
        Q = np.real(sqrtm(np.matmul(np.matmul(Z.T, D), Z)))
        Q_inv = np.linalg.inv(1e-6 * np.eye(Q.shape[0]) + Q)
        Z = np.matmul(Z, Q_inv)

    LL = np.matmul(Z.T, np.matmul(L, Z))
    LL = (LL + LL.T)/2

    # Compute eigenvectors
    Y = compute_top_eigen(LL, k)
    YY = np.matmul(Z, Y)

    # Run k-means
    clusters = kmeans(YY, k, normalize_evec)

    return clusters

def normal_sc(adj_mat: np.ndarray, num_clusters: int, normalize_laplacian: bool = False, normalize_evec: bool = False) \
        -> np.ndarray:
    """
    :param adj_mat: (num_nodes, num_nodes) Adjacency matrix of the observed graph
    :param num_clusters: Number of clusters to discover
    :param normalize_laplacian: Whether to use normalized Laplacian or not
    :param normalize_evec: Whether to normalize the rows of eigenvector matrix before running k-means
    :return clusters: (num_nodes,) The cluster assignment for each node
    """

    # Compute the Laplacian
    laplacian = compute_laplacian(adj_mat, normalize_laplacian)

    # Compute eigenvectors
    vec = compute_top_eigen(laplacian, num_clusters)

    # Run k-means
    clusters = kmeans(vec, num_clusters, normalize_evec)

    return clusters

