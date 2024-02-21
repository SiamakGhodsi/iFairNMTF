import torch
import numpy as np
import os
import scipy as sp
import copy
from scipy.spatial import distance_matrix

def similarity_constraint(A, F, k):
    z = F.shape[1]
    n = A.shape[0]
    D = distance_matrix(F,F)
    R = D.copy()

    for i in range(n):
        R[i, :][R[i, :].argsort()[0:n - k]] = 0
        R[:, i][R[:, i].argsort()[0:n - k]] = 0
    return R

def nullspace(At, rcond=None):
    """A = At.numpy()
    ns = sp.linalg.null_space(A, rcond=None)"""

    # return the nullspace of matrix F to constitute Z
    ut, st, vht = torch.Tensor.svd(At, some=False,compute_uv=True)
    vht=vht.T
    Mt, Nt = ut.shape[0], vht.shape[1]
    if rcond is None:
        rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
    tolt = torch.max(st) * rcondt
    numt= torch.sum(st > tolt, dtype=int)
    ns = vht[numt:,:].T.conj()
    return ns

def compute_F(sensitive):
    """
    :param groups: (num_nodes,) Vector indicating protected group memberships
    :return F: Group_fairness constraint matrix as in Kleindesnner
    """
    n = len(sensitive)
    # converting sensitive to a vector with entries in [h] and building F
    sens_unique = np.unique(sensitive)
    h = len(sens_unique)
    #sens_unique = reshape(sens_unique, [1, h]);

    sensitiveNEW = copy.deepcopy(sensitive)

    temp = 0;
    for i in sens_unique:
        ind = np.where(np.isin(sensitive, i))
        sensitiveNEW[ind] = temp
        temp = temp + 1;

    F = np.zeros((n, h - 1));

    for ell in range(h - 1):
        temp = np.where(np.isin(sensitiveNEW, ell))
        F[temp[0], ell] = 1;
        groupSize = len(temp[0]);
        F[:, ell] = F[:, ell] - groupSize/n;

    return F

def compute_R0(sensitive):
    """
    :param sensitive: (num_nodes,) Vector indicating protected group memberships
    :return R: nxn Representation graph constraint
    """
    n = len(sensitive)
    # counting number of protected groups
    sens_unique = np.unique(sensitive)
    h = len(sens_unique)

    group_one_hot = np.eye(h)[sensitive, :]
    similarity_matrix = np.matmul(group_one_hot, group_one_hot.T)
    diag = np.eye((n))

    R = similarity_matrix - diag

    return R

def compute_RS(sensitive):
    """
    :param sensitive: (num_nodes,) Vector indicating protected group memberships
    :return R: nxn Representation graph constraint
    """
    n = len(sensitive)
    # counting number of protected groups
    sens_unique = np.unique(sensitive)
    h = len(sens_unique)

    group_one_hot = np.eye(h)[sensitive, :]
    similarity_matrix = np.matmul(group_one_hot, group_one_hot.T)
    diag = np.eye((n))

    R = similarity_matrix - diag
    R_normal = R/R.sum(axis=1, keepdims=True)

    return R_normal

def compute_RD(sensitive):
    """
    :param sensitive: (num_nodes,) Vector indicating protected group memberships
    :return R: nxn Representation graph constraint
    """
    n = len(sensitive)
    # counting number of protected groups
    sens_unique = np.unique(sensitive)
    h = len(sens_unique)

    group_one_hot = np.eye(h)[sensitive, :]
    R = 1- np.matmul(group_one_hot, group_one_hot.T)
    #diag = np.eye((n))

    #R = similarity_matrix - diag
    R_normal = R/R.sum(axis=1, keepdims=True)

    return R_normal

def svd_init(Adj, k):
    #H = torch.rand(n, k, dtype=torch.float)
    u, s, v = torch.svd(Adj)
    W = torch.diag(s[:k] + torch.tensor(0.1))
    return W

def joint_Laplacian(groups):
    """
    :param groups: (num_nodes,) Vector indicating protected group memberships
    :return L: nxn Representation graph constraint
    """

    RS = compute_RS(groups)
    RD = compute_RD(groups)

    R1D = torch.tensor(RD, dtype=torch.float)
    R1S = torch.tensor(RS, dtype=torch.float)
    R = R1D - R1S
    L = torch.diag(torch.sum(R, dim=1)) - R
    Lp = (torch.abs(L) + L) / 2
    Ln = (torch.abs(L) - L) / 2
    return L, Ln, Lp