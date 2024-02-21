import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from utils import *
from evaluations import *
import os 
import time
from algorithms import *
#from kmeans_pytorch import kmeans
import networkx as nx
import matplotlib.pyplot as plt
import copy

eps = torch.tensor(0.000001)

## GPU or CPU
GPU = True
if GPU:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("num GPUs", torch.cuda.device_count())
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
    print("CPU")

# ------------------------------------------------ Diaries ---------------------------------------------

path1 = "data/School/"
path2 = "results/New/School/"

A = (pd.read_csv(path1 + "diaries.csv", header=None)).to_numpy()
F = (pd.read_csv(path1 + "cd_attr.csv", header=None)).to_numpy()

all_in_one = np.ones(F.shape[0])
uniqe_vals, count = np.unique(F, return_counts=True)
Diaries_balance = min(count)/max(count)

print(len(F), len(A))
print("Dataset balance = ", Diaries_balance)

# -------------------------------------------------- grid-search ---------------------------------------

k = list(range(2, 15))
#lambdas = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10, 100]
lambdas = [0, 0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8,
           1, 1.2, 1.4, 1.5, 1.6, 1.8, 2, 2.5, 3 ,3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 15,
           20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]


df1 = pd.DataFrame(columns=['network','attr','method','lam','nodes (n)',
                            'clusters (k)', 'groups (h)', 'Q_FNMTF', 'B_FNMTF'])
n = A.shape[0]

Adj = torch.tensor(A, dtype=torch.float)
standard_groups = reflow_clusters(F.ravel())
groups = np.transpose(standard_groups)
L, Ln, Lp = joint_Laplacian(groups)

G1 = nx.from_numpy_array(A, create_using=nx.Graph, parallel_edges=False)

for num_c in k:
    for lam in lambdas:
        iter = 500
        num_repeats = 5

        Q_iter, B_iter= ([] for i in range(2))
        for param in range(num_repeats):

            # iFairNMTF
            H1, W1, err1 = iFairNMTF(Adj, L, Ln, Lp, num_c, lam, iter)

            # transform fuzzy memberships (overlaps) to strict (disjoint)
            predfnmtf = torch.argmax(H1, dim=1)

            pred_FNMTF = reflow_clusters(predfnmtf.numpy())

            # calculate balance
            balances_FNMTF, B1 = compute_group_balance(pred_FNMTF.numpy(), groups.numpy())

            # convert predicted labels to networkx clusters
            coms_FNMTF = lab2com(pred_FNMTF)  # lab2com0(pred_FNMTF.numpy())

            # compute modularity
            Q1 = nx.community.modularity(G1, coms_FNMTF)

            Q_iter.append(Q1); B_iter.append(B1);

        Q_FNMTF = sum(Q_iter) / num_repeats
        B_FNMTF = sum(B_iter) / num_repeats

        # row of results to be appended to df
        results = dict()
        results = {'network': "Diaries", 'attr': "Gender", 'method': "FairNMF", 'lam': lam, 'nodes (n)': n,
                   'clusters (k)': num_c, 'groups (h)': len(np.unique(groups)), 'Q_FNMTF': Q_FNMTF,
                   'B_FNMTF': B_FNMTF}
        temp = pd.DataFrame(pd.Series(results)).transpose()
        df1 = pd.concat([df1, temp], ignore_index=True)

df1.to_csv('Diaries_k_lam_gridsearch_FNMTF.csv', index=False)
print (df1)

# --------------------------------------------- Diaries comparisons -------------------------------------------
"""
k = list(range(2, 15))

df = pd.DataFrame(columns=['network', 'attr', 'method', 'nodes (n)', 'clusters (k)', 'groups (h)', 'modularity',
                           'average balance']) #, 'cluster balances'])
n = A.shape[0]

Adj = torch.tensor(A, dtype=torch.float)
standard_groups = reflow_clusters(F.ravel())
groups = np.transpose(standard_groups)
#F_gen = compute_F(groups)

G1 = nx.from_numpy_array(A, create_using=nx.Graph, parallel_edges=False)

for num_c in k:

    num_repeats = 3
    Q_iter_ifsc, B_iter_ifsc, Q_iter_fsc, B_iter_fsc, Q_iter_sc, B_iter_sc = ([] for ii in range(6))

    for param in range(num_repeats):
        # Individual Fair_SC
        predifsc = ind_fair_sc(A, groups, num_c)
        pred_IFSC = reflow_clusters(predifsc)
        # calculate balance
        balances_IFSC, Bifsc = compute_group_balance(pred_IFSC.numpy(), groups.numpy())
        # convert predicted labels to networkx clusters
        coms_IFSC = lab2com0(pred_IFSC)
        # compute modularity
        Qifsc = nx.community.modularity(G1, coms_IFSC)

        Q_iter_ifsc.append(Qifsc)
        B_iter_ifsc.append(Bifsc)

        # Fair_SC
        predfsc = group_fair_sc(A, groups, num_c)
        pred_FSC = reflow_clusters(predfsc)
        # calculate balance
        balances_FSC, Bfsc = compute_group_balance(pred_FSC.numpy(), groups.numpy())
        # convert predicted labels to networkx clusters
        coms_FSC = lab2com0(pred_FSC)
        # compute modularity
        Qfsc = nx.community.modularity(G1, coms_FSC)

        Q_iter_fsc.append(Qfsc)
        B_iter_fsc.append(Bfsc)

        # SC
        predsc = normal_sc(A, num_c)
        pred_SC = reflow_clusters(predsc)
        # calculate balance
        balances_SC, Bsc = compute_group_balance(pred_SC.numpy(), groups.numpy())
        # convert predicted labels to networkx clusters
        coms_SC = lab2com0(pred_SC)
        # compute modularity
        Qsc = nx.community.modularity(G1, coms_SC)

        Q_iter_sc.append(Qsc)
        B_iter_sc.append(Bsc)

        Q_iter_fsc.append(0)
        B_iter_fsc.append(0)
        Q_iter_sc.append(0)
        B_iter_sc.append(0)

    Q_IFSC = sum(Q_iter_ifsc) / num_repeats
    avg_balance_IFSC = sum(B_iter_ifsc) / num_repeats

    Q_FSC = sum(Q_iter_fsc) / num_repeats
    avg_balance_FSC = sum(B_iter_fsc) / num_repeats

    Q_SC = sum(Q_iter_sc) / num_repeats
    avg_balance_SC = sum(B_iter_sc) / num_repeats

    # row of results to be appended to df
    col1 = ["Diaries" for i in range(3)]
    col2 = ["Gender" for i in range(3)]
    col3 = ["ifair_sc", "fair_sc", "vanilla_sc"]
    col4 = [n for i in range(3)]
    col5 = [num_c for i in range(3)]
    col6 = [len(np.unique(groups)) for i in range(3)]
    col7 = [Q_IFSC, Q_FSC, Q_SC]
    col8 = [avg_balance_IFSC, avg_balance_FSC, avg_balance_SC]
    #col9 = [balances_FSC, balances_SC]
    results = dict()
    results = {'network': col1, 'attr': col2, 'method': col3, 'nodes (n)': col4, 'clusters (k)': col5,
               'groups (h)': col6, 'modularity': col7, 'average balance': col8} #, 'cluster balances': col9}
    temp = pd.DataFrame((results))
    df = pd.concat([df,temp], ignore_index=True)
    print(temp)

print(df)
df.to_csv('Diaries.csv', index=False)
"""