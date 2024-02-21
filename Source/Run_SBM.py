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
from generators import *
import sFSC
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

# --------------------------------------------- conditioned SBM --------------------------------------------------
# parameters
n = [2000, 5000, 10000]
k = [5]
h = 5
lambdas = [0, 0.001, 0.1, 0.5, 1, 2, 3, 5, 8, 10, 20, 50, 100, 1000]

df_FNMTF = pd.DataFrame(columns=['network','attr','method','nodes (n)','clusters (k)','lam',
                                'groups (h)','accurcay','average balance'])

for nodes in n:
    # calculate logarithmic probablity of edges a,b,c,d based on size of n
    logarithmic = (np.log(nodes)/nodes)**(2/3)
    multiplier = [10, 7, 4, 1]
    a = multiplier[0]* logarithmic
    b = multiplier[1]* logarithmic
    c = multiplier[2]* logarithmic
    d = multiplier[3]* logarithmic

    iter = 500
    for num_c in k:
        for lam in lambdas:

            adj_mat, fair_mat, clusters, groups = gen_kleindessner(nodes, num_c, h, a, b, c, d)
            Adj = torch.tensor(adj_mat, dtype=torch.float)
            L, Ln, Lp = joint_Laplacian(groups)
            num_repeats = 1
            Acc_iter_fnmtf, B_iter_fnmtf = [], []

            for param in range(num_repeats):

                # iFairNMTF
                H1, W1, err1 = iFairNMTF(Adj, L, Ln, Lp, num_c, lam, iter)

                # transform fuzzy memberships (overlaps) to strict (disjoint)
                predfnmtf = torch.argmax(H1, dim=1)

                # convert predicted labels to clusters
                min_mistakes, reflow_true, aligned_clusters = align_clusters(clusters, predfnmtf.numpy())
                # calculate balance and error
                _, B = compute_group_balance(aligned_clusters.astype(int), groups)

                error = min_mistakes/nodes
                acc= 1 - error
                Acc_iter_fnmtf.append(acc)
                B_iter_fnmtf.append(B)

            Acc_FNMTF = sum(Acc_iter_fnmtf) / num_repeats
            B_FNMTF = sum(B_iter_fnmtf) / num_repeats

            # row of results to be appended to df
            results = dict()
            results = {'network': "SBM", 'attr': "distributed", 'method': "ifair_nmtf", 'nodes (n)': nodes,
            'clusters (k)': num_c, 'lam': lam, 'groups (h)': h, 'accurcay': Acc_FNMTF, 'average balance': B_FNMTF}

            # FNMF results being saved
            temp = pd.DataFrame(pd.Series(results)).transpose()
            df_FNMTF = pd.concat([df_FNMTF,temp], ignore_index=True)
            # display(temp)
            print(temp)

#display(df_FNMF)
print(df_FNMTF)
df_FNMTF.to_csv('Fair_NMTF_SBM_k_lam_gridsearch' + '.csv', index=False)

# ---------------------------------------- conditioned SBM comparisons----------------------------------------------

df_comp = pd.DataFrame(
    columns=['network', 'attr', 'method', 'nodes (n)', 'clusters (k)', 'groups (h)', 'a = 0.7 log(n)/n**2/3$',
             'accurcay', 'average balance'])

for nodes in n:
    # calculate logarithmic probablity of edges a,b,c,d based on size of n
    logarithmic = (np.log(nodes) / nodes) ** (2 / 3)
    multiplier = [10, 7, 4, 1]
    a = multiplier[0] * logarithmic
    b = multiplier[1] * logarithmic
    c = multiplier[2] * logarithmic
    d = multiplier[3] * logarithmic

    iter = 500
    for num_c in k:
        adj_mat, fair_mat, clusters, groups = gen_kleindessner(nodes, num_c, h, a, b, c, d)
        # F = compute_F(groups)
        num_repeats = 3

        Acc_iter_sfsc, B_iter_sfsc = [], []
        Acc_iter_ifsc, B_iter_ifsc = [], []
        Acc_iter_fsc, B_iter_fsc = [], []
        Acc_iter_sc, B_iter_sc = [], []

        for param in range(num_repeats):
            """
            #-------------------- FairSC -------------------
            predFSC = group_fair_sc(adj_mat, groups, num_c)
            # convert predicted labels to clusters
            min_mistakes, reflow_true, aligned_clusters = align_clusters(clusters, predFSC)
            # calculate balance and error
            _, B = compute_group_balance(aligned_clusters.astype(int), groups)

            error = min_mistakes/nodes
            acc= 1 - error
            Acc_iter_fsc.append(acc)
            B_iter_fsc.append(B)

            #-------------------- SC ---------------------
            predSC = normal_sc(adj_mat, num_c)
            # convert predicted labels to clusters
            min_mistakes, reflow_true, aligned_clusters = align_clusters(clusters, predSC)
            # calculate balance and error
            _, B = compute_group_balance(aligned_clusters.astype(int), groups)

            error = min_mistakes/nodes
            acc= 1 - error
            Acc_iter_sc.append(acc)
            B_iter_sc.append(B)

            # -------------------- sFSC ---------------------
            FF = compute_F(groups)
            sfsc = sFSC.initialize()
            D = np.diag(adj_mat.sum(axis=1))

            pred_sfsc = sfsc.sFSC(adj_mat.astype(float), D.astype(float), FF.astype(float), 3)
            pred_np = np.ravel(pred_sfsc).astype(int)

            min_mistakes, reflow_true, aligned_clusters = align_clusters(clusters, pred_np)
            # calculate balance and error
            _, B = compute_group_balance(aligned_clusters.astype(int), groups)

            error = min_mistakes / nodes
            acc = 1 - error
            Acc_iter_sfsc.append(acc)
            B_iter_sfsc.append(B)
            """
            # -------------------- iFSC ---------------------
            prediFSC = ind_fair_sc(adj_mat, groups, num_c)
            # convert predicted labels to clusters
            min_mistakes, reflow_true, aligned_clusters = align_clusters(clusters, prediFSC)
            # calculate balance and error
            _, B = compute_group_balance(aligned_clusters.astype(int), groups)

            error = min_mistakes / nodes
            acc = 1 - error
            Acc_iter_ifsc.append(acc)
            B_iter_ifsc.append(B)

        Acc_iFSC = sum(Acc_iter_ifsc) / num_repeats
        B_iFSC = sum(B_iter_ifsc) / num_repeats
        Acc_sFSC = sum(Acc_iter_sfsc) / num_repeats
        B_sFSC = sum(B_iter_sfsc) / num_repeats
        Acc_FSC = sum(Acc_iter_fsc) / num_repeats
        B_FSC = sum(B_iter_fsc) / num_repeats
        Acc_SC = sum(Acc_iter_sc) / num_repeats
        B_SC = sum(B_iter_sc) / num_repeats

        Acc_FSC = 0;
        B_FSC = 0;
        Acc_SC = 0;
        B_SC = 0;
        B_iFSC = 0;
        Acc_sFSC = 0

        # competitor results being saved
        # row of results to be appended to a df
        col1 = ["SBM" for i in range(6)]
        col2 = ["distributed" for i in range(6)]
        col3 = ["ifair_nmtf", "vanilla_sc", "fair_sc", "sfsc", "ifsc", "dmon"]
        col4 = [nodes for i in range(6)]
        col5 = [num_c for i in range(6)]
        col6 = [h for i in range(6)]
        col7 = [a for i in range(6)]
        col8 = [0, Acc_SC, Acc_FSC, Acc_sFSC, Acc_iFSC, 0]
        col9 = [0, B_SC, B_FSC, B_sFSC, B_iFSC, 0]

        results = dict()
        results = {'network': col1, 'attr': col2, 'method': col3, 'nodes (n)': col4, 'clusters (k)': col5,
                   'groups (h)': col6, 'a = 0.7 log(n)/n**2/3$': col7, 'accurcay': col8, 'average balance': col9}
        temp = pd.DataFrame((results))
        df_comp = pd.concat([df_comp, temp], ignore_index=True)
        # display(temp)

# display(df_comp)
print(df_comp)
df_comp.to_csv('SBM_k_n_gridsearch' + '.csv', index=False)
