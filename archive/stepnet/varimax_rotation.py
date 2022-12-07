from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import tensorflow as tf

import getpass
ui = getpass.getuser()
if ui == 'laura':
    p = '/home/laura'
elif ui == 'lauradriscoll':
    p = '/Users/lauradriscoll/Documents'

net = 'stepnet'
PATH_YANGNET = os.path.join(p,'code/multitask-nets',net) 

sys.path.insert(0, PATH_YANGNET)
from network import Model

from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler

from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
import scipy
import pylab
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

def varimax(Phi, gamma = 1.0, q = 100, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        print(i)
        if d_old!=0 and d/d_old < 1 + tol: break
    return R, dot(Phi, R)

def rotate_D(m, method = 'ward', cel_max_d = 3.5, criterion = 'distance', n_comps = 20):

    print(m)

    lesion_folder = 'lesion_fps_hierarchical_'+method+'_'+criterion+'_max_d'+str(cel_max_d)
    save_dir = os.path.join(m,lesion_folder)
    cluster_var = np.load(os.path.join(save_dir,'cluster_var.npz'))

    D = cluster_var['D'].T
    feature_names_original = [cluster_var['tick_names'][s] for s in range(len(cluster_var['tick_names']))]
    components_set = {}

    #ALIGN INDICES
    feature_names_master = ['DNMS go1', 'DMS go1', 'RT Go go1', 'DMC go1', 'DNMC go1',
           'Dly Go go1', 'Dly Anti go1', 'Ctx Dly DM 2 go1',
           'Ctx Dly DM 1 go1', 'MultSen Dly DM go1', 'Dly DM 1 go1',
           'Dly DM 2 go1', 'RT Anti go1', 'Go go1', 'Anti go1', 'Anti stim1',
           'Dly Anti stim1', 'Dly DM 1 stim1', 'Go stim1', 'Dly Go stim1',
           'Dly DM 2 stim1', 'Ctx Dly DM 1 stim1', 'Ctx Dly DM 2 stim1',
           'MultSen Dly DM stim1', 'Dly DM 1 delay2', 'Dly DM 2 delay2',
           'Dly Anti delay1', 'Ctx Dly DM 2 delay2', 'Ctx Dly DM 1 delay2',
           'MultSen Dly DM delay2', 'Dly Go delay1', 'Ctx Dly DM 1 delay1',
           'MultSen Dly DM delay1', 'Dly DM 1 delay1', 'Dly DM 2 delay1',
           'Ctx Dly DM 2 delay1', 'DMS stim1', 'DNMS stim1', 'DMC stim1',
           'DNMC stim1', 'Dly DM 1 stim2', 'Dly DM 2 stim2',
           'Ctx Dly DM 1 stim2', 'Ctx Dly DM 2 stim2', 'MultSen Dly DM stim2',
           'DNMS delay1', 'DMS delay1', 'DMC delay1', 'DNMC delay1']
       

    feat_order = [feature_names_original.index(s) for i,s in enumerate(feature_names_master)]
    X = D[:,feat_order]
    feature_names = feature_names_master

    #SAVE original atlas
    fig = plt.figure(figsize=(23, 5))
    ax = plt.axes()
    plt.imshow(X.T, cmap="magma")
    plt.title('Before')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(list(feature_names),fontsize = 6)
    plt.xlabel('Units')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir,'dynamic_modules_atlas_BEFORE'+'.png'))

    #Get corr before correlation
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes()
    im = ax.imshow(np.corrcoef(X.T), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(list(feature_names), rotation=90,fontsize = 6)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(list(feature_names),fontsize = 7)
    plt.colorbar(im).ax.set_ylabel("$r$", rotation=0)
    ax.set_title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'atlas_corr_BEFORE'+'.png'))

    #Get unrotated factors
    methods = [('Unrotated PCA', PCA()),
               ('Unrotated FA', FactorAnalysis())]#,
               #('Varimax FA', FactorAnalysis(rotation='varimax'))]
    fig, axes = plt.subplots(ncols=len(methods), figsize=(10, 8))
    for ax, (met, fa) in zip(axes, methods):
        fa.set_params(n_components=n_comps)
        fa.fit(X)
        components = fa.components_.T
        components_set[met] = components
        vmax = np.abs(components).max()
        ax.imshow(components, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
        ax.set_yticks(np.arange(len(feature_names)))
        if ax.is_first_col():
            ax.set_yticklabels(feature_names)
        else:
            ax.set_yticklabels([])
        ax.set_title(str(met))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'unrotated_factors'+'.png'))

    R, X_v = varimax(X.T)

    #SAVE new atlas
    fig = plt.figure(figsize=(23, 5))
    ax = plt.axes()
    plt.imshow(X_v, cmap="magma")
    plt.title('After')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(list(feature_names),fontsize = 6)
    plt.xlabel('Units')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir,'dynamic_modules_atlas_AFTER'+'.png'))

    #SAVE new atlas rotated
    # method = 'ward'
    # cel_max_d = 3.5
    # criterion = 'distance'
    # n_comps = 20

    sparse_inds = np.sum(abs(X_v),axis = 0)>1e-3
    X_sparse = X_v[:,sparse_inds]

    # Compute and plot dendrogram.
    fig = plt.figure(figsize=(24, 15))
    axdendro = fig.add_axes([0.09,0.1,0.05,0.75])
    Y = sch.linkage(X_sparse, method=method)

    if criterion == 'maxclust':
        max_d = 14 #max number of task clusters
        clusters = fcluster(Y, max_d, criterion='maxclust') #CHANGE hard coded 14 clusters
    else:
        max_d = 5 #threshold for task clusters
        clusters = fcluster(Y, max_d, criterion='distance')

    Z = sch.dendrogram(Y, orientation='left',labels = feature_names_master,
                       leaf_font_size = 11,color_threshold=max_d)
    axdendro.set_xticks([])
    # axdendro.set_yticks([])
    axdendro.spines['top'].set_visible(False)
    axdendro.spines['right'].set_visible(False)
    axdendro.spines['bottom'].set_visible(False)
    axdendro.spines['left'].set_visible(False)

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.22,0.1,0.75,0.75])
    index_left = Z['leaves']
    tick_names_sorted = [feature_names_master[i] for i in index_left]
    X_sparse = X_sparse[index_left,:]

    # cel_num = [CA.ind_active[x] for x in index_top]
    axdendro_top = fig.add_axes([0.22,.9,0.75,0.1])
    Y = sch.linkage(X_sparse.T, method=method)

    if criterion== 'maxclust':
        clusters = fcluster(Y, cel_max_d, criterion='maxclust') #CHANGE hard coded 14 clusters
        Z = sch.dendrogram(Y, orientation='top',labels = clusters, #CA.ind_active #clusters
                       leaf_font_size = 11,color_threshold=0)

    else:
        clusters = fcluster(Y, cel_max_d, criterion='distance')
        Z = sch.dendrogram(Y, orientation='top',labels = clusters, #CA.ind_active #clusters
                       leaf_font_size = 11,color_threshold=cel_max_d)

    axdendro_top.set_yticks([])
    axdendro_top.spines['top'].set_visible(False)
    axdendro_top.spines['right'].set_visible(False)
    axdendro_top.spines['bottom'].set_visible(False)
    axdendro_top.spines['left'].set_visible(False)

    index_top = Z['leaves']
    X_sparse = X_sparse[:,index_top]
    clusters_sorted = clusters[index_top]
    im = axmatrix.matshow(X_sparse, aspect='auto', origin='lower',cmap='magma')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    # fig.colorbar(im,orientation='horizontal')
    plt.savefig(os.path.join(save_dir,'dendro_atlas_AFTER'+'.png'))

    #Get corr after rotation
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes()
    im = ax.imshow(np.corrcoef(X_sparse), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(list(feature_names), rotation=90,fontsize = 6)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(list(feature_names),fontsize = 7)
    plt.colorbar(im).ax.set_ylabel("$r$", rotation=0)
    ax.set_title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'atlas_corr_AFTER'+'.png'))

    #Get rotated factors
    methods = [('Rotated PCA', PCA()),
               ('Rotated FA', FactorAnalysis())]
    fig, axes = plt.subplots(ncols=len(methods), figsize=(10, 8))
    for ax, (method, fa) in zip(axes, methods):
        fa.set_params(n_components=n_comps)
        fa.fit(X_sparse.T)
        components = fa.components_.T
        components_set[method] = components
        vmax = np.abs(components).max()
        ax.imshow(components, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
        ax.set_yticks(np.arange(len(feature_names)))
        if ax.is_first_col():
            ax.set_yticklabels(feature_names)
        else:
            ax.set_yticklabels([])
        ax.set_title(str(method))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'rotated_factors'+'.png'))
    rotate_D_var = {'D':D,
                'R':R,
                'components_set':components_set,
                'D_rotated':X_v,
                'D_rotated_sparse':X_sparse,
                'feature_names_master':feature_names_master}
    np.savez(os.path.join(save_dir,'rotate_D_var.npz'),**rotate_D_var)
    print(save_dir)