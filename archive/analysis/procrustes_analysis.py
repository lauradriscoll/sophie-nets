from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import numpy as np
import numpy.random as npr
import tensorflow as tf
import sys
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

import numpy as np
from scipy import linalg as LA
import datetime
from scipy.linalg import orthogonal_procrustes
import json
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression


######## EDIT THIS #####
net = 'stepnet'
tasks = np.concatenate((range(6),range(11,20)))
act = 'softplus'
untrained = False
model_n = 0

if act == 'relu':
    dir_set = ['lowD/combos/most','lowD/grad_norm_l2h000001/most','lowD/grad_norm_l2001/most'] 
else:
    dir_set = ['crystals/softplus/no_reg','crystals/softplus/l2h00001','crystals/softplus/l2w0001']

dir_specific = dir_set[1]#'crystals/highd_inputs/all_rules_4/softplus/no_reg_tune_width/'#'crystals/softplus/no_noise' #dir_set[2]
titles = ['context','stim1','go1']
#########################

import getpass
ui = getpass.getuser()
if ui == 'laura':
    p = '/home/laura'
elif ui == 'lauradriscoll':
    p = '/Users/lauradriscoll/Documents'
PATH_YANGNET = os.path.join(p,'code/multitask-nets',net) 

sys.path.insert(0, PATH_YANGNET)
from task import generate_trials, rule_name, rule_index_map, rules_dict
from network import Model
import tools
from tools_lnd import gen_trials_from_model_dir, make_procrustes_mat_stim, make_procrustes_mat_mov, make_h_combined, name_best_ckpt, make_procrustes_mat_stim

fldr = os.path.join('procrustes_analysis/')
if not os.path.exists(fldr):
    os.makedirs(fldr)

display_names = ['DelayGo', 'ReactGo', 'MemoryGo', 'DelayAnti', 'ReactAnti', 'MemoryAnti',
              'Integration1', 'Integration2', 'CxtIntegration1', 'CxtIntegration2', 'MultiIntegration',
              'MemoryIntegration1', 'MemoryIntegration2', 'CxtMemoryInt1', 'CxtMemoryIntegration2', 'MultiMemoryIntegration',
              'MemoryMatchSample', 'MemoryMatchSampleNogo', 'MemoryCategoryGo', 'MemoryCategoryNoGo']

rule_set = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']


model_dir_all = os.path.join(p,'data/rnn/multitask/',dir_specific,str(model_n))
if untrained is True:
    ckpt_n_dir = os.path.join(model_dir_all,'ckpts/model.ckpt-1')
else:
    ckpt_n = name_best_ckpt(model_dir_all,'delaygo')
    ckpt_n_dir = os.path.join(model_dir_all,'ckpts/model.ckpt-' + str(int(ckpt_n)))

fig = plt.figure(figsize=(5, 5))
cmap=plt.get_cmap('Greys')
fname = os.path.join(model_dir_all, 'log.json')

with open(fname, 'r') as f:
    log_all = json.load(f)
for r in tasks:
    c = cmap(r/20)
    ax = fig.add_subplot(1,1,1)
    x = np.log(log_all['cost_'+rules_dict['all'][r]])
    plt.plot(x,'-',c = c)
    ax.set_xlabel('Training Step (x 1000)')
    ax.set_ylabel('Log Cost [for each task]')
    plt.ylim([-6,2])
plt.show()   

trial = gen_trials_from_model_dir(model_dir_all,'delaygo',mode='test')
n_total_trials = np.shape(trial.x)[1]# num trials in delaygo task
n_trials_use = n_total_trials
skip_trials = int(n_total_trials/n_trials_use)
#mod1 and mod2 are every other trial so we grab 8 trials from both groups spanning 360 stim angles
trial_set = np.concatenate((range(0,n_total_trials,skip_trials),range(1,n_total_trials,skip_trials)))

h_context_combined, h_stim_late_combined, h_stim_early_combined = make_h_combined(model_dir_all,ckpt_n_dir,tasks,trial_set)

# Distances across different tasks
dist = DistanceMetric.get_metric('euclidean')
X_euc_context = dist.pairwise(np.squeeze(h_context_combined))
procrust = {}
epoch = titles[1]
procrust[epoch] = make_procrustes_mat_stim(model_dir_all,ckpt_n_dir,epoch,tasks)
epoch = titles[2]
procrust[epoch] = make_procrustes_mat_mov(model_dir_all,ckpt_n_dir,epoch,tasks,10)


nr = len(procrust.keys())+2
nc = len(titles)
fig = plt.figure(figsize=(4*nr, 3*nc),tight_layout=True,facecolor='white')
plt.rcParams.update({'font.size': 16})
task_names_sorted = [rules_dict['all'][i] for i in tasks]

for k_ind in range(len(procrust.keys())):
    k = procrust.keys()[k_ind]

    ax = plt.subplot(nr,nc,1)
    plt.imshow(X_euc_context,cmap = 'gray')
    plt.title(titles[0]+' '+dir_specific+'_model_'+str(model_n))
    ax.set_xticks(range(0,len(tasks)))
    ax.set_yticks(range(0,len(tasks)))
    ax.set_yticklabels(task_names_sorted)
    fig.autofmt_xdate()
    ax.set_xticklabels(task_names_sorted)
    plt.colorbar()

    ax = plt.subplot(nr,nc,2)
    plt.imshow(procrust[titles[1]][k],cmap = 'gray')
    plt.title(titles[1]+' '+dir_specific+'_model_'+str(model_n))
    ax.set_xticks(range(0,len(tasks)))
    ax.set_yticks(range(0,len(tasks)))
    ax.set_yticklabels(task_names_sorted)
    fig.autofmt_xdate()
    ax.set_xticklabels(task_names_sorted)
    plt.colorbar()

    ax = plt.subplot(nr,nc,3)
    plt.imshow(procrust[titles[2]][k],cmap = 'gray')
    plt.title(titles[2]+' '+dir_specific+'_model_'+str(model_n))
    ax.set_xticks(range(0,len(tasks)))
    ax.set_yticks(range(0,len(tasks)))
    ax.set_yticklabels(task_names_sorted)
    fig.autofmt_xdate()
    ax.set_xticklabels(task_names_sorted)
    plt.colorbar()

fldr = os.path.join('procrustes_analysis/' + dir_specific +'/')
if not os.path.exists(fldr):
    os.makedirs(fldr)

cmap=plt.get_cmap('tab10')
nD = 10

embedding = MDS(n_components=2, dissimilarity='precomputed')
X_fix_transformed = embedding.fit_transform(X_euc_context)

embedding = MDS(n_components=2, dissimilarity='precomputed')
W = procrust[titles[1]]['Disparity']+procrust[titles[1]]['Disparity'].transpose()
X_stim_transformed = embedding.fit_transform(W/2)

embedding = MDS(n_components=2, dissimilarity='precomputed')
W = procrust[titles[2]]['Disparity']+ procrust[titles[2]]['Disparity'].transpose()
X_go_transformed = embedding.fit_transform(W/2)

ind = 0
for ind in range(len(tasks)):
    r = tasks[ind]
    if r == 0 or r == 3:
        c = cmap(0/10)
    elif r == 1 or r == 4:
        c = cmap(5/10)
    elif r == 11 or r == 12:
        c = cmap(4/10)
    elif r == 18 or r == 19:
        c = cmap(3/10)
    elif r == 16 or r == 17:
        c = cmap(2/10)
    else:
        c = cmap(1/10)
        
    ax = fig.add_subplot(nr,nc,1+nc)
    plt.plot(X_fix_transformed[ind,0],X_fix_transformed[ind,1],'o',c = c)
    plt.text(X_fix_transformed[ind,0],X_fix_transformed[ind,1],display_names[r],fontsize=16,fontweight='bold',
        bbox={'facecolor':c, 'alpha':0.5, 'pad':2, 'edgecolor':'none'})
    plt.title(titles[0])
    plt.axis('square')
    
    ax = fig.add_subplot(nr,nc,2+nc)
    plt.plot(X_stim_transformed[ind,0],X_stim_transformed[ind,1],'o',c = c)
    plt.text(X_stim_transformed[ind,0],X_stim_transformed[ind,1],display_names[r],fontsize=16,fontweight='bold',
        bbox={'facecolor':c, 'alpha':0.5, 'pad':2, 'edgecolor':'none'})
    plt.title(titles[1])
    plt.axis('square')
    
    ax = fig.add_subplot(nr,nc,3+nc)
    plt.plot(X_go_transformed[ind,0],X_go_transformed[ind,1],'o',c = c)
    plt.text(X_go_transformed[ind,0],X_go_transformed[ind,1],display_names[r],fontsize=16,fontweight='bold',
        bbox={'facecolor':c, 'alpha':0.5, 'pad':2, 'edgecolor':'none'})
    plt.title(titles[2])
    plt.axis('square')
    
    ind+=1

plt.title(titles[2] + ' ' + dir_specific+'_model_'+str(model_n))
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.axis('square')


for epoch_ind in range(1,3):
    epoch = titles[epoch_ind]
    X = X_euc_context.flatten().reshape(-1, 1)
    y = procrust[epoch]['Disparity'].flatten().reshape(-1, 1)
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    Rsqrd = reg.score(X, y)
    
    ax = fig.add_subplot(nr,nc,epoch_ind+2*nc)
    plt.plot(X,y,'.')
    plt.plot(np.sort(X,axis = 0),reg.predict(np.sort(X,axis = 0)),'-k')
    plt.title(dir_specific+'_model_'+str(model_n))
    plt.xlabel('fix disparity')
    plt.ylabel(epoch + ' disparity')
    plt.text(.6*np.max(X),.1*np.max(y),'R^2 = ' + '{0:.2f}'.format(Rsqrd))

    if untrained is True:
        file_name = 'procrustes_model_'+str(model_n)+'untrained'
    else:
        file_name = 'procrustes_model_'+str(model_n)

save_struct = {}
save_struct['Rsqrd'] = Rsqrd
save_struct['procrust'] = procrust
save_struct['X_euc_context'] = X_euc_context
save_struct['tasks'] = tasks
save_struct['titles'] = titles
save_struct['dir_specific'] = dir_specific
    
np.save(fldr + file_name + '.npy', save_struct)
plt.savefig(fldr + file_name + '.svg')
plt.show()
