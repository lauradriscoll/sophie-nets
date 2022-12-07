from __future__ import absolute_import, division, print_function
import glob
import os
import pdb
import numpy as np
import numpy.random as npr
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import pdb
import json
import getpass
from scipy import stats
from sklearn import linear_model
from numpy import linalg as LA
import numpy.random as npr
from sklearn.decomposition import PCA
import imageio
from sklearn.manifold import MDS
from scipy.spatial import distance
from sklearn.cluster import KMeans

ui = getpass.getuser()
if ui == 'laura':
    p = '/home/laura'
elif ui == 'lauradriscoll':
    p = '/Users/lauradriscoll/Documents'
elif ui == 'lndrisco':
    p = '/home/users/lndrisco'

net = 'stepnet'
PATH_YANGNET = os.path.join(p,'code/multitask-nets',net)

sys.path.insert(0, PATH_YANGNET)
from task import generate_trials, rule_name, rule_index_map, rules_dict
from network import Model
import tools
from tools_lnd import name_best_ckpt, generate_Beta_epoch, same_stim_trial
from tools_lnd import get_T_inds, gen_trials_from_model_dir, gen_trials_from_model_dir, align_output_inds
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

PATH_TO_RECURRENT_WHISPERER = p+'/code/recurrent-whisperer'#'/home/laura/code/recurrent-whisperer'#
sys.path.insert(0, PATH_TO_RECURRENT_WHISPERER)
from RecurrentWhisperer import RecurrentWhisperer

PATH_TO_FIXED_POINT_FINDER = p+'/code/fixed-point-finder' #'/home/laura/code/fixed-point-finder-experimental'#
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinder import FixedPointFinder

from tools_lnd import gen_X_from_model_dir, make_axes, TDR_axes, make_MDS_dst, rule_set_names, plot_N, out_axes, get_color_inds

color_inds, color_inds_nu = get_color_inds()

def get_model_params(m):
    model = Model(m)
    with tf.Session() as sess:
        model.restore()
        var_list = model.var_list
        params = [sess.run(var) for var in var_list]

    w_in = params[0]
    b_in = params[1]
    w_out = params[2]
    b_out = params[3]
    return w_in, b_in, w_out, b_out

def make_h_out(model_dir_all,ckpt_n_dir,tasks):
    
    h_combined = []
    h_combined_long = []
    y_combined = []
    epoch = 'go1'
    rule_master = 'delaygo'
    
    trial_master = gen_trials_from_model_dir(model_dir_all,rule_master,noise_on = False)
    w_in, b_in, w_out, b_out = get_model_params(model_dir_all)

    for r in range(len(tasks)):
        
        rule2 = rules_dict['all'][tasks[r]]

        trial2 = gen_trials_from_model_dir(model_dir_all,rule2,mode='random',noise_on = False,batch_size = 2000)
        trial2 = align_output_inds(trial_master, trial2)

        _,x2 = gen_X_from_model_dir(model_dir_all,trial2,d = ckpt_n_dir)
        h_temp = x2[:,:,trial2.epochs['go1'][0]-1].T
        h_temp_long = x2[:,:,trial2.epochs['go1'][0]:trial2.epochs['go1'][0]+15].T
        h_temp_long_vec = np.reshape(h_temp_long,(h_temp_long.shape[0],-1)).T
        y_hat = np.matmul(x2[:,:,-1].T, w_out) + b_out
        theta_temp = np.arctan2(y_hat[:,1],y_hat[:,2])[np.newaxis,:]

        if h_combined == []:
            h_combined = h_temp
            h_combined_long = h_temp_long_vec
            y_combined = theta_temp
        else:
            h_combined = np.concatenate((h_combined, h_temp), axis=0)
            h_combined_long = np.concatenate((h_combined_long, h_temp_long_vec), axis=0)
            y_combined = np.concatenate((y_combined, theta_temp), axis=0)
            
    return h_combined, h_combined_long, y_combined

def plot_h_mds(X_transformed,tasks,n_trials_each,line_alpha = .1,rot = -10,dot_alpha = .3,
               color_inds = color_inds, size=20, ha="left", va="top"):
    
    for ri in range(len(tasks)):
        rule_ind = tasks[ri]
        c = color_inds[rule_ind]
        X_trial = X_transformed[(ri*n_trials_each):((ri+1)*n_trials_each),:]

        plt.plot(X_trial[:,0],X_trial[:,1],'o',c = c,alpha = dot_alpha,label = rule_set_names[rule_ind])
        label_trial = npr.randint(np.shape(X_trial)[0])
        plt.text(np.mean(X_trial[label_trial,0]),np.mean(X_trial[label_trial,1]),rule_set_names[rule_ind],size=size, rotation=rot,
                 ha=ha, va=va,bbox=dict(boxstyle="square",ec='None',fc=c,alpha = dot_alpha))

    plt.xlabel('MDS 1')
    plt.ylabel('MDS 2')
    return

def plot_h_mds_out(X_transformed,y_out,tasks,n_trials_each,line_alpha = .1,rot = -10,dot_alpha = .3,
               size=20, ha="left", va="top"):
    
    cmap=plt.get_cmap('rainbow')
    
    for ri in range(len(tasks)):
        
        y_out_adjust = y_out[ri,:]
        y_out_adjust[y_out_adjust<0] = 2*np.pi+y_out_adjust[y_out_adjust<0]
    
    
        rule_ind = tasks[ri]
        X_trial = X_transformed[(ri*n_trials_each):((ri+1)*n_trials_each),:]

        for xi in range(np.shape(X_trial)[0]):
            c = cmap(y_out_adjust[xi]/(2*np.pi))
            plt.plot(X_trial[xi,0],X_trial[xi,1],'o',c = c,alpha = dot_alpha,label = rule_set_names[rule_ind])
        
    plt.xlabel('MDS 1')
    plt.ylabel('MDS 2')
    return

def plot_h_mds_steps(X_transformed,n_interp,nfps,dot_alpha = .3):
    cmap=plt.get_cmap('plasma')
        
    for ri in range(n_interp):
        c = cmap(ri/n_interp)
        X_trial = X_transformed[(ri*nfps):((ri+1)*nfps),:]
        plt.plot(X_trial[:,0],X_trial[:,1],'o',c = c,alpha = dot_alpha)
        
    plt.xlabel('MDS 1')
    plt.ylabel('MDS 2')
    return

def plot_h_mds_steps_qstar(X_transformed,qstar_vals,size=20,dot_alpha = .3):
    
    cmap=plt.get_cmap('viridis')
    plt.scatter(X_transformed[:,0],X_transformed[:,1],size,-np.log10(qstar_vals),alpha = dot_alpha)
        
    plt.xlabel('MDS 1')
    plt.ylabel('MDS 2')
    return

def make_X_steps_task(m, step_file, epoch, task_list, n_fps_init, t_num = 0, n_interp = 20):
    
    trial = gen_trials_from_model_dir(m,task_list[0],noise_on = False)
    _,x1 = gen_X_from_model_dir(m,trial)
    T_inds = get_T_inds(trial,epoch)
    inds_use = [int(x) for x in np.linspace(T_inds[1],x1.shape[2]-1,n_interp)]
    h_tf = x1[:,:,inds_use]
    fp_steps = []

    for step_i in range(n_interp):
        n_fps = n_fps_init

        filename = task_list[0]+'_'+task_list[1]+'_trial_'+str(t)+'_step_'+str(step_i)
        filename = os.path.join(m,step_file,epoch,task_list[0]+'_'+task_list[1],filename+'.npz')
        fp_struct = np.load(filename)

        sorted_fps = fp_struct['xstar']
        if np.shape(sorted_fps)[0]>n_fps:
            kmeans = KMeans(n_clusters=n_fps, random_state=0).fit(sorted_fps)
            _,fp_inds = np.unique(kmeans.labels_,return_index=True)
        else: 
            fp_inds = range(np.shape(sorted_fps)[0])

        n_fps = len(fp_inds)
        dst_FP = np.zeros((n_fps))
        for s in range(n_fps):
            dst_FP[s] = LA.norm(h_tf[:,t,step_i] - sorted_fps[fp_inds[s],:])

        sorted_inds = [fp_inds[int(x)] for x in np.argsort(dst_FP)]
        fps_sorted = sorted_fps[sorted_inds,:]

        if fp_steps == []:
            fp_steps = fps_sorted
            num_found = [np.shape(fps_sorted)[0],]
            qstar_vals = fp_struct['qstar'][sorted_inds]
        else:
            fp_steps = np.concatenate((fp_steps, fps_sorted), axis=0)
            num_found = num_found+ [np.shape(fps_sorted)[0],]
            qstar_vals = np.concatenate((qstar_vals,fp_struct['qstar'][sorted_inds]))
            
    	X_steps,dst = make_MDS_dst(fp_steps)
    
    return X_steps,dst,fp_steps,num_found,qstar_vals

def make_X_steps_epoch(m, step_file, rule, epoch_set, n_fps_init, t_num = 0, n_interp = 20):
    
    trial = gen_trials_from_model_dir(m,rule,noise_on = False)
    _,x1 = gen_X_from_model_dir(m,trial)
    T_inds = get_T_inds(trial,epoch)
    inds_use = [int(x) for x in np.linspace(T_inds[1],x1.shape[2]-1,n_interp)]
    h_tf = x1[:,:,inds_use]
    fp_steps = []

    for step_i in range(n_interp):
        n_fps = n_fps_init

        filename = epoch_set[0]+'_'+epoch_set[1]+'_trial_'+str(t)+'_step_'+str(step_i)
        filename = os.path.join(m,step_file,rule,filename+'.npz')
        fp_struct = np.load(filename)

        sorted_fps = fp_struct['xstar']
        if np.shape(sorted_fps)[0]>n_fps:
            kmeans = KMeans(n_clusters=n_fps, random_state=0).fit(sorted_fps)
            _,fp_inds = np.unique(kmeans.labels_,return_index=True)
        else: 
            fp_inds = range(np.shape(sorted_fps)[0])

        n_fps = len(fp_inds)
        dst_FP = np.zeros((n_fps))
        for s in range(n_fps):
            dst_FP[s] = LA.norm(h_tf[:,t,step_i] - sorted_fps[fp_inds[s],:])

        sorted_inds = [fp_inds[int(x)] for x in np.argsort(dst_FP)]
        fps_sorted = sorted_fps[sorted_inds,:]

        if fp_steps == []:
            fp_steps = fps_sorted
            num_found = [np.shape(fps_sorted)[0],]
            qstar_vals = fp_struct['qstar'][sorted_inds]
        else:
            fp_steps = np.concatenate((fp_steps, fps_sorted), axis=0)
            num_found = num_found+ [np.shape(fps_sorted)[0],]
            qstar_vals = np.concatenate((qstar_vals,fp_struct['qstar'][sorted_inds]))
            
	X_steps,dst = make_MDS_dst(fp_steps)
    
    return X_steps,dst,fp_steps,num_found,qstar_vals

def plot_steps_tasks(m, task_list,X, qstar_vals, n_interp, n_fps, plotN = True, plotN_epoch = 'delay1'):

    trial_set = range(0,400,10)
    
    D1 = make_axes(m,task_list[0],'go1',ind = 0)
    D2 = make_axes(m,task_list[1],'go1',ind = 0)
    w_in, b_in, w_out, b_out = get_model_params(m)
    D_go = w_out[:,1:]
    
    fig = plt.figure(figsize=(21, 15))
    
    ax1 = plt.subplot(2,3,1)
    X_steps = np.dot(X,D1)
    plot_h_mds_steps(X_steps,n_interp,n_fps,dot_alpha = .5)
    if plotN:
        trial1 = gen_trials_from_model_dir(m,task_list[0])
        _,x = gen_X_from_model_dir(m,trial1)
        T_inds = get_T_inds(trial1,plotN_epoch)
        x1 = np.transpose(x[:,:,T_inds],(1,2,0))
        plot_N(x1[trial_set,:,:],D1.T,trial1.y_loc[-1,trial_set])
    TDR_axes(task_list[0],ax1)

    ax2 = plt.subplot(2,3,2)
    X_steps = np.dot(X,D2)
    plot_h_mds_steps(X_steps,n_interp,n_fps,dot_alpha = .5)    
    if plotN:
        trial2 = same_stim_trial(trial1, 16)
        _,x = gen_X_from_model_dir(m,trial2)
        T_inds = get_T_inds(trial2,plotN_epoch)
        x2 = np.transpose(x[:,:,T_inds],(1,2,0))
        plot_N(x2[trial_set,:,:],D2.T,trial2.y_loc[-1,trial_set])
    TDR_axes(task_list[1],ax2)

    ax3 = plt.subplot(2,3,3)
    X_steps = np.dot(X,D_go)
    plot_h_mds_steps(X_steps,n_interp,n_fps,dot_alpha = .5)
    if plotN:
        plot_N(x1[trial_set,:,:],D_go.T,trial1.y_loc[-1,trial_set])
    out_axes(ax3)
    legend_set = ['step '+ str(x) for x in range(1,21)]
    plt.legend((legend_set),bbox_to_anchor=(1.05, 1))

    ax4 = plt.subplot(2,3,4)
    X_steps = np.dot(X,D1)
    plot_h_mds_steps_qstar(X_steps,qstar_vals,size = 30)
    TDR_axes(task_list[0],ax4)

    ax5 = plt.subplot(2,3,5)
    X_steps = np.dot(X,D2)
    plot_h_mds_steps_qstar(X_steps,qstar_vals,size = 30)
    TDR_axes(task_list[1],ax5)

    ax6 = plt.subplot(2,3,6)
    X_steps = np.dot(X,D_go)
    plot_h_mds_steps_qstar(X_steps,qstar_vals,size = 30)
    out_axes(ax6)
    cbar = plt.colorbar()
    cbar.set_label('-log10(qstar)',rotation = 90)

# ##PART E
t = 0
t_num = t
task_list = ['delaygo','dmsgo'] 
epoch = 'delay1'
rule = 'delaygo'
n_interp = 20
n_fps_init = 60
step_file = 'fixed_pts_stepX_tasks_fwd'
m = '/Users/lauradriscoll/Documents/data/rnn/multitask/crystals_no_noise/softplus/l2w0001/0/'

# X_steps,D,fp_steps,num_found,qstar_vals = make_X_steps_task(m, step_file, epoch, task_list, n_fps_init, t_num = 0, n_interp = 20)
# X = np.squeeze(fp_steps).astype(np.float64)
# plot_steps_tasks(m,task_list,X,qstar_vals,n_interp,n_fps_init,plotN = True, plotN_epoch = 'delay1')

# figpath = os.path.join(m,'output_figure',step_file)
# figname = task_list[0]+'_'+task_list[1]+'_colored_by_both.pdf'
# if not os.path.exists(figpath):
#     os.makedirs(figpath)
# plt.savefig(os.path.join(figpath,figname))


# ##PART F
# t = 0
# t_num = t
# epoch_set = ['delay1','go1']
# epoch = epoch_set[0]
# rule = 'delaygo'
# n_interp = 20
# n_fps_init = 60
# step_file = 'fixed_pts_stepX_reverse'
# m = '/Users/lauradriscoll/Documents/data/rnn/multitask/crystals_no_noise/softplus/l2w0001/0/'

# X_steps,D,fp_steps,num_found,qstar_vals = make_X_steps_epoch(m, step_file, rule, epoch_set, n_fps_init, t_num = 0, n_interp = 20)

# D_delay = make_axes(m,rule,'delay1',ind = -1)
# w_in, b_in, w_out, b_out = get_model_params(m)
# D_go = w_out[:,1:]

# X = np.squeeze(fp_steps).astype(np.float64)

# fig = plt.figure(figsize=(11, 11))
# ax1 = plt.subplot(2,2,1)
# X_steps = np.dot(X,D_delay)
# plot_h_mds_steps(X_steps,n_interp,n_fps_init,dot_alpha = .5)
# TDR_axes('Delay',ax1)

# ax2 = plt.subplot(2,2,2)
# X_steps = np.dot(X,D_go)
# plot_h_mds_steps(X_steps,n_interp,n_fps_init,dot_alpha = .5)
# out_axes(ax2)
# ax2.axis('square')
# legend_set = ['step '+ str(x) for x in range(1,21)]
# plt.legend((legend_set),bbox_to_anchor=(1.3, 1),fontsize = 'small')

# ax1 = plt.subplot(2,2,3)
# X_steps = np.dot(X,D_delay)
# plot_h_mds_steps_qstar(X_steps,qstar_vals,size = 30)
# TDR_axes('Delay',ax1)

# ax2 = plt.subplot(2,2,4)
# X_steps = np.dot(X,D_go)
# plot_h_mds_steps_qstar(X_steps,qstar_vals,size = 30)
# out_axes(ax2)
# ax2.axis('square')

# axins = inset_axes(ax2,
#                    width="5%",  # width = 5% of parent_bbox width
#                    height="70%",  # height : 50%
#                    loc='lower left',
#                    bbox_to_anchor=(1.05, 0., 1, 1),
#                    bbox_transform=ax2.transAxes,
#                    borderpad=0,
#                    )

# cbar = plt.colorbar(cax = axins)
# cbar.set_label('-log10(qstar)',rotation = 90)

# figpath = os.path.join(m,'output_figure',step_file)
# figname = task_list[0]+'_'+task_list[1]+'_colored_by_both.pdf'
# if not os.path.exists(figpath):
#     os.makedirs(figpath)
# plt.savefig(os.path.join(figpath,figname))

## PART G
ckpt_n = name_best_ckpt(m,'multidelaydm')
ckpt_n_dir = os.path.join(m,'ckpts/model.ckpt-' + str(int(ckpt_n)))

trial = gen_trials_from_model_dir(m,'delaygo',mode='test')
n_trials_each = np.shape(trial.x)[1]

tasks = [1,4,18,19,3,5,0,2,11,12,15,13,14,16]
h_out, _, y_out = make_h_out(m,ckpt_n_dir,tasks)
X_out,D = make_MDS_dst(h_out)

new_names = [rule_set_names[x] for x in tasks]

fig = plt.figure(figsize=(21,7),tight_layout=True,facecolor='white')
plt.rcParams.update({'font.size': 14})

ax1 = plt.subplot(1,3,1)
n_trials_each = 400
plot_h_mds(X_out,tasks,n_trials_each,line_alpha = .1,rot = 0,dot_alpha = .3,
               color_inds = color_inds, ha="left", va="top",size= 14)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax1 = plt.subplot(1,3,2)
plot_h_mds_out(X_out,y_out,tasks,n_trials_each,line_alpha = .1,rot = 0,dot_alpha = .3,
               ha="left", va="top",size= 14)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax2 = plt.subplot(1,3,3)
# n_trials_each = 100
plt.imshow(D,'gray')
tick_locs = range(int(n_trials_each/2),n_trials_each*len(tasks),n_trials_each)
plt.xticks(tick_locs,new_names,rotation = 45, ha="right", va="top")
plt.yticks(tick_locs,new_names)

cbar = plt.colorbar(fraction = .03)
cbar.set_label('Euc. Distance')

plt.scatter(range(len(D)),-20*np.ones(len(D)),20,y_out.flatten(),cmap = 'rainbow',marker = '|')
plt.scatter(-20*np.ones(len(D)),range(len(D)),20,y_out.flatten(),cmap = 'rainbow',marker = '_')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

figpath = os.path.join(m,'output_figure')
figname = 'MDS_out_summary_before_go'
if not os.path.exists(figpath):
    os.makedirs(figpath)
plt.savefig(os.path.join(figpath,figname+'.pdf'))
plt.show()

all_fig_var = {}
all_fig_var = {'D':D,
    'X_out':X_out,
    'y_out':y_out,
    'tasks':tasks, 
    'n_trials_each':n_trials_each}

np.savez(os.path.join(figpath,figname+'vars.npz'),**all_fig_var)

#Part H

# procrust = make_procrustes_mat_mov(m,ckpt_n_dir,tasks,nD = 30,err_lim = .2)