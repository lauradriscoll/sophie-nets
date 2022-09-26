
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import json

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
from task import generate_trials, rule_name, rule_index_map, rules_dict
from network import Model, get_perf, FixedPoint_Model
import tools
from analysis import clustering, standard_analysis, variance
from network import get_perf
from task import generate_trials
from tools_lnd import get_T_inds, make_h_all, gen_X_from_model_dir, gen_trials_from_model_dir, PC_axes
from numpy import linalg as LA
import numpy.random as npr
from sklearn.decomposition import PCA
from scipy.spatial import distance
from numpy import linalg as LA
from tools_lnd import eigenspectrum_axes

model_n = 0
file_spec = 'l2w0001' 
dir_specific_all = os.path.join('crystals','softplus',file_spec)#,supp)
    
m = os.path.join(p,'data/rnn/multitask/',net,dir_specific_all,str(model_n))

figpath = os.path.join(p,'code','overleaf','multitask-nets','cluster-lesions','figs')
if not os.path.exists(figpath):
    os.makedirs(figpath)

def dst_to_h(h,sorted_fps):
    X = np.squeeze(sorted_fps).astype(np.float64)
    dst = np.zeros((np.shape(X)[0]))
    for xi in range(np.shape(X)[0]):
            dst[xi] = distance.euclidean(h, X[xi,:])
    return dst

def proximate_fp(h,sorted_fps):
    proximate_fps = np.argsort(dst_to_h(h,sorted_fps))
    return proximate_fps

def get_h_diff(model_dir, task_list, lesion_units_list, d = []):

    model = FixedPoint_Model(model_dir)
    hp = model.hp

    h_diff = {}

    for ri in range(len(task_list)):
        rule = task_list[ri]

        trial = generate_trials(rule, hp, 'test', noise_on = 'False')

        _,h = gen_X_from_model_dir(model_dir,trial,d = d,lesion_units_list = [])
        _,h_lesion = gen_X_from_model_dir(model_dir,trial,d = d,lesion_units_list = lesion_units_list)

        h_diff[rule] = h - h_lesion

    return h_diff

def compare_fp_lesions(m,lesion_cluster,task_list,color_by = 'stim'):
    
    cmap_grad = plt.get_cmap('rainbow')
    n_tasks = len(task_list)
    epoch_list = ['fix1','stim1','go1']
    ti = 0

    for ei in range(len(epoch_list)):

        epoch = epoch_list[ei]
        fig = plt.figure(figsize=(3*2,3*n_tasks),tight_layout=True,facecolor='white')

        for ri in range(len(task_list)):
            rule = task_list[ri]
            trial = gen_trials_from_model_dir(m,rule,noise_on = False)

            if  epoch in trial.epochs.keys():
                B = np.shape(trial.y_loc)[1]
                trial_set = range(0,B,int(B/10))
                T_inds = get_T_inds(trial,epoch)
                out_theta = int(180*trial.y_loc[-1,ti]/np.pi)
                
                for subplot_i in range(2):

                    if subplot_i>0:
                        ind_l = np.where(CA.labels == lesion_cluster)[0]
                        lesion_units_list = [CA.ind_active[ind_l]][0]
                        f = os.path.join(m,'lesion_fps','tf_fixed_pts_lesion'+str(lesion_cluster+1),rule,epoch+'_'+str(out_theta)+'.0.npz')
                        tit_lesion = 'LESION_'+str(lesion_cluster)
                        a_plot = .8
                        fp_color = 'r'
                    else:
                        lesion_units_list = []
                        f = os.path.join(m,'tf_fixed_pts_all_init',rule,epoch+'_'+str(out_theta)+'.0.npz')
                        
                        if not os.path.isfile(f):
                            nonzero_stim = trial.stim_locs[0,:]<100
                            stim_names = '_'.join(str(int(180*x/np.pi)) for x in trial.stim_locs[ti,nonzero_stim])
                            filename = epoch+'_trial'+str(ti)+'_x'+stim_names+'_y'+str(out_theta)+'.npz'
                            f = os.path.join(m,'tf_fixed_pts_all_init',rule,filename)
                            
                        tit_lesion = 'NO_LESION'
                        a_plot = .3
                        fp_color = 'dodgerblue'

                    fp_struct = np.load(f)
                    xstar = fp_struct['xstar']
                    
                    _,h_all = gen_X_from_model_dir(m,trial,lesion_units_list = lesion_units_list)

                    if subplot_i==0:
                        D_fp = {}
                        n_components = 3
                        pca = PCA(n_components = n_components)
                        X_flat = np.reshape(h_all[:,ti,T_inds],(-1,hp['n_rnn']))
            #             X_flat = xstar
                        fp_pca = pca.fit_transform(X_flat)
                        D_fp['axes'] = pca.components_.T
                        D_fp['labels'] = ['PCA_'+str(x+1) for x in range(n_components)]
                    
                        D_use = D_fp['axes']
                        axes_labels = D_fp['labels']

                    h_ind = h_all[:,ti,T_inds[-1]]
                    proximate_fps = proximate_fp(h_ind,xstar)
                    fp_num = proximate_fps[0]
                    evals, _ = LA.eig(fp_struct['J_xstar'][fp_num,:,:]) 
                    
                    ax1 = fig.add_subplot(n_tasks,2,1+ri*2)
                    D_h = np.dot(D_use.T,h_all[:,ti,T_inds])
                    plt.plot(D_h[0,:],D_h[1,:],'-',c = fp_color,alpha = a_plot)
                    plt.plot(D_h[0,0],D_h[1,0],'x',c = fp_color,alpha = a_plot)
                    plt.plot(D_h[0,-1],D_h[1,-1],'^',c = fp_color,alpha = a_plot)
                    D_fp_all = np.dot(D_use.T,xstar.T)
                    plt.plot(D_fp_all[0,:],D_fp_all[1,:],'.',c = fp_color,markersize = 2,alpha = a_plot)
                    D_fp_proximal = np.dot(D_use.T,xstar[fp_num,:])
                    plt.plot(D_fp_proximal[0],D_fp_proximal[1],'o',c = fp_color,markersize = 10,alpha = a_plot)
                    
                    ax2 = fig.add_subplot(n_tasks,2,2+ri*2)
                    ax2.plot(evals.real,evals.imag,'o',c = fp_color,markersize = 5,alpha = .3)
            #         ax2.plot(evals.real,evals.imag,'.',c = fp_color,alpha = .3)
                
                xs = np.linspace(-1, 1, 1000)
                ys = np.sqrt(1 - xs**2)
                ax2.plot(xs, ys,':k',linewidth = 1)
                ax2.plot(xs, -ys,':k',linewidth = 1)
                plt.xlim((.5,1.1))
                plt.ylim((-.25,.25))
            #     plt.xticks(fontsize = 18)
            #     plt.yticks(fontsize = 18)
                eigenspectrum_axes(epoch,ax2)
                ax2.set_aspect('equal')  
                PC_axes(ax1) 
                ax1.set_title(rule +' '+ epoch + ' lesion# '+str(lesion_cluster))


            figname = epoch+'_fp_lesion'
            plt.savefig(os.path.join(save_dir,figname+'.pdf'))

CA = clustering.Analysis(m, data_type='epoch')

####LOOP THROUGH LESIONS
lesion_var_dir = os.path.join(m,'lesion_variables','lesion_variables'+'.npz')
lesion_var_f = os.path.join(lesion_var_dir,'lesion_variables'+'.npz')

if os.path.exists(lesion_var_f):
    lesion_variables = np.load(lesion_var_f)
    perfs_changes = lesion_variables['perfs_changes']
    print('loaded')
else:
    print('write')
    perfs_changes, cost_changes = CA.lesions()

    lesion_variables = {'perfs_changes':perfs_changes,
                        'cost_changes':cost_changes}

    if not os.path.exists(lesion_var_dir):
        os.makedirs(lesion_var_dir)
    np.savez(lesion_var_f,**lesion_variables)


for cluster in [2,6,8]:#range(np.max(CA.labels)):
    save_dir = os.path.join(figpath,'lesion '+str(cluster))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = FixedPoint_Model(CA.model_dir)
    hp = model.hp

    # fig = plt.figure(figsize=(7,7),tight_layout=True,facecolor='white')
    # ax = plt.subplot(2,2,1)
    # perfs_change_cluster = perfs_changes[cluster,:]
    # plt.plot(perfs_change_cluster,'.k',alpha = .5)
    # # plt.plot([0,len(perfs_change_cluster)],[-.5,-.5],':k',alpha = .3)
    # tasks_affected = np.where(perfs_changes[cluster,:]<-.1)[0]
    # if len(tasks_affected)>0:
    #     plt.plot(tasks_affected,perfs_change_cluster[tasks_affected],'or',alpha = .8)
    # task_list = [hp['rule_trains'][x] for x in tasks_affected]
    # for x in range(len(hp['rule_trains'])):
    #     plt.text(x-.5,perfs_change_cluster[x]-.1,hp['rule_trains'][x],rotation = 90,fontsize = 14)
    # plt.ylabel('Performance Change',fontsize = 14)
    # # plt.xlabel('Tasks',fontsize = 14)
    # plt.ylim([-1,.1])
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.set_xticks([])

    # figname = 'perf_change'
    # plt.savefig(os.path.join(save_dir,figname+'.pdf'))
    # plt.savefig(os.path.join(save_dir,figname+'.png'))

    # #CALC H DIFF
    # # lesion cluster
    # ind_l = np.where(CA.labels == cluster)[0]
    # lesion_units_list = [CA.ind_active[ind_l]][0]

    # rule_train = hp['rule_trains']
    # # h_diff = get_h_diff(CA.model_dir, rule_train, lesion_units_list)

    # # # PLOT H DIFF

    # # nc = 5
    # # nr = int(np.ceil(len(rule_train)/nc))
    # # fig = plt.figure(figsize=(nc*1.8,nr*1.2),tight_layout=True,facecolor='white')
    # # for ri in range(len(rule_train)):
    # #     rule = rule_train[ri]
    # #     trial = generate_trials(rule, hp, 'test', noise_on = 'False')

    # #     ax = plt.subplot(nr,nc,ri+1)

    # #     subselect_trials = h_diff[rule][:,npr.permutation(h_diff[rule].shape[1])[:80],:]
    # #     plt.plot(LA.norm(subselect_trials,axis = 0).T,c = 'k',alpha = .1)

    # #     for epoch in trial.epochs.keys():
    # #         T_inds = get_T_inds(trial,epoch) 
    # #         plt.plot([T_inds[-1],T_inds[-1]],[0,12],'dodgerblue',alpha = .3)

    # #     plt.title(rule)
    # #     plt.ylim([0,12])
    # #     ax.spines['right'].set_visible(False)
    # #     ax.spines['top'].set_visible(False)
        
    # # figname = 'h_diff'
    # # plt.savefig(os.path.join(save_dir,figname+'.pdf'))
    # # plt.savefig(os.path.join(save_dir,figname+'.png'))

    # # SINGLE UNIT ACTIVATIONS
    # cmap = plt.get_cmap('rainbow')

    # cel_use = CA.ind_active[CA.labels==cluster]
    # n_trials = 80
    # n_trials_large = 3200
    # trial_order_large = npr.permutation(n_trials_large)

    # _, h_all_byrule = make_h_all(m)

    # if len(cel_use)>4:
    #     subplot_width = len(cel_use)
    #     subplot_correction = 0
    # else:
    #     subplot_width = 5
    #     subplot_correction = subplot_width-len(cel_use)

    # fig = plt.figure(figsize=(4*subplot_width,4*len(task_list)),tight_layout=True,facecolor='white')

    # for ri in range(len(task_list)):
    #     rule = task_list[ri]
    #     trial = generate_trials(rule, hp, 'test', noise_on=False)
    #     X_use = h_all_byrule[rule]
    #     trial_order = trial_order_large[trial_order_large<np.shape(X_use)[1]][:n_trials]

    #     cel_i = 0
    #     for unit_i in cel_use:
    #         cel_i+=1
    #         ax = plt.subplot(len(task_list),subplot_width,cel_i+subplot_width*ri+subplot_correction)
    #         c_inds = trial.y_loc[-1,trial_order]/np.max(trial.y_loc[-1,:])
    #         X_flat = np.reshape(X_use[:,trial_order,:],(-1,hp['n_rnn']))
    #         for n in range(n_trials):
    #             plt.plot(X_flat[range(n,len(X_flat),n_trials),unit_i],'-',c = cmap(c_inds[n]),alpha = .2)
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['bottom'].set_visible(False)
    #         ax.set_xticks([])
    #         if rule==task_list[0]:
    #             plt.title('cell # '+(str(unit_i)))
    #         if rule==task_list[-1]:
    #             plt.xlabel('time in trial')
    #         if cel_i==1:
    #             plt.ylabel(rule+' activation')
                
    # figname = 'single_unit_activations'
    # plt.savefig(os.path.join(save_dir,figname+'.pdf'))
    # # plt.savefig(os.path.join(save_dir,figname+'.png'))


    # # MAKE PCs

    # X_use = []
    # n_components = 5

    # D_pca = {}
    # if len(task_list)>0:
    #     for rule in task_list:
    #         trial_order = npr.permutation(n_trials)
    #         X_rule = h_all_byrule[rule]
    #         X_flat = np.reshape(X_rule[:,trial_order,:],(-1,hp['n_rnn']))
    #         if len(X_use)==0:
    #             X_use = X_flat
    #         else:
    #             X_use = np.concatenate((X_use,X_flat),axis = 0)

    #     pca = PCA(n_components = n_components)
    #     fp_pca = pca.fit_transform(X_use)
    #     D_pca['axes'] = pca.components_.T
    #     D_pca['labels'] = ['pca_'+str(x+1) for x in range(n_components)]

    #     from tools_lnd import get_model_params
    #     w_in, b_in, w_out, b_out = get_model_params(m)

    #     D_out = {}
    #     D_out['axes'] = w_out[:,1:]
    #     D_out['labels'] = ['W_{out} cos(theta)','W_{out} sin(theta)']

    # # VIS PCs

    #     cels = CA.ind_active[CA.labels==cluster]

    #     fig = plt.figure(figsize=(3*n_components,2),tight_layout=True,facecolor='white')

    #     for x in range(n_components):

    #         ax = plt.subplot(1,n_components,x+1)
    #         plt.plot(D_pca['axes'][:,x],'.k',label = 'all cells',alpha = .4)
    #         plt.plot(cels,D_pca['axes'][cels,x],'o',color = 'orangered',label = 'lesion cells',alpha = .4)
    #         plt.title(D_pca['labels'][x])
    #         if x==n_components-1:
    #             plt.legend()
                
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['bottom'].set_visible(False)
    #         ax.set_xticks([])
            
    #         figname = 'pc_contribution'
    #         plt.savefig(os.path.join(save_dir,figname+'.pdf'))
    #         plt.savefig(os.path.join(save_dir,figname+'.png'))

    #     # POPULATION VIS
    #     ind_l = np.where(CA.labels == cluster)[0]
    #     lesion_units_list = [CA.ind_active[ind_l]][0]

    #     D_use = D_out['axes']
    #     axes_labels = D_out['labels']

    #     D_use = D_pca['axes']
    #     axes_labels = D_pca['labels']

    #     cmap = plt.get_cmap('rainbow')

    #     fig = plt.figure(figsize=(12, 3*len(task_list)))
    #     model = Model(CA.model_dir)
    #     hp = model.hp
    #     with tf.Session() as sess:
    #         model.restore()
    #         model._sigma=0
    #         if len(lesion_units_list)>0:
    #             model.lesion_units(sess, lesion_units_list)

    #         for ri in range(len(task_list)):
    #             rule = task_list[ri]
    #             trial = generate_trials(rule, hp, 'test', noise_on = False)
    #             feed_dict = tools.gen_feed_dict(model, trial, hp)
    #             h_tf = sess.run([model.h], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)

    #             for trial_i in range(0,np.shape(trial.x)[1],int(np.shape(trial.x)[1]/20)):
    #                 c = cmap(trial.y_loc[-1,trial_i]/(2*np.pi))
                    
    #                 ax = plt.subplot(len(task_list),4,1+(4*ri))
    #                 X_rule = np.dot(h_all_byrule[rule][:,trial_i,:],D_pca['axes'])
    #                 plt.plot(X_rule[:,0],X_rule[:,1],c = c,alpha = .5)
    #                 if ri==len(task_list):
    #                     plt.xlabel(axes_labels[0])
    #                 plt.ylabel(axes_labels[1])
    #                 ax.spines['right'].set_visible(False)
    #                 ax.spines['top'].set_visible(False) 
    #                 ax.spines['bottom'].set_visible(False)
    #                 ax.spines['left'].set_visible(False)  
    #                 ax.set_xticks([]) 
    #                 ax.set_yticks([])
    #                 plt.title('WITHOUT lesion : \n'+rule, y=1.0, pad=-14)
    #                 ylims = ax.get_ylim()
    #                 xlims = ax.get_xlim()
                    
    #                 ax = plt.subplot(len(task_list),4,2+(4*ri))
    #                 X_dot = np.dot(h_tf[0][:,trial_i,:],D_pca['axes'])
    #                 plt.plot(X_dot[:,0],X_dot[:,1],c = c,alpha = .5)
    #                 plt.xlim(xlims)
    #                 plt.ylim(ylims)
    #                 if ri==len(task_list):
    #                     plt.xlabel(axes_labels[0])
    #                 plt.ylabel(axes_labels[1])
    #                 ax.spines['right'].set_visible(False)
    #                 ax.spines['top'].set_visible(False) 
    #                 ax.spines['bottom'].set_visible(False)
    #                 ax.spines['left'].set_visible(False)  
    #                 ax.set_xticks([]) 
    #                 ax.set_yticks([])
    #                 plt.title('WITH lesion #'+str(cluster+1)+' : \n'+rule, y=1.0, pad=-14)
                    
    #                 ax = plt.subplot(len(task_list),4,3+(4*ri))
    #                 X_rule = np.dot(h_all_byrule[rule][:,trial_i,:],D_out['axes'])
    #                 plt.plot(X_rule[:,0],X_rule[:,1],c = c,alpha = .5)
    #                 if ri==len(task_list):
    #                     plt.xlabel('out 1')
    #                 plt.ylabel('out 2')
    #                 ax.spines['right'].set_visible(False)
    #                 ax.spines['top'].set_visible(False) 
    #                 ax.spines['bottom'].set_visible(False)
    #                 ax.spines['left'].set_visible(False)  
    #                 ax.set_xticks([]) 
    #                 ax.set_yticks([])
    #                 plt.title('WITHOUT lesion : \n'+rule, y=1.0, pad=-14)
    #                 ylims = ax.get_ylim()
    #                 xlims = ax.get_xlim()
                    
    #                 ax = plt.subplot(len(task_list),4,4+(4*ri))
    #                 X_dot = np.dot(h_tf[0][:,trial_i,:],D_out['axes'])
    #                 plt.plot(X_dot[:,0],X_dot[:,1],c = c,alpha = .5)
    #                 plt.xlim(xlims)
    #                 plt.ylim(ylims)
    #                 if ri==len(task_list):
    #                     plt.xlabel('out 1')
    #                 plt.ylabel('out 2')
    #                 ax.spines['right'].set_visible(False)
    #                 ax.spines['top'].set_visible(False) 
    #                 ax.spines['bottom'].set_visible(False)
    #                 ax.spines['left'].set_visible(False) 
    #                 ax.set_xticks([]) 
    #                 ax.set_yticks([])
    #                 plt.title('WITH lesion #'+str(cluster+1)+' : \n'+rule, y=1.0, pad=-14)
                    
    #             figname = 'population_compare'
    #             plt.savefig(os.path.join(save_dir,figname+'.pdf'))
    #             plt.savefig(os.path.join(save_dir,figname+'.png'))
    # else:

    #     fig = plt.figure(figsize=(3*n_components,2),tight_layout=True,facecolor='white')
    #     figname = 'pc_contribution'
    #     plt.savefig(os.path.join(save_dir,figname+'.pdf'))

    #     fig = plt.figure(figsize=(12, 3*len(task_list)))
    #     figname = 'population_compare'
    #     plt.savefig(os.path.join(save_dir,figname+'.pdf'))
    compare_fp_lesions(CA.model_dir,cluster,hp['rule_trains'][:-2],color_by = 'stim')   

