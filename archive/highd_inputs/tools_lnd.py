from __future__ import division

import os
import sys
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import re
from datetime import datetime as datetime
from tensorflow.python.ops import parallel_for as pfor
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from scipy import linalg as LA
from sklearn.neighbors import DistanceMetric
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
import numpy.random as npr

import task
from task import generate_trials,rules_dict
from network import Model, get_perf
import tools
import train

def gen_trials_from_model_dir(model_dir,rule,mode='test'):
    model = Model(model_dir)
    with tf.Session() as sess:
        model.restore()
        # model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
#         params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        # create a trial
        trial = generate_trials(rule, hparams, mode=mode, noise_on=True, batch_size =100, delay_fac =1)
    return trial  

def gen_X_from_model_dir(model_dir,d,trial):
    model = Model(model_dir)
    with tf.Session() as sess:
        model.saver.restore(sess,d)
        # model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        hparams = model.hp
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        # run model
        h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
        x = np.transpose(h_tf,(2,1,0)) # h_tf[:,range(1,n_trials),:],(2,1,0))
        X = np.reshape(x,(x.shape[0],-1))
    return X, x    #return orthogonal complement of hidden unit activity to ouput projection matrix

def gen_X_from_model_dir_epoch(model_dir,d,trial,epoch):
    model = Model(model_dir)
    with tf.Session() as sess:
        model.saver.restore(sess,d)
        model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        # create a trial       
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        # run model
        h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
        
        if trial.epochs[epoch][1] is None:
            epoch_range = range(trial.epochs[epoch][0],np.shape(h_tf)[0])
        elif trial.epochs[epoch][0] is None:
            epoch_range = range(0,trial.epochs[epoch][1])
        else:
            epoch_range = range(trial.epochs[epoch][0],trial.epochs[epoch][1])

        x = np.transpose(h_tf[epoch_range,:,:],(2,1,0)) #h_tf[:,range(1,n_trials),:],(2,1,0))
        X = np.reshape(x,(x.shape[0],-1))
    return X, x    #return hidden unit activity

def restore_ckpt(model_dir, ckpt_n):
    ckpt_n_dir = os.path.join(model_dir,'ckpts/model.ckpt-' + str(int(ckpt_n)) + '.meta')
    model = Model(model_dir)
    with tf.Session() as sess:
        model.saver.restore(sess,ckpt_n_dir)
    return model

def find_ckpts(model_dir):
    s_all = []
    ckpt_n_dir = os.path.join(model_dir,'ckpts/')
    for file in os.listdir(ckpt_n_dir):
        if file.endswith('.meta'):
            m = re.search('model.ckpt(.+?).meta', file)
            if m:
                found = m.group(1)
            s_all = np.concatenate((s_all,np.expand_dims(abs(int(found)),axis=0)),axis = 0)
    return s_all.astype(int)

def name_best_ckpt(model_dir,rule):
    s_all = find_ckpts(model_dir)
    s_all_inds = np.sort(s_all)
    s_all_inds = s_all_inds.astype(int)
    fname = os.path.join(model_dir, 'log.json')
    
    with open(fname, 'r') as f:
        log_all = json.load(f)
        x = log_all['cost_'+rule]
           
    y = [x[int(j/1000)] for j in s_all_inds[:-1]]
    ind = int(s_all_inds[np.argmin(y)])
    return ind

def get_model_params(model_dir):

    model = Model(model_dir)
    with tf.Session() as sess:
        model.restore()
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]

    w_in = params[0]
    b_in = params[1]
    w_out = params[2]
    b_out = params[3]

    return w_in, b_in, w_out, b_out

def get_path_names():
    import getpass
    ui = getpass.getuser()
    if ui == 'laura':
        p = '/home/laura'
    elif ui == 'lauradriscoll':
        p = '/Users/lauradriscoll/Documents'
    return p

def plot_N(X, D, clist, linewidth = 1):
    """Plot activity is some 2D space.

        Args:
            X: neural activity in Trials x Time x Neurons
            D: Neurons x 2 plotting dims
        """
    cmap=plt.get_cmap('rainbow')
    S = np.shape(X)[0]
    
    for s in range(S):
        c = cmap(int(clist[s])/max(clist))
        X_trial = np.dot(X[s,:,:],D.T)
        plt.plot(X_trial[0,0],X_trial[0,1],'^',c = c, linewidth = linewidth)
        plt.plot(X_trial[:,0],X_trial[:,1],'-',c = c, linewidth = linewidth)
        plt.plot(X_trial[-1,0],X_trial[-1,1],'o',c = c, linewidth = linewidth)

def do_MDS(X_in):
    X = np.squeeze(X_in).astype(np.float64)
    row_means = np.mean(X, axis=0)
    X_center = X - np.tile(row_means,(X.shape[0],1))
    embedding = MDS(n_components=2)
    X_out = embedding.fit_transform(X_center)
    return X_out

def plot_FP(X, D, qvals, rdots, ldots):
    """Plot activity is some 2D space.

        Args:
            X: Fixed points in #Fps x Neurons
            D: Neurons x 2 plotting dims
    
        """
    log_qvals = np.log10(qvals)

    qmap=plt.get_cmap('gray')
    S = np.shape(X)[0]
    
    for s in range(S):
        c = 'k'
        X_trial = np.dot(X[s,:],D.T)
        plt.plot(X_trial[0],X_trial[1],'*',c = c)
        l = np.concatenate((X_trial - .05*ldots[s], X_trial + .05*ldots[s]),1)
        plt.plot(l[0:4:2],l[1:4:2],'m')
        r = np.concatenate((X_trial - .5*rdots[s], X_trial + .5*rdots[s]),1)
        plt.plot(r[0:4:2],r[1:4:2],'k')

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rot_mat(theta):
    R = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    return R

def tranform_in_rPC(X,R,X_ss):
    Xr_ss = np.dot(R,X_ss.T).T
    Xr = np.dot(R,X.T).T
    if Xr_ss[1,1]>0:
        Xr = np.dot(Xr,np.array(((1,0),(0,-1))))
    return Xr

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def prep_procrustes(data1, data2):
    r"""Procrustes analysis, a similarity test for two data sets.
    
    Parameters
    ----------
    data1 : array_like
        Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    data2 : array_like
        n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).
    Returns
    -------
    mtx1 : array_like
        A standardized version of `data1`.
    mtx2 : array_like
        The orientation of `data2` that best fits `data1`. Centered, but not
        necessarily :math:`tr(AA^{T}) = 1`.
    disparity : float
        :math:`M^{2}` as defined above.
  
    """
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2
    
    return mtx1,mtx2

# def procrustes(mtx1, mtx2):
#     # transform mtx2 to minimize disparity
#     R, s = orthogonal_procrustes(mtx1, mtx2)
#     mtx2 = np.dot(mtx2, R.T) * s

#     # measure the dissimilarity between the two datasets
#     disparity = np.sum(np.square(mtx1 - mtx2))

#     return mtx1, mtx2, disparity, R, s

def same_stim_trial(trial_master, task_num):
    n_stim_per_ring = int(np.shape(trial_master.y)[2]-1)
    stim_rep_size = int(2*n_stim_per_ring+1)
    trial_task_num = trial_master
    trial_task_num.x[:,:,stim_rep_size:] = 0
    trial_task_num.x[:,:,stim_rep_size+task_num] = 1
    return trial_task_num

def pca_denoise(X1,X2,nD):
    pca = PCA(n_components = nD)
    X12 = np.concatenate((X1,X2),axis=1)
    _ = pca.fit_transform(X12.T)
    X1_pca = pca.transform(X1.T)
    X2_pca = pca.transform(X2.T)
    return X1_pca, X2_pca


def procrustes_fit(mtx1, mtx2):
    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R, s

def procrustes_test(mtx1, mtx2, R, s):
    # transform mtx2 to minimize disparity
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity

def make_procrustes_mat_stim(model_dir_all,ckpt_n_dir,epoch,tasks,nD = 10):
    
    procrust = {}
    procrust['Disparity'] = np.zeros((len(tasks),len(tasks)))
    procrust['Scaling'] = np.zeros((len(tasks),len(tasks)))
    
    rule = 'delaygo'
    trial_all = gen_trials_from_model_dir(model_dir_all,rule)
    trial_all_test = gen_trials_from_model_dir(model_dir_all,rule)

    for t1_ind in range(len(tasks)):
        t1 = tasks[t1_ind]

        trial1 = same_stim_trial(trial_all, t1)
        X1,_ = gen_X_from_model_dir_epoch(model_dir_all,ckpt_n_dir,trial1,epoch)

        trial1_test = same_stim_trial(trial_all_test, t1)
        X1_test,_ = gen_X_from_model_dir_epoch(model_dir_all,ckpt_n_dir,trial1_test,epoch)

        for t2_ind in range(len(tasks)):
            if t1_ind !=t2_ind:
                t2 = tasks[t2_ind]

                trial2 = same_stim_trial(trial_all, t2)
                X2,_ = gen_X_from_model_dir_epoch(model_dir_all,ckpt_n_dir,trial2,epoch)
                X1_pca,X2_pca = pca_denoise(X1,X2,nD)
                prep_mtx1, prep_mtx2 = prep_procrustes(X1_pca,X2_pca)
                _, _, disparity_train, R, s = procrustes_fit(prep_mtx1, prep_mtx2)

                trial2_test = same_stim_trial(trial_all_test, t2)
                X2_test,_ = gen_X_from_model_dir_epoch(model_dir_all,ckpt_n_dir,trial2_test,epoch)
                X1_pca_test,X2_pca_test = pca_denoise(X1_test,X2_test,nD)
                prep_mtx1_test, prep_mtx2_test = prep_procrustes(X1_pca_test,X2_pca_test)
                mtx1, mtx2, disparity_test = procrustes_test(prep_mtx1_test, prep_mtx2_test, R, s)

                procrust['Disparity'][t1_ind,t2_ind] = disparity_test
                procrust['Scaling'][t1_ind,t2_ind] = s
    return procrust

def align_output_inds(trial_master, trial_temp):

    indices = range(np.shape(trial_master.y_loc)[1])
    n_out = np.shape(trial_master.y)[2]-1

    for ii in range(np.shape(trial_master.y_loc)[1]):
        if np.max(np.sum(abs(trial_master.x[:,ii,1:(1+n_out)]),axis = 1),axis = 0)>0:
            ind_use = np.max(np.sum(abs(trial_temp.x[:,:,1:(1+n_out)]),axis = 2),axis = 0)>0
        else:
            ind_use = np.max(np.sum(abs(trial_temp.x[:,:,(1+n_out):(1+2*n_out)]),axis = 2),axis = 0)>0

        loc_diff = abs(trial_temp.y_loc[-1,:]-trial_master.y_loc[-1,ii])%(2*np.pi)
        align_ind = [int(i) for i, x in enumerate(loc_diff) if x == min(loc_diff)]
        align_ind_choosey = [x for i, x in enumerate(align_ind) if ind_use[x]]
        if len(align_ind_choosey)==0:
            align_ind_choosey = align_ind
        indices[ii] = align_ind_choosey[npr.randint(len(align_ind_choosey))]
    
    trial_temp_new = trial_temp
    trial_temp_new.x = trial_temp_new.x[:,indices,:]
    trial_temp_new.y = trial_temp_new.y[:,indices,:]
    trial_temp_new.y_loc = trial_temp_new.y_loc[:,indices]
    return trial_temp_new

def project_to_output(model_dir_all,X):
    w_in, b_in, w_out, b_out = get_model_params(model_dir_all)
    y = np.dot(X.T, w_out) + b_out
    return y

def gen_mov_x(model_dir_all,ckpt_n_dir,rule,trial_master):
    trial = gen_trials_from_model_dir(model_dir_all,rule)
    trial = align_output_inds(trial_master, trial)
    _,x = gen_X_from_model_dir_epoch(model_dir_all,ckpt_n_dir,trial,'go1')
    x_out = project_to_output(model_dir_all,x[:,:,-1])
    err = np.sum(np.square(x_out[:,1:] - trial.y[-1,:,1:]),axis=1)
    return err, x

def make_procrustes_mat_mov(model_dir_all,ckpt_n_dir,epoch,tasks,nD = 10,err_lim = .2):
    
    procrust = {}
    procrust['Disparity'] = np.zeros((len(tasks),len(tasks)))
    procrust['Scaling'] = np.zeros((len(tasks),len(tasks)))
    
    rule_all = 'dmcgo'
    trial_all = gen_trials_from_model_dir(model_dir_all,rule_all)
    trial_all_test = gen_trials_from_model_dir(model_dir_all,rule_all)

    for t1_ind in range(len(tasks)):
        t1 = tasks[t1_ind]
        err1, x1 = gen_mov_x(model_dir_all,ckpt_n_dir,rules_dict['all'][t1],trial_all)
        _, x1_test = gen_mov_x(model_dir_all,ckpt_n_dir,rules_dict['all'][t1],trial_all_test)

        for t2_ind in range(len(tasks)):
            if t1_ind !=t2_ind:
                t2 = tasks[t2_ind]
                err2, x2 = gen_mov_x(model_dir_all,ckpt_n_dir,rules_dict['all'][t2],trial_all)
                _, x2_test = gen_mov_x(model_dir_all,ckpt_n_dir,rules_dict['all'][t2],trial_all_test)

                if ckpt_n_dir[-1]==str(1):
                    use_trials = np.multiply(err1>-err_lim,err2>-err_lim) #keep all trials
                else:
                    use_trials = np.multiply(err1<err_lim,err2<err_lim)

                if np.sum(use_trials)>8:
                    X2 = np.reshape(x2[:,use_trials,:15],(np.shape(x2)[0],-1))
                    X1 = np.reshape(x1[:,use_trials,:15],(np.shape(x1)[0],-1))

                    X2_test = np.reshape(x2_test[:,use_trials,:15],(np.shape(x2)[0],-1))
                    X1_test = np.reshape(x1_test[:,use_trials,:15],(np.shape(x1)[0],-1))
                else:
                    raise ValueError('Less than ' + str(np.sum(use_trials)) + ' trials to compare.')

                X1_pca,X2_pca = pca_denoise(X1,X2,nD)
                prep_mtx1, prep_mtx2 = prep_procrustes(X1_pca,X2_pca)
                _, _, disparity_train, R, s = procrustes_fit(prep_mtx1, prep_mtx2)

                X1_pca_test,X2_pca_test = pca_denoise(X1_test,X2_test,nD)
                prep_mtx1_test, prep_mtx2_test = prep_procrustes(X1_pca_test,X2_pca_test)
                mtx1, mtx2, disparity_test = procrustes_test(prep_mtx1_test, prep_mtx2_test, R, s)

                procrust['Disparity'][t1_ind,t2_ind] = disparity_test
                procrust['Scaling'][t1_ind,t2_ind] = s
    return procrust

def make_h_combined(model_dir_all,ckpt_n_dir,tasks,trial_set,n_steps_early = 5):
    
    h_context_combined = []
    h_stim_early_combined = []
    h_stim_late_combined = []
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        rule = 'delaygo'
        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial = generate_trials(rule, hparams, mode='test', noise_on=False, delay_fac =1)
        
        #get size of relevant variables to init mats
        n_inputs = np.shape(trial.x)[2]
        N = np.shape(params[0])[1]
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]
        time_set = [trial.epochs['stim1'][0]] #beginning of stim period
        n_stim_dims = np.shape(trial.x)[2]-20


        for r in range(len(tasks)):
            r_all_tasks_ind = tasks[r]
            
            trial.x[:,:,n_stim_dims:] = 0 #set all tasks to 0 #(n_time, n_trials, n_inputs)
            trial.x[:,:,n_stim_dims+r_all_tasks_ind] = 1 #except for this task
            
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

            # comparing Jacobians to proximity of hidden state across tasks
            # we focus on end of the context period, early, and late in the stim period
            h_context = np.reshape(h_tf[trial.epochs['stim1'][0]-1,trial_set,:],(1,-1)) # h @ end of context period
            h_stim_early = np.reshape(h_tf[trial.epochs['stim1'][0]+n_steps_early,trial_set,:],(1,-1)) # h @ 5 steps into stim
            h_stim_late = np.reshape(h_tf[trial.epochs['stim1'][1],trial_set,:],(1,-1)) # h @ end of stim period

            #concatenate activity states across tasks
            if h_context_combined == []:
                h_context_combined = h_context[np.newaxis,:]
                h_stim_late_combined = h_stim_late[np.newaxis,:]
                h_stim_early_combined = h_stim_early[np.newaxis,:]
            else:
                h_context_combined = np.concatenate((h_context_combined, h_context[np.newaxis,:]), axis=0)
                h_stim_late_combined = np.concatenate((h_stim_late_combined, h_stim_late[np.newaxis,:]), axis=0)
                h_stim_early_combined = np.concatenate((h_stim_early_combined, h_stim_early[np.newaxis,:]), axis=0)

    return h_context_combined, h_stim_late_combined, h_stim_early_combined

def make_Jac(model_dir_all,ckpt_n_dir,rule,task_set,time_set,trial_set):
    n_tasks = len(task_set)
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial = generate_trials('delaygo', hparams, mode='test', noise_on=False, delay_fac =1)
        
        #get size of relevant variables to init mats
        n_inputs = np.shape(trial.x)[2]
        N = np.shape(params[0])[1]
        n_stim_dims = n_inputs - 20
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]

        #init mats
        J_np_u = np.zeros((n_tasks,len(trial_set),N,n_inputs))
        J_np_u_dot_delu = np.zeros((n_tasks,len(trial_set),N))
        eff_in = np.zeros((n_tasks,len(trial_set)))

        for r in range(n_tasks):
            r_all_tasks_ind = task_set[r]
            
            trial.x[:,:,n_stim_dims:] = 0 #set all tasks to 0 #(n_time, n_trials, n_inputs)
            trial.x[:,:,n_stim_dims+r_all_tasks_ind] = 1 #except for this task
            
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

            for trial_i in range(len(trial_set)): #depending on the analysis I was including one or many trials

                inputs = np.squeeze(trial.x[time_set,trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                inputs = inputs[np.newaxis,:]

                states = h_tf[time_set,trial_set[trial_i],:]
                states = states[np.newaxis,:]
                
                #calc Jac wrt inputs
                inputs_context = np.squeeze(trial.x[time_set-1,trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                inputs_context = inputs_context[np.newaxis,:]
                delta_inputs = inputs - inputs_context

                inputs_tf_context = tf.constant(inputs_context, dtype=tf.float32)
                states_tf = tf.constant(states, dtype=tf.float32)
                output, new_states = model.cell(inputs_tf_context, states_tf)
                F_context = new_states

                J_tf_u = pfor.batch_jacobian(F_context, inputs_tf_context, use_pfor=False)
                J_np_u[r,trial_i,:,:] = sess.run(J_tf_u)
                J_np_u_dot_delu[r,trial_i,:] = np.squeeze(np.dot(J_np_u[r,trial_i,:,:],delta_inputs.T))
                eff_in[r,trial_i] = LA.norm(J_np_u_dot_delu[r,trial_i,:])
                    
    return J_np_u_dot_delu, eff_in

def make_h_and_Jac(model_dir_all,ckpt_n_dir,rule,task_set,time_set,trial_set):

    h_context_combined = []
    h_stim_early_combined = []
    h_stim_late_combined = []
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial = generate_trials('delaygo', hparams, mode='test', noise_on=False, delay_fac =1)
        
        #get size of relevant variables to init mats
        n_inputs = np.shape(trial.x)[2]
        N = np.shape(params[0])[1]
        n_stim_dims = n_inputs - 20
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]
        time_set = [trial.epochs['stim1'][0]] #beginning of stim period

        #init mats
        J_np_u = np.zeros((n_tasks,len(trial_set),len(time_set),N,n_inputs))
        J_np_u_dot_delu = np.zeros((n_tasks,len(trial_set),len(time_set),N))

        for r in range(n_tasks):
            r_all_tasks_ind = task_set[r]
            
            trial.x[:,:,n_stim_dims:] = 0 #set all tasks to 0 #(n_time, n_trials, n_inputs)
            trial.x[:,:,n_stim_dims+r_all_tasks_ind] = 1 #except for this task
            
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

            # comparing Jacobians to proximity of hidden state across tasks
            # we focus on end of the context period, early, and late in the stim period
            h_context = np.reshape(h_tf[trial.epochs['stim1'][0]-1,trial_set,:],(1,-1)) # h @ end of context period
            h_stim_early = np.reshape(h_tf[trial.epochs['stim1'][0]+n_steps_early,trial_set,:],(1,-1)) # h @ 5 steps into stim
            h_stim_late = np.reshape(h_tf[trial.epochs['stim1'][1],trial_set,:],(1,-1)) # h @ end of stim period

            #concatenate activity states across tasks
            if h_context_combined == []:
                h_context_combined = h_context[np.newaxis,:]
                h_stim_late_combined = h_stim_late[np.newaxis,:]
                h_stim_early_combined = h_stim_early[np.newaxis,:]
            else:
                h_context_combined = np.concatenate((h_context_combined, h_context[np.newaxis,:]), axis=0)
                h_stim_late_combined = np.concatenate((h_stim_late_combined, h_stim_late[np.newaxis,:]), axis=0)
                h_stim_early_combined = np.concatenate((h_stim_early_combined, h_stim_early[np.newaxis,:]), axis=0)

            for trial_i in range(len(trial_set)): #depending on the analysis I was including one or many trials
                for time_i in range(len(time_set)): #also including one or many time pts

                    inputs = np.squeeze(trial.x[time_set[time_i],trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs = inputs[np.newaxis,:]

                    states = h_tf[time_set[time_i],trial_set[trial_i],:]
                    states = states[np.newaxis,:]
                    
                    #calc Jac wrt inputs
                    inputs_context = np.squeeze(trial.x[time_set[time_i]-1,trial_set[trial_i],:]) #(n_time, n_condition, n_inputs)
                    inputs_context = inputs_context[np.newaxis,:]
                    delta_inputs = inputs - inputs_context

                    inputs_tf_context = tf.constant(inputs_context, dtype=tf.float32)
                    states_tf = tf.constant(states, dtype=tf.float32)
                    output, new_states = model.cell(inputs_tf_context, states_tf)
                    F_context = new_states

                    J_tf_u = pfor.batch_jacobian(F_context, inputs_tf_context, use_pfor=False)
                    J_np_u[r,trial_i,time_i,:,:] = sess.run(J_tf_u)
                    J_np_u_dot_delu[r,trial_i,time_i,:] = np.squeeze(np.dot(J_np_u[r,trial_i,time_i,:,:],delta_inputs.T))
                    
    return J_np_u_dot_delu, h_context_combined, h_stim_late_combined, h_stim_early_combined

def make_axes(model_dir_all,ckpt_n_dir,rule_master,epoch,ind = -1):

    model = Model(model_dir_all)
    with tf.Session() as sess:

        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial_master = generate_trials(rule_master, hparams, mode = 'test', batch_size = 400, noise_on=False, delay_fac =1)
        feed_dict = tools.gen_feed_dict(model, trial_master, hparams)
        h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

    Beta_epoch = generate_Beta_epoch(h_tf,trial_master,ind)
    X_pca = Beta_epoch[epoch]    
    D = np.concatenate((np.expand_dims(X_pca[:,0],axis=1),np.expand_dims(X_pca[:,1],axis=1)),axis = 1)
    return D

def generate_Beta_epoch(h_tf,trial,ind = -1):
    Beta_epoch = {}

    for epoch in trial.epochs.keys():

        T_inds = get_T_inds(trial,epoch)
        T_use = T_inds[ind]
            
        inds_use = np.min(trial.stim_strength,axis=1)>.5
        X = h_tf[T_use,inds_use,:].T
        X_zscore = stats.zscore(X, axis=1)
        X_zscore_nonan = X_zscore
        X_zscore_nonan[np.isnan(X_zscore)] = 0
        r = X_zscore_nonan

        stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
        stim2_locs = np.min(trial.stim_locs[:,[1,3]],axis=1)
        y_loc = trial.y_loc[-1,:]

        if epoch == 'stim1' or epoch == 'delay1':
            angle_var = stim1_locs[inds_use]
        elif epoch =='stim2' or epoch == 'delay2':
            angle_var = stim2_locs[inds_use]
        elif epoch =='go1' or epoch == 'fix1':
            angle_var = y_loc[inds_use]


        y1 = np.expand_dims(np.sin(angle_var),axis = 1)
        y2 = np.expand_dims(np.cos(angle_var),axis = 1)
        y = np.concatenate((y1,y2),axis=1)

        lm = linear_model.LinearRegression()
        model = lm.fit(y,r.T)
        Beta = model.coef_
        Beta_epoch[epoch],_ = LA.qr(Beta)

    return Beta_epoch


