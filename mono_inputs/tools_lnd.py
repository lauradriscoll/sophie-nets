from __future__ import division

import os
import sys
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import json
from datetime import datetime as datetime
from tensorflow.python.ops import parallel_for as pfor
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from numpy import linalg as LA
import numpy.random as npr
from scipy import stats
from sklearn.manifold import MDS
from scipy.spatial import distance
from tensorflow.python.ops import parallel_for as pfor
import absl
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

import task
from task import generate_trials, rule_name, rules_dict
from network import Model, FixedPoint_Model, get_perf
import tools
import train
from collections import OrderedDict
from analysis import clustering, standard_analysis, variance

rule_set_names = ['DelayGo', 'ReactGo', 'MemoryGo', 'DelayAnti', 'ReactAnti', 'MemoryAnti',
              'Decison1', 'Decison2', 'ContextDecison1', 'ContextDecison2', 'MultiDecison',
              'DelayDecison1', 'DelayDecison2', 'ContextDelayDecison1', 'ContextDelayDecison2', 'MultiDelayDecison',
              'DelayMatch2SampleGo', 'DelayMatch2SampleNogo', 'DelayMatch2CategoryGo', 'DelayMatch2CategoryNoGo']

def gen_trials_from_model_dir(model_dir,rule,mode='test',noise_on = True,batch_size = 500):
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
        trial = generate_trials(rule, hparams, mode=mode, noise_on=noise_on, batch_size =batch_size)
    return trial  

def gen_X_from_model_dir(model_dir,trial,d = [],lesion_units_list = []):
    model = Model(model_dir)
    with tf.Session() as sess:

        if len(d)==0:
            model.restore()
        else:
            model.saver.restore(sess,d)

        if len(lesion_units_list)>0:
            model.lesion_units(sess, lesion_units_list)

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

def gen_X_from_model_dir_epoch(model_dir,trial,epoch,d = []):
    model = Model(model_dir)
    with tf.Session() as sess:

        if len(d)==0:
            model.restore()
        else:
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

def get_model_params(model_dir,ckpt_n_dir = []):

    model = Model(model_dir)
    with tf.Session() as sess:
        if len(ckpt_n_dir)==0:
            model.restore()
        else:
            model.saver.restore(sess,ckpt_n_dir)
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

def plot_training(m):

    model = FixedPoint_Model(m)
    with tf.Session() as sess:
        model.restore()
        hp = model.hp
    task_list = hp['rule_trains']

    fig = plt.figure(figsize=(5, 5))
    cmap=plt.get_cmap('Greys')
    fname = os.path.join(m, 'log.json')

    with open(fname, 'r') as f:
        log_all = json.load(f)
    for r in range(len(task_list)):
        c = cmap((r+1)/(len(task_list)+1))
        ax = fig.add_subplot(1,1,1)
        x = np.log(log_all['cost_'+task_list[r]])
        plt.plot(x,'-',c = c,alpha = .5)
        ax.set_xlabel('Training Step (x 1000)')
        ax.set_ylabel('Log Cost [for each task]')
    #     plt.ylim([-6,2])

    plt.title(m)

    save_dir = os.path.join(m,'training_figs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,'cost_over_training.png'))

def plot_N(X, D, clist, linewidth = 1, alpha = .5, linestyle = '-', cmap_c = 'hsv',markersize = 10):
    """Plot activity is some 2D space.

        Args:
            X: neural activity in Trials x Time x Neurons
            D: Neurons x 2 plotting dims
        """

    cmap=plt.get_cmap(cmap_c)
    S = np.shape(X)[0]
    
    for s in range(S):

        if len(clist)==1:
            c = clist[0]
        else:
            c = cmap(clist[s]/max(clist))

        X_trial = np.dot(X[s,:,:],D.T)
        plt.plot(X_trial[-1,0],X_trial[-1,1],'^',c = c, linewidth = linewidth, alpha = alpha, markersize = markersize)
        plt.plot(X_trial[:,0],X_trial[:,1],linestyle,c = c, linewidth = linewidth, alpha = alpha, markersize = markersize)
        plt.plot(X_trial[0,0],X_trial[0,1],'o',c = c, linewidth = linewidth, alpha = alpha,markersize = markersize)

def plot_N3D(ax, X, D, clist, linewidth = 1, alpha = .5, linestyle = '-'):
    """Plot activity is some 2D space.

        Args:
            X: neural activity in Trials x Time x Neurons
            D: Neurons x 2 plotting dims
        """

    cmap=plt.get_cmap('rainbow')
    S = np.shape(X)[0]
    
    for s in range(S):

        if isinstance(clist, str) :
            c = clist
        elif len(clist)==1:
            c = clist[0]
        else:
            c = cmap(clist[s]/max(clist))

        X_trial = np.dot(X[s,:,:],D.T)
        ax.plot3D(X_trial[:,0],X_trial[:,1],X_trial[:,2],linestyle,c = c, linewidth = linewidth, alpha = alpha)



def plot_FP(X, D, eig_decomps, c='k', al = .2, lw = .5):

    """Plot activity is some 2D space.

        Args:
            X: Fixed points in #Fps x Neurons
            D: Neurons x 2 plotting dims
    
        """
    S = np.shape(X)[0]
    lf = 7
    rf = 7
    D = D[:2,:] #reduce dim in >2
    
    for s in range(S):
        
        X_trial = np.dot(X[s,:],D.T)
        
        n_arg = np.argwhere(eig_decomps[s]['evals']>1)+1
        if len(n_arg)>0:
            for arg in range(np.max(n_arg)):
                rdots = np.dot(np.real(eig_decomps[s]['R'][:, arg]).T,D.T)
                ldots = np.dot(np.real(eig_decomps[s]['L'][:, arg]).T,D.T)
                overlap = np.dot(rdots,ldots.T)
                r = np.concatenate((X_trial - rf*overlap*rdots, X_trial + rf*overlap*rdots),0)
                plt.plot(r[0:4:2],r[1:4:2], c = 'k' ,alpha = .2,linewidth = .5)
        
        n_arg = np.argwhere(eig_decomps[s]['evals']<.3)
        if len(n_arg)>0:
            for arg in range(np.min(n_arg),len(eig_decomps[s]['evals'])):
                rdots = np.dot(np.real(eig_decomps[s]['R'][:, arg]).T,D.T)
                ldots = np.dot(np.real(eig_decomps[s]['L'][:, arg]).T,D.T)
                overlap = np.dot(rdots,ldots.T)
                r = np.concatenate((X_trial - rf*overlap*rdots, X_trial + rf*overlap*rdots),0)
                plt.plot(r[0:4:2],r[1:4:2],'b',alpha = .2,linewidth = .5)

        if np.max(eig_decomps[s]['evals'].real)<1:
            markerfacecolor = c
        else:
            markerfacecolor = 'None'
            
        plt.plot(X_trial[0], X_trial[1], 'o', markerfacecolor = markerfacecolor, markeredgecolor = c, markersize = 10, alpha = .5)

        
def plot_FP_jitter_3D(m,D_use,rule,t_num,fp_epoch,sorted_fps,fp_inds,jit_fps=True,
                   xlabel = 'FP set PC1',ylabel = 'FP set PC2',rand_step_coef = 0.1,n_steps = 100,
                   lw = 3,al = .6,linestyle = '-',n_jit = 0,unstable_qlim = -6,c = 'k'):

    cmap=plt.get_cmap('rainbow')
        
    model = Model(m)
    with tf.Session() as sess:
        model.restore()
        model._sigma=0
        hparams = model.hp
        alpha = hparams['dt']/hparams['tau']
        var_list = model.var_list
        params = [sess.run(var) for var in var_list]

        trial = generate_trials(rule, hparams, mode='test',noise_on=False)
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        h_tf, _ = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
        T,S,N = np.shape(h_tf)
        T_inds = get_T_inds(trial,fp_epoch) # grab epoch time indexing
        x_t = np.matlib.repmat(trial.x[T_inds[1],t_num,:],n_steps,1)
        
        if jit_fps==True:
            for fp_ind in fp_inds:
                for jit in range(n_jit):
                    h0 = sorted_fps[fp_ind,:] + rand_step_coef*npr.randn(N)
                    h_t = vanilla_run_with_h0(params, x_t, h0, hparams)
                    jitter = np.dot(h_t,D_use)
                    ax.plot3D(jitter[:,0],jitter[:,1],jitter[:,2],'-',c = 'k',linewidth = .1)
                
        
        for jit in range(1):
            h0 = h_tf[T_inds[0],t_num,:]
            h_t = vanilla_run_with_h0(params, x_t, h0, alpha)
            jitter = np.dot(h_t,D_use)
            ax.plot3D(jitter[:,0],jitter[:,1],jitter[:,2],'-',c = c,linewidth = 3)

def dst_to_h(h,sorted_fps):
    X = np.squeeze(sorted_fps).astype(np.float64)
    dst = np.zeros((np.shape(X)[0]))
    for xi in range(np.shape(X)[0]):
            dst[xi] = distance.euclidean(h, X[xi,:])
    return dst

def proximate_fp(h,fps):
    if len(fps)==1:
        proximate_fps = 0
    else:
        sorted_fps_list = np.argsort(dst_to_h(h,fps))
        proximate_fps = sorted_fps_list[0]
    return proximate_fps

def load_proximal_fp_from_f(f,h):
    fp_struct = np.load(f)
    xstar = fp_struct['xstar']
    
    proximate_fps = proximate_fp(h,xstar)
    fp_num = proximate_fps
    evals, _ = LA.eig(fp_struct['J_xstar'][fp_num,:,:])
    
    return xstar, fp_struct['J_xstar'], fp_num, evals

class rnn_obj(object):

    default_hps = {
        'tf_dtype': 'float32',
        'random_seed': 0, #IS THIS RIGHT? FIX
        'rnn_cell_feed_dict': {},
    }

    def __init__(self, rnn_cell, sess,
        random_seed=default_hps['random_seed'],
        tf_dtype=default_hps['tf_dtype'],
        rnn_cell_feed_dict=default_hps['rnn_cell_feed_dict']):
        
        self.rnn_cell = rnn_cell
        self.rnn_cell_feed_dict = rnn_cell_feed_dict
        self.session = sess
        self.tf_dtype = getattr(tf, tf_dtype)
        
    def _grab_RNN(self, initial_states, inputs):

        x = tf.Variable(initial_states, dtype=self.tf_dtype)
        x_rnncell = x

        inputs_tf = tf.constant(inputs, dtype=self.tf_dtype)

        output, F_rnncell = self.rnn_cell(inputs_tf, x_rnncell)
        F = F_rnncell
        
        init = tf.variables_initializer(var_list=[x,])
        self.session.run(init)

        return x, F

def calc_Jac(sess,rnn_o,states,inputs):
    x_tf, F_tf = rnn_o._grab_RNN(states, inputs)
    try:
       dFdx_tf = pfor.batch_jacobian(F_tf, x_tf)
    except absl.flags._exceptions.UnparsedFlagAccessError:
       dFdx_tf = pfor.batch_jacobian(F_tf, x_tf, use_pfor=False)
    dFdx_np = sess.run(dFdx_tf)
    return dFdx_np

def calc_relevant_jacobian(m,fp_struct,fp_num,state_diff = []):
    
    model = FixedPoint_Model(m)
    with tf.Session() as sess:
        model.restore()
        hp = model.hp

        rnn_o_hps = {}
        rnn_o = []
        rnn_o = rnn_obj(model.cell,sess, **rnn_o_hps) 

        if len(state_diff)==0:
            state = fp_struct['xstar'][fp_num][np.newaxis,:]
        else:
            state = state_diff[np.newaxis,:]

        inputs = fp_struct['inputs'][fp_num][np.newaxis,:]

        dFdx = calc_Jac(sess,rnn_o,state,inputs)
    return dFdx

def calc_jacobian_at(m,inputs,state,lesion_units = []):
    
    model = FixedPoint_Model(m)
    with tf.Session() as sess:

        model.restore()

        if len(lesion_units)>0:
            model.lesion_units(sess, lesion_units)

        hp = model.hp

        rnn_o_hps = {}
        rnn_o = []
        rnn_o = rnn_obj(model.cell,sess, **rnn_o_hps) 

        dFdx = calc_Jac(sess,rnn_o,state,inputs)
    return dFdx

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

def calc_R_angle(R):
    return np.arccos((np.trace(R)-1)/2)

def tranform_in_rPC(X,R,X_ss):
    Xr_ss = np.dot(R,X_ss.T).T
    Xr = np.dot(R,X.T).T
    if Xr_ss[1,1]>0:
        Xr = np.dot(Xr,np.array(((1,0),(0,-1))))
    return Xr

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def make_Jac_u_dot_delu(model_dir_all,ckpt_n_dir,rule,task_set,time_set,trial_set):
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
        trial = generate_trials(rule, hparams, mode='test', noise_on=False)
        
        #get size of relevant variables to init mats
        n_inputs = np.shape(trial.x)[2]
        N = np.shape(params[0])[1]
        n_stim_dims = n_inputs - 20
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]

        #init mats
        J_np_u = np.zeros((n_tasks,len(trial_set),len(time_set),N,n_inputs))
        J_np_u_dot_delu = np.zeros((n_tasks,len(trial_set),len(time_set),N))

        for r in range(n_tasks):
            r_all_tasks_ind = task_set[r]
            
            trial.x[:,:,n_stim_dims:] = 0 #set all tasks to 0 #(n_time, n_trials, n_inputs)
            trial.x[:,:,n_stim_dims+r_all_tasks_ind] = 1 #except for this task
            
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

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
                    
    return J_np_u_dot_delu

def make_Jac_x(model_dir_all,ckpt_n_dir,rule,task_set,time_set,trial_set):
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
        trial = generate_trials(rule, hparams, mode='test', noise_on=False)
        
        #get size of relevant variables to init mats
        n_inputs = np.shape(trial.x)[2]
        N = np.shape(params[0])[1]
        n_stim_dims = n_inputs - 20
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]

        #init mats
        J_np_x = np.zeros((n_tasks,len(trial_set),len(time_set),N,N))

        for r in range(n_tasks):
            r_all_tasks_ind = task_set[r]
            
            trial.x[:,:,n_stim_dims:] = 0 #set all tasks to 0 #(n_time, n_trials, n_inputs)
            trial.x[:,:,n_stim_dims+r_all_tasks_ind] = 1 #except for this task
            
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

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

                    J_tf_x = pfor.batch_jacobian(F_context, states_tf, use_pfor=False)
                    J_np_x[r,trial_i,time_i,:,:] = sess.run(J_tf_x)
                    
    return J_np_x

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
        trial = generate_trials('delaygo', hparams, mode='test', noise_on=False)
        
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

def prep_procrustes(data1, data2):
    """Procrustes analysis, a similarity test for two data sets.
    
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
    stim_rep_size = int(n_stim_per_ring+1)
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
    procrust['R']= np.zeros((len(tasks),len(tasks)))
    
    
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
                procrust['R'][t1_ind,t2_ind] = calc_R_angle(R)
    return procrust

def same_mov_inds(trial_master, trial_temp): #used to be called align_output_inds

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

def make_procrustes_mat_mov(model_dir_all,ckpt_n_dir,tasks,nD = 10,err_lim = .2):
    
    procrust = {}
    procrust['Disparity'] = np.zeros((len(tasks),len(tasks)))
    procrust['Scaling'] = np.zeros((len(tasks),len(tasks)))
    procrust['R']= np.zeros((len(tasks),len(tasks)))
    
    rule_master = 'delaygo'
    trial_master = gen_trials_from_model_dir(model_dir_all,rule_master)
    trial_master_test = gen_trials_from_model_dir(model_dir_all,rule_master)

    for t1_ind in range(len(tasks)):
        t1 = tasks[t1_ind]
        err1, x1 = gen_mov_x(model_dir_all,ckpt_n_dir,rules_dict['all'][t1],trial_master)
        _, x1_test = gen_mov_x(model_dir_all,ckpt_n_dir,rules_dict['all'][t1],trial_master_test)

        for t2_ind in range(len(tasks)):
            if t1_ind !=t2_ind:
                t2 = tasks[t2_ind]
                err2, x2 = gen_mov_x(model_dir_all,ckpt_n_dir,rules_dict['all'][t2],trial_master)
                _, x2_test = gen_mov_x(model_dir_all,ckpt_n_dir,rules_dict['all'][t2],trial_master_test)

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
                procrust['R'][t1_ind,t2_ind] = calc_R_angle(R)
    return procrust

def make_h_combined(model_dir_all,tasks,trial_set,epoch,ind = -1,ckpt_n_dir = []):

    h_combined = []

    model = Model(model_dir_all)
    with tf.Session() as sess:

        rule = 'delaygo'
        if len(ckpt_n_dir)>0:
            model.saver.restore(sess,ckpt_n_dir)
        else:
            model.restore()
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial_master = generate_trials(rule, hparams, mode='random', batch_size = 100,noise_on=True)

        #get size of relevant variables to init mats
        n_inputs = np.shape(trial_master.x)[2]
        N = np.shape(params[0])[1]
        #change this depending on when in the trial you're looking [must be a transition btwn epochs]
        n_stim_dims = np.shape(trial_master.x)[2]-20
        T_inds = get_T_inds(trial_master,epoch)


        for r in range(len(tasks)):
            r_all_tasks_ind = tasks[r]

            trial_master = generate_trials(rule, hparams, mode='random', batch_size = 100,noise_on=True)
            trial = same_stim_trial(trial_master, r_all_tasks_ind)

            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

            h_epoch = h_tf[T_inds,:,:]
            h_temp = h_epoch[ind,trial_set,:]

            #concatenate activity states across tasks
            if h_combined == []:
                h_combined = h_temp
            else:
                h_combined = np.concatenate((h_combined, h_temp), axis=0)

    return h_combined

def make_h_all(m,mode = 'test',rules = []):
    model = FixedPoint_Model(m, sigma_rec=0)
    with tf.Session() as sess:
        model.restore()
        model._sigma=0

        h_all_byrule = OrderedDict()
        h_all_byepoch = OrderedDict()

        hp = model.hp
        n_hidden = hp['n_rnn']
        
        if len(rules)==0:
            rules = hp['rule_trains']

        for rule in rules:
            trial = generate_trials(rule, hp, mode = mode, noise_on=False, batch_size = 100)
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            h = sess.run(model.h, feed_dict=feed_dict)

            for e_name, e_time in trial.epochs.items():
                # if 'fix' not in e_name:  # Ignore fixation period
                h_all_byepoch[(rule, e_name)] = h[e_time[0]:e_time[1],:,:]
                
            h_all_byrule[rule] = h
                
    return h_all_byepoch, h_all_byrule

def generate_Beta_epoch(h_tf,trial,ind = -1,mod = 'either', ind_adjust = 0):
    Beta_epoch = {}

    for epoch in trial.epochs.keys():

        T_inds = get_T_inds(trial,epoch)
        T_use = T_inds[ind]
            
        inds_use = np.min(trial.stim_strength,axis=1)>.5
        # X = h_tf[T_use,inds_use,:].T
        # X_zscore = stats.zscore(X, axis=1)
        # X_zscore_nonan = X_zscore
        # X_zscore_nonan[np.isnan(X_zscore)] = 0
        # r = X_zscore_nonan

        r = h_tf[T_use,inds_use,:].T

        if mod is 'either':
            stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
            stim2_locs = np.min(trial.stim_locs[:,[1,3]],axis=1)
        elif mod==1:
            stim1_locs = trial.stim_locs[:,0]
            stim2_locs = trial.stim_locs[:,1]
        elif mod==2:
            stim1_locs = trial.stim_locs[:,2]
            stim2_locs = trial.stim_locs[:,3]

        y_loc = trial.y_loc[-1,:]

        if epoch == 'stim1' or epoch == 'delay1':
            angle_var = stim1_locs[inds_use]
        elif epoch =='stim2' or epoch == 'delay2':
            angle_var = stim2_locs[inds_use]
        elif epoch =='go1' or epoch == 'fix1':
            angle_var = stim1_locs[inds_use]

        y1 = np.expand_dims(np.sin(angle_var),axis = 1)
        y2 = np.expand_dims(np.cos(angle_var),axis = 1)
        y = np.concatenate((y1,y2),axis=1)

        lm = linear_model.LinearRegression()
        model = lm.fit(y,r.T)
        Beta = model.coef_
        Beta_epoch[epoch],_ = LA.qr(Beta)

        #Make sure vectors are oriented appropriately
        #first identify a trial that should be in quadrant 1
        quad1_arg = np.argmin((angle_var - np.pi/4)%(2*np.pi))
        quad1_x = h_tf[T_use,quad1_arg,:]
        dr_loc = np.dot(quad1_x,Beta_epoch[epoch])

        #flip vectors so that point is actually in quadrant 1
        if dr_loc[0]<0:
            Beta_epoch[epoch][:,0] = -Beta_epoch[epoch][:,0]
            
        if dr_loc[1]<0:
            Beta_epoch[epoch][:,1] = -Beta_epoch[epoch][:,1]

    return Beta_epoch


def make_Beta(m,task_list,fp_epoch,ind=-1):
    h_all,trial_all,_ = make_h_trial_rule(m,mode = 'test',noise_on = False)
    T_inds = get_T_inds(trial_all[task_list[0]],fp_epoch)
    B,N = np.shape(h_all[task_list[0]][T_inds[ind],:,:])
    R = np.zeros((len(task_list)*B*len(T_inds),N))
    Y = np.zeros((9,len(task_list)*B*len(T_inds)))

    for rule_i in range(len(task_list)):
        rule = task_list[rule_i]
        T_inds = [get_T_inds(trial_all[rule],fp_epoch)[ind],]
        trial_inds = range(0,np.shape(h_all[rule])[1],int(np.shape(h_all[rule])[1]/B))
        h_subselect = h_all[rule][:,trial_inds,:]
        r = np.reshape(h_subselect[T_inds,:,:],(len(T_inds)*B,N))
        
        isanti = 'anti' in task_list[rule_i]
        ismemory = 'delay' in task_list[rule_i]
        y_anti = isanti*np.ones((np.shape(r)[0],1))
        y_memory = ismemory*np.ones((np.shape(r)[0],1))
        y_stim_all = np.repeat(np.min(trial_all[rule].stim_locs[:,[0,2]],axis=1),len(T_inds))
        y_stim = y_stim_all[trial_inds]
        y_out = np.repeat(trial_all[rule].y_loc[-1,trial_inds],len(T_inds))
        
        inds = range((B*len(T_inds))*rule_i,(B*len(T_inds))*(rule_i+1))
        
        R[inds,:] = r
        Y[0,inds] = np.expand_dims(isanti,axis = 0)
        Y[1,inds] = np.expand_dims(ismemory,axis = 0)
        Y[2,inds] = np.expand_dims(np.sin(y_stim),axis = 0)
        Y[3,inds] = np.expand_dims(np.cos(y_stim),axis = 0)
        Y[4,inds] = np.expand_dims(np.sin(y_out),axis = 0)
        Y[5,inds] = np.expand_dims(np.cos(y_out),axis = 0)
        Y[6,inds] = np.repeat(T_inds,B)
        
        Y_labels = ['ANTI','MEMORY','STIM '+ r' $\cos{\theta}$','STIM '+ r' $\sin{\theta}$',
                    'OUT '+ r' $\cos{\theta}$','OUT '+ r' $\sin{\theta}$','TIME']
        
    lm = linear_model.LinearRegression()
    model = lm.fit(Y.T,R)
    Beta = model.coef_
    Beta_qr,_ = LA.qr(Beta)
    return Beta_qr,Y_labels

def make_axes(model_dir_all,rule_master,epoch,ind = -1,mod = 'either',ckpt_n_dir=[]):

    model = Model(model_dir_all)
    with tf.Session() as sess:

        if len(ckpt_n_dir)==0:
            model.restore()
        else:
            model.saver.restore(sess,ckpt_n_dir)

        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        trial_master = generate_trials(rule_master, hparams, mode = 'test', batch_size = 400, noise_on=False)
        feed_dict = tools.gen_feed_dict(model, trial_master, hparams)
        h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)

    # print(mod)
    Beta_epoch = generate_Beta_epoch(h_tf,trial_master,ind,mod = mod)
    X_pca = Beta_epoch[epoch]    
    D = np.concatenate((np.expand_dims(X_pca[:,0],axis=1),np.expand_dims(X_pca[:,1],axis=1)),axis = 1)
    return D

def get_D(dims,h,trial,tasks,epoch,ind = -1):
    D = {}

    if dims=='pca':
        for ri in range(len(tasks)):
            rule = tasks[ri]
            pca = PCA(n_components = 100)
            X = np.reshape(h[rule],(-1,N))
            _ = pca.fit_transform(X)
            D[rule] = pca.components_
    elif dims=='tdr':
        for ri in range(len(tasks)):
            rule = tasks[ri]
            Beta_temp = generate_Beta_epoch(h[rule],trial[rule],ind = ind)
            if (rule[:2] == 'fd') & (epoch == 'delay1'):
                D[rule] = Beta_temp['stim1'].T
            else:
                D[rule] = Beta_temp[epoch].T
    return D

def get_T_inds(trial,epoch):

    T_end = trial.epochs[epoch][1] 
    if T_end is None:
        T_end = np.shape(trial.x)[0]

    T_start = trial.epochs[epoch][0]
    if T_start is None:
        T_start = 0

    T_inds = range(T_start,T_end)

    return T_inds

def generate_Beta_timeseries(h_tf,trial,T_inds,align_group):
    T,S,N = np.shape(h_tf)
    Beta_timeseries = np.empty((N,2,len(T_inds)))

    for t in T_inds:
            
        inds_use = np.min(trial.stim_strength,axis=1)>.5
        X = h_tf[t,inds_use,:].T
        X_zscore = stats.zscore(X, axis=1)
        X_zscore_nonan = X_zscore
        X_zscore_nonan[np.isnan(X_zscore)] = 0
        r = X_zscore_nonan

        stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
        stim2_locs = np.min(trial.stim_locs[:,[1,3]],axis=1)
        y_loc = trial.y_loc[-1,:]
        
        if align_group == 'stim1':
            angle_var = stim1_locs[inds_use]
        elif align_group =='stim2':
            angle_var = stim2_locs[inds_use]
        elif align_group =='go1':
            angle_var = y_loc[inds_use]
        
#         if t<trial.epochs['stim2'][0]:
#             angle_var = stim1_locs[inds_use]
#         elif t<trial.epochs['go1'][0]:
#             angle_var = stim2_locs[inds_use]
#         else:
#             angle_var = y_loc[inds_use]

        y1 = np.expand_dims(np.sin(angle_var),axis = 1)
        y2 = np.expand_dims(np.cos(angle_var),axis = 1)
        y = np.concatenate((y1,y2),axis=1)

        lm = linear_model.LinearRegression()
        model = lm.fit(y,r.T)
        Beta = model.coef_
        Beta_timeseries[:,:,t],_ = LA.qr(Beta)

    return Beta_timeseries

def get_stim_cats(trial):
    #stim locations and category ids
    stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
    stim2_locs = np.min(trial.stim_locs[:,[1,3]],axis=1)

    stim1_cats = stim1_locs<np.pi # Category of stimulus 1
    stim2_cats = stim2_locs<np.pi # Category of stimulus 2
    matchs = stim1_cats == stim2_cats
    
    return stim1_locs, stim2_locs, stim1_cats, stim2_cats

def get_Jacs(model_dir_all, ckpt_n_dir, rule_num, trial_master):

    fpf = []
    J_np = {}
    
    model = Model(model_dir_all)
    with tf.Session() as sess:

        model.saver.restore(sess,ckpt_n_dir)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hparams = model.hp
        
        trial = same_stim_trial(trial_master, rule_num)
        
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_trials, n_neuron)
        
        stim1_locs, stim2_locs, stim1_cats, stim2_cats = get_stim_cats(trial)

        stim1 = trial.epochs['stim1']
        stim2 = trial.epochs['go1']

        inputs_np = []
        inputs_np.append(trial.x[stim1[0],:,:])
        inputs_np.append(trial.x[stim2[0],stim1_cats,:])
        inputs_np.append(trial.x[stim2[0],stim1_cats==0,:])

        states_np = []
        states_np.append(h_tf[stim1[0]-1,:,:])
        states_np.append(h_tf[stim2[0]-1,stim1_cats,:])
        states_np.append(h_tf[stim2[0]-1,stim1_cats==0,:])

        fpf = FixedPointFinder(model.cell,sess)
        
        for bi in range(len(states_np)):
            x_tf, F_tf = fpf._grab_RNN(states_np[bi], inputs_np[bi])
            J_tf = pfor.batch_jacobian(F_tf, x_tf, use_pfor = False)
            J_np[bi] = fpf.session.run(J_tf)
    
    return J_np

def make_h_trial_rule(model_dir_all,mode = 'random',noise_on = False,task_set = []):
    
    trial = {}
    h = {}
    
    model = FixedPoint_Model(model_dir_all)
    with tf.Session() as sess:

        model.restore()#model.saver.restore(sess,ckpt_n_dir)
        # model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hp = model.hp

        if len(task_set)==0:
            task_set = hp['rule_trains']

        for rule in task_set:
            trial[rule] = generate_trials(rule, hp, mode,
                            batch_size=400,noise_on = noise_on)

            # Generating feed_dict.
            feed_dict = tools.gen_feed_dict(model, trial[rule], hp)
            h[rule] = sess.run(model.h, feed_dict=feed_dict)
            
    return h, trial, hp['rule_trains']

def make_cat_h_rules(model_dir_all,mode = 'random',noise_on = False, task_set = []):
    
    cat_h = []
    
    model = FixedPoint_Model(model_dir_all)
    with tf.Session() as sess:

        model.restore()#model.saver.restore(sess,ckpt_n_dir)
        # model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get hparams
        hp = model.hp
        n_rnn = hp['n_rnn']

        if len(task_set)<1:
            task_set = hp['rule_trains']

        for rule in task_set:
            trial = generate_trials(rule, hp, mode,
                            batch_size=400,noise_on = noise_on)

            # Generating feed_dict.
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            h = sess.run(model.h, feed_dict=feed_dict)

            h_temp = np.reshape(np.transpose(h,(2,0,1)),(n_rnn,-1))

            if len(cat_h)==0:
                cat_h = h_temp
            else:
                cat_h = np.concatenate((cat_h,h_temp),axis = 1)
            
    return cat_h

def make_fp_struct(m,fp_file,rule,epoch,ind_stim_loc,trial_set = range(0,360,36)):

    fps = []
    J_xstar = []
    trial_label = []

    if (rule[:2]=='fd') & (epoch=='delay1'):
        epoch_temp = 'stim1'

        for ti in trial_set:
            filename = os.path.join(m,fp_file,rule,epoch_temp+'_'+str(round(ti,2))+'.npz')
            fp_struct = np.load(filename)
            fp_num = np.argsort(np.log10(fp_struct['qstar']))

            fps_temp = fp_struct['xstar'][fp_num,:]
            J_xstar_temp = fp_struct['J_xstar'][fp_num,:,:]

            if len(np.shape(fps_temp))==1:
                fps = fps_temp[np.newaxis,:]
                J_xstar = J_xstar_temp[np.newaxis,:,:]
                trial_label = ti*np.ones(len(fp_num))
            else:
                fps = np.concatenate((fps,fps_temp[np.newaxis,:]),axis = 0)
                J_xstar = np.concatenate((J_xstar,J_xstar_temp[np.newaxis,:,:]),axis = 0)
                trial_label = np.concatenate((trial_label,ti*np.ones(len(fp_num))),axis = 0)

    else:
        filename = os.path.join(m,fp_file,rule,epoch+'_'+str(round(ind_stim_loc,2))+'.npz')
        fp_struct = np.load(filename)
        fp_num = np.argsort(np.log10(fp_struct['qstar']))

        if len(np.shape(fp_struct['xstar'][fp_num,:]))==1:
            fps = fp_struct['xstar'][fp_num,:][np.newaxis,:]
            J_xstar = fp_struct['J_xstar'][fp_num,:,:][np.newaxis,:,:]
        else:
            fps = fp_struct['xstar'][fp_num,:]
            J_xstar = fp_struct['J_xstar'][fp_num,:,:]
        
    return fps, J_xstar

def make_fp_tdr_fig(m,fp_file,rule1,rule2,epoch,ind_stim_loc,tit,trial_set = range(0,360,36),dims = 'tdr'):
    
    nr = 1
    nc = 1
    ms = 10
    
    h,trial,tasks = make_h_trial_rule(m)
    D = get_D(dims,h,trial,[rule1,],epoch,ind = -1)

    fig = plt.figure(figsize=(5.5*nc,4.5*nr),tight_layout=True,facecolor='white')
    cmap=plt.get_cmap('hsv')
    
    for ind_stim_loc in trial_set:
        ind_stim_loc_anti = (ind_stim_loc+180)%360

        if rule1[-4:]=='anti':
            if (rule1 == 'delayanti') & (epoch!='stim1'):
                ind_stim_loc_anti=180
            fps, J_xstar = make_fp_struct(m,fp_file,rule1,epoch,ind_stim_loc_anti,trial_set = trial_set)
        else:
            if (rule1 == 'delaygo') & (epoch!='stim1'):
                ind_stim_loc=0
            fps, J_xstar = make_fp_struct(m,fp_file,rule1,epoch,ind_stim_loc,trial_set = trial_set)

        plt.subplot(nr,nc,1)
        fp_tdr = np.dot(fps,D[rule1].T)
        if (epoch=='delay1') or (epoch=='go1'):
            plt.plot(fp_tdr[:,0],fp_tdr[:,1],'o',c = 'dodgerblue',markersize = ms)
        else:
            plt.plot(fp_tdr[:,0],fp_tdr[:,1],'o',c = cmap(ind_stim_loc/360),markersize = ms)

        if rule2[-4:]=='anti':
            if (rule2 == 'delayanti') & (epoch!='stim1'):
                ind_stim_loc_anti=180
            fps, J_xstar = make_fp_struct(m,fp_file,rule2,epoch,ind_stim_loc_anti,trial_set = trial_set)
        else:
            if (rule2 == 'delaygo') & (epoch!='stim1'):
                ind_stim_loc=0
            fps, J_xstar = make_fp_struct(m,fp_file,rule2,epoch,ind_stim_loc,trial_set = trial_set)

        plt.subplot(nr,nc,1)
        fp_tdr = np.dot(fps,D[rule1].T)
        if (epoch=='delay1') or (epoch=='go1'):
            plt.plot(fp_tdr[:,0],fp_tdr[:,1],'o',c = 'orangered',markersize = ms)
        else:
            print(epoch)
            plt.plot(fp_tdr[:,0],fp_tdr[:,1],'o',c = cmap(ind_stim_loc/360),markerfacecolor = 'w',markersize = ms)
        
    ax = plt.subplot(nr,nc,1)
    if dims == 'tdr':
        plt.xlabel(rule1 + ' TDR input 1')
        plt.ylabel(rule1 + ' TDR input 1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('Fixed Points : ' + tit)
    plt.legend((rule1,rule2))
    return ax

def plot_epoch_dynamics(m,fp_file,epoch,h,trial,rule,D_use,
                        y_set = range(0,360,36),h_epoch = [],plot_eigenspect = True,lim=4, ax_type = 'tdr',
                        epoch_axes =[],stim_loc_fp = 0):


    xs = np.linspace(-1, 1, 1000)
    ys = np.sqrt(1 - xs**2)
    
    epoch_name, rule_name, epoch_axes_name, h_epoch = take_names(epoch,rule,epoch_axes = epoch_axes,
                                                                 h_epoch = h_epoch)
        
    nr = 1
#     if plot_eigenspect:
#         nc = 2
#     else:
#         nc = 1
    nc = 2
        
    al = .2
    
    fig = plt.figure(figsize=(4.5*nc,5*nr),tight_layout=True,facecolor='white')
    cmap = plt.get_cmap('hsv')
    
    stim1_locs = np.min(trial[rule].stim_locs[:,[0,2]],axis=1)
    y_locs = trial[rule].y_loc[-1,:]
    
    ax = plt.subplot(nr,nc,1)
    T_inds = get_T_inds(trial[rule],h_epoch)
    h_tdr = np.empty((len(T_inds),np.shape(h[rule])[1]))
    for t in range(0,np.shape(h[rule])[1],2):
        h_tdr_temp = np.dot(h[rule][T_inds,t,:],D_use)
#         print(stim1_locs[t])
        if stim1_locs[t]==stim_loc_fp:
            plt.plot(h_tdr_temp[:,0],h_tdr_temp[:,1],c = cmap(stim1_locs[t]/(2*np.pi)),alpha = 1,linewidth = 3)
            plt.plot(h_tdr_temp[0,0],h_tdr_temp[0,1],'x',c = cmap(stim1_locs[t]/(2*np.pi)),alpha = 1,
                     markersize = 10,linewidth = 2)
        else:
            plt.plot(h_tdr_temp[:,0],h_tdr_temp[:,1],c = cmap(stim1_locs[t]/(2*np.pi)),alpha = al,linewidth = 2)
            plt.plot(h_tdr_temp[0,0],h_tdr_temp[0,1],'x',c = cmap(stim1_locs[t]/(2*np.pi)),alpha = al,
                     markersize = 10,linewidth = 2)
        
    lim = lim
    ax = add_ax_labels(ax,ax_type,lim,epoch_axes_name,rule_name)
        
    plt.title(r"$\bf{" + rule_name + "}$"+ '\n '+epoch_name+' dynamics',y = .9)
    if ax_type!='mix':
        ax.set_aspect('equal')
    
    for ind_stim_loc_anti in y_set:
        fps_anti, J_xstar = make_fp_struct(m,fp_file,rule,epoch,ind_stim_loc_anti)
        eig_decomps = comp_eig_decomp(J_xstar)
        fps_tdr_anti = np.dot(fps_anti,D_use)
        fp_c = cmap(ind_stim_loc_anti/(360))
        plt.plot(fps_tdr_anti[:,0],fps_tdr_anti[:,1],'o',c = 'k',alpha = .5,markersize=6)
        plot_FP(fps_anti, D_use.T, eig_decomps, c='k')

        if plot_eigenspect:
            ax2 = fig.add_axes([.45, .45, .2, .2])
#             ax = plt.subplot(2*nr,2*nc,7)
            for fp_num in range(np.shape(J_xstar)[0]):
                evals, _ = LA.eig(J_xstar[fp_num,:,:]) 
                ax2.plot(evals.real,evals.imag,'.k',alpha = .3,markerfacecolor = 'k')

            ax2.plot(xs, ys,':k',linewidth = 1)
            ax2.plot(xs, -ys,':k',linewidth = 1)
            # plt.xlim((.7,1.1))
            # plt.ylim((-.15,.15))
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            eigenspectrum_axes(epoch,ax2)
            ax2.set_aspect('equal')   
            
    return ax

def make_FP_axs(f,m_all,rule_in,fp_epoch,n_fps = 5,axs ='pca_fp', clust = 'False',n_components = 3):
        
    if axs == 'tdr':
        D_use = make_axes(m_all,rule_in,fp_epoch)
    elif axs =='pca_fp':

        fp_struct = np.load(f)
        sorted_fps = fp_struct['xstar']

        if (clust == 'True') & (len(sorted_fps)>2):
            kmeans = KMeans(n_clusters=np.min((n_fps,np.shape(fp_struct['xstar'])[0])), random_state=0).fit(sorted_fps)
            _,fp_inds = np.unique(kmeans.labels_,return_index=True)
        
        else:
            fp_inds = range(len(sorted_fps))

        pca = PCA(n_components = n_components)
        fp_pca = pca.fit_transform(sorted_fps[fp_inds,:])
        D_use = pca.components_.T
    elif axs =='pca_h':
        trial = gen_trials_from_model_dir(m_all,rule_in,mode='random',noise_on = False)
        X, _ = gen_X_from_model_dir_epoch(m_all,trial,fp_epoch)
        pca = PCA(n_components = n_components)
        fp_pca = pca.fit_transform(X.T)
        D_use = pca.components_.T
    elif axs =='out':
        w_in, b_in, w_out, b_out = get_model_params(m_all)
        D_use = w_out[:,1:]

    return D_use
    
def add_ax_labels(ax,ax_type,lim,epoch_axes_name,rule_name):
    if ax_type == 'out':
        (ax)
    elif ax_type == 'pca':
        PC_axes(ax)
        plt.ylim((-lim,lim))
        plt.xlim((-lim,lim))
    elif ax_type == 'tdr':
        TDR_axes(epoch_axes_name,ax,rule_name)
        plt.ylim((-lim,lim))
        plt.xlim((-lim,lim))
    elif ax_type == 'mix':
        plt.ylim((-1.5,1.5))
        plt.xlim((-lim,lim))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
#         plt.xlabel('TDR : '+r"$\bf{" + rule_name + "}$"+ ' '+ epoch_axes_name +r' $\cos{\theta}$',fontsize = 20)
        plt.xlabel('TDR : '+r' $\sin{\theta}$',fontsize = 20)
        plt.ylabel('Output ' + r' $\sin{\theta}$',fontsize = 20)
    return (ax)

def TDR_axes(epoch,ax,rule_name):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
#     plt.xlabel('TDR : '+epoch+ r' $\cos{\theta}$')
#     plt.ylabel('TDR : ' +epoch+ r' $\sin{\theta}$')
    plt.xlabel('TDR : '+r"$\bf{" + rule_name + "}$"+ ' '+r' $\cos{\theta}$',fontsize = 20) #+r"$\bf{" + rule_name + "}$"+ ' '
    plt.ylabel('TDR : '+r"$\bf{" + rule_name + "}$"+ ' '+r' $\sin{\theta}$',fontsize = 20) #+r"$\bf{" + rule_name + "}$"+ ' '
    
def eigenspectrum_axes(epoch,ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Real Part',fontsize = 20)
    plt.ylabel('Imaginary Part',fontsize = 20,labelpad=1)
    
def out_axes(ax):

    plt.ylim((-1.5,1.5))    
    plt.xlim((-1.5,1.5))  
    
    plt.xlabel('Output ' + r' $\cos{\theta}$',fontsize = 20)
    plt.ylabel('Output ' + r' $\sin{\theta}$',fontsize = 20)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    
def PC_axes(ax):

    ax.set_xlabel('h PC1',fontsize = 20)
    ax.set_ylabel('h PC2',fontsize = 20)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def FP_PC_axes(ax):

    ax.set_xlabel('FP PC1',fontsize = 20)
    ax.set_ylabel('FP PC2',fontsize = 20)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 

def get_color_inds():
    tab20 = plt.get_cmap('tab20')
    tab20b = plt.get_cmap('tab20b')
    tab20c = plt.get_cmap('tab20c')
    color_inds = (tab20c(8/20),tab20c(9/20),tab20c(10/20),tab20c(4/20),tab20c(5/20),tab20c(6/20),
                 tab20b(0/20),tab20b(12/20),tab20b(1/20),tab20b(13/20),tab20b(16/20),
                 tab20b(2/20),tab20b(14/20),tab20b(3/20),tab20b(15/20),tab20b(17/20),
                 tab20(18/20),tab20(19/20),tab20(16/20),tab20(17/20))


    color_inds_nu = (tab20c(8/20),tab20c(9/20),tab20b(0/20),tab20c(4/20),tab20c(5/20),tab20b(12/20),
                 tab20b(0/20),tab20b(12/20),tab20b(1/20),tab20b(13/20),tab20b(16/20),
                 tab20b(2/20),tab20b(14/20),tab20b(3/20),tab20b(15/20),tab20b(17/20),
                 tab20(18/20),tab20(19/20),tab20(16/20),tab20(17/20))

    return color_inds, color_inds_nu

def plot_FP_jitter(m,D_use,rule_master,t_num,fp_epoch,sorted_fps,fp_inds,eig_decomps,rule_set,
                   xlabel = 'FP set PC1',ylabel = 'FP set PC2',rand_step_coef = 0.1,n_steps = 100,
                   lw = 3,al = .6,linestyle = '-',n_jit = 0,c_master = 'k',lesion_units = []):

    cmap=plt.get_cmap('rainbow')
        
    model = FixedPoint_Model(m)
    with tf.Session() as sess:
        model.restore()
        if len(lesion_units)>0:
            model.lesion_units(sess, lesion_units)
        model._sigma=0
        hparams = model.hp
        alpha = hparams['dt']/hparams['tau']
        var_list = model.var_list
        params = [sess.run(var) for var in var_list]

        for rule in rule_set:
            trial = generate_trials(rule, hparams, mode='test',noise_on=False)
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf, _ = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
            T,S,N = np.shape(h_tf)
            T_inds = get_T_inds(trial,fp_epoch) # grab epoch time indexing

            for s in range(0,S,int(S/8)):

                if c_master == 'y_locs':
                    c_inds = trial.y_loc[-1,:]
                    c = cmap(c_inds[s]/(2*np.pi))
                elif c_master == 'x_locs':
                    stim1_locs = np.min(trial.stim_locs[:,[0,2]],axis=1)
                    c = cmap(stim1_locs[s]/(2*np.pi))
                else:
                    c = c_master

                X_trial = np.dot(h_tf[T_inds,s,:],D_use)
                plt.plot(X_trial[0,0],X_trial[0,1],'x',c = c, alpha = al, linewidth = lw)
                plt.plot(X_trial[:,0],X_trial[:,1],linestyle,c = c, alpha = al, linewidth = lw)
                plt.plot(X_trial[-1,0],X_trial[-1,1],'^',c = c, alpha = al, linewidth = lw)

        trial = generate_trials(rule_master, hparams, mode='test',noise_on=False)
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
        T_inds = get_T_inds(trial,fp_epoch)

        if c_master == 'y_locs':
            c_inds = trial.y_loc[-1,:]
            c = cmap(c_inds[t_num]/(2*np.pi))
        elif c_master == 'x_locs':
            stim1_locs = np.min(trial.stim_locs,axis=1)
            c = cmap(stim1_locs[t_num]/(2*np.pi))
        else:
            c = c_master

        X_trial = np.dot(h_tf[T_inds,t_num,:],D_use)
        plt.plot(X_trial[0,0],X_trial[0,1],'x',c = c, alpha = .8, linewidth = 5)
        plt.plot(X_trial[:,0],X_trial[:,1],linestyle,c = c, alpha = .8, linewidth = 5)
        plt.plot(X_trial[-1,0],X_trial[-1,1],'^',c = c, alpha = .8, linewidth = 5, markersize = 10)

        for fp_ind in fp_inds:

            if np.max(eig_decomps[fp_ind]['evals'])>.99:
                markerfacecolor = 'None'

            else:
                markerfacecolor = c

            x_t = np.matlib.repmat(trial.x[T_inds[0],t_num,:],n_steps,1)
            for jit in range(n_jit):
                h0 = sorted_fps[fp_ind,:] + rand_step_coef*npr.randn(N)
                h_t = vanilla_run_with_h0(params, x_t, h0, hparams)
                jitter = np.dot(h_t,D_use)
                plt.plot(jitter[-1,0],jitter[-1,1],'^k',linewidth = .1,alpha = .1)
                plt.plot(jitter[:,0],jitter[:,1],'-k',linewidth = .1)

            fp = np.dot(sorted_fps[fp_ind,:],D_use)
            # print(np.shape(fp))
            plt.plot(fp[0],fp[1],'o',c = c,linewidth = 5,markersize = 6,markerfacecolor = markerfacecolor)

            h_t = vanilla_run_with_h0(params, x_t, h_tf[T_inds[0],t_num,:], hparams)
            jitter = np.dot(h_t,D_use)
            plt.plot(jitter[-1,0],jitter[-1,1],'^k',c = c)
            plt.plot(jitter[:,0],jitter[:,1],'-k',c = c,linewidth = 1)
            plt.plot(jitter[0,0],jitter[0,1],'xk',c = c,linewidth = 5,markersize = 10)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(rule_master)

def out_affine(params, h):
    return np.dot(params[2].T,h)+params[3]

def relu(x):
    f = x * (x > 0)
    return f

def rnn_vanilla(params, h, x, alpha, activation):

    if activation == 'softplus':
        _activation = lambda x: np.log(np.exp(x) + 1)
    elif activation == 'tanh':
        _activation = lambda x: np.tanh(x)
    elif activation == 'relu':
        _activation = lambda x: x * (x > 0)
    elif activation == 'power':
        _activation = lambda x: (x * (x > 0))**2
    elif activation == 'retanh':
        _activation = lambda x: np.tanh(x * (x > 0))
    
    xh = np.concatenate([x,h], axis=0)
    gate_inputs = np.dot(params[0].T,xh)+params[1]
    noise = 0
    output = _activation(gate_inputs) # + noise

    h_new = (1-alpha) * h + alpha * output
    
    return h_new

def vanilla_run_with_h0(params, x_t, h0, hparams):

    dt = hparams['dt']
    tau = hparams['tau']
    alpha = dt/tau
    activation = hparams['activation']

    h = h0
    h_t = []
    h_t.append(np.expand_dims(h0,axis=1))
    for x in x_t:
        h = rnn_vanilla(params, np.squeeze(h), np.squeeze(x.T), alpha, activation)
        h_t.append(np.expand_dims(h,axis=1))

    h_t = np.squeeze(np.array(h_t))  
    return h_t

def vanilla_run_at_fp(params, x_t, h0, alpha):

    h = h0
    h_t = []
    h_t.append(h)
    for x in x_t:
        h = rnn_vanilla(params, np.squeeze(h), np.squeeze(x.T), alpha, activation)
        h_t.append(np.expand_dims(h,axis=1))
        
    return h_t

def make_dst_mat(X):
    X = np.squeeze(X).astype(np.float64)
    dst = np.zeros((np.shape(X)[0],np.shape(X)[0]))
    for xi in range(np.shape(X)[0]):
        for yi in range(np.shape(X)[0]):
            dst[xi,yi] = distance.euclidean(X[xi,:], X[yi,:])
    return dst

def make_angle_mat(X):
    X = np.squeeze(X).astype(np.float64)

    theta_mat = np.zeros((np.shape(X)[0],np.shape(X)[0]))
    for xi in range(np.shape(X)[0]):
        v1 = X[xi,:]
        for yi in range(np.shape(X)[0]):
            v2 = X[yi,:]
            theta_mat[xi,yi] = angle_between(v1, v2)

    return theta_mat

    

def make_MDS_dst(h_out):
    X = np.squeeze(h_out).astype(np.float64)
    dst = np.zeros((np.shape(X)[0],np.shape(X)[0]))
    for xi in range(np.shape(X)[0]):
        for yi in range(np.shape(X)[0]):
            dst[xi,yi] = distance.euclidean(X[xi,:], X[yi,:])
            
    embedding = MDS(n_components = 2,dissimilarity = 'precomputed')
    X_out = embedding.fit_transform(dst)
    return X_out,dst
    
def comp_eig_decomp(Ms, sort_by='real',do_compute_lefts=True):
  """Compute the eigenvalues of the matrix M. No assumptions are made on M.

  Arguments: 
    M: 3D np.array nmatrices x dim x dim matrix
    do_compute_lefts: Compute the left eigenvectors? Requires a pseudo-inverse 
      call.

  Returns: 
    list of dictionaries with eigenvalues components: sorted 
      eigenvalues, sorted right eigenvectors, and sored left eigenvectors 
      (as column vectors).
  """
  if sort_by == 'magnitude':
    sort_fun = np.abs
  elif sort_by == 'real':
    sort_fun = np.real
  else:
    assert False, "Not implemented yet."      
  
  decomps = []
  L = None  
  for M in Ms:
    evals, R = LA.eig(M)    
    indices = np.flipud(np.argsort(sort_fun(evals)))
    if do_compute_lefts:
      L = LA.pinv(R).T  # as columns      
      L = L[:, indices]
    decomps.append({'evals' : evals[indices], 'R' : R[:, indices],  'L' : L})
  
  return decomps

def take_names(epoch,rule,epoch_axes = [],h_epoch = []):
    epochs = ['fix1','stim1','delay1','stim2','delay2','go1']
    epoch_names = ['context','stimulus','memory','stimulus','memory','response']

    print(epoch)

    ei = [i for i,e in enumerate(epochs) if e==epoch]
    epoch_name = epoch_names[ei[0]]
    
    rules = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

    rule_names = ['DelayPro', 'ReactPro', 'MemoryPro', 'DelayAnti', 'ReactAnti', 'MemoryAnti',
              'DelayDecison1', 'DelayDecison2', 'ContextDelayDecison1', 'ContextDelayDecison2', 'MultiDelayDecison',
              'DelayMatch2SamplePro', 'DelayMatch2SampleAnti', 'DelayMatch2CategoryPro', 'DelayMatch2CategoryAnti']

    ri = [i for i,e in enumerate(rules) if e==rule]
    rule_name = rule_names[ri[0]]
    
    if len(epoch_axes)<1:
        epoch_axes_name = epoch_names[ei[0]]
    else:
        ei = [i for i,e in enumerate(epochs) if e==epoch_axes]
        epoch_axes_name = epoch_names[ei[0]]
    
    if len(h_epoch)==0:
        h_epoch = epoch
    
    return epoch_name, rule_name, epoch_axes_name, h_epoch

import scipy
import pylab
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster

def make_dendro(m,method = 'ward',cel_max_d = 3.5,criterion = 'distance'):

    CA = clustering.Analysis(m, data_type='epoch')
    tick_names = [rule_name[key[0]]+' '+key[1] for key in CA.keys]

    # Generate features and distance matrix.
    D  = CA.h_normvar_all.T

    # Compute and plot dendrogram.
    fig = pylab.figure(figsize=(24, 15))
    axdendro = fig.add_axes([0.09,0.1,0.05,0.75])
    Y = sch.linkage(D, method=method)

    if criterion == 'maxclust':
        max_d = 14 #max number of task clusters
        clusters = fcluster(Y, max_d, criterion='maxclust') #CHANGE hard coded 14 clusters
    else:
        max_d = 5 #threshold for task clusters
        clusters = fcluster(Y, max_d, criterion='distance')

    Z = sch.dendrogram(Y, orientation='left',labels = tick_names,
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
    tick_names_sorted = [tick_names[i] for i in index_left]
    D = D[index_left,:]

    # cel_num = [CA.ind_active[x] for x in index_top]
    axdendro_top = fig.add_axes([0.22,.9,0.75,0.1])
    Y = sch.linkage(D.T, method=method)

    if criterion== 'maxclust':
        clusters = fcluster(Y, cel_max_d, criterion='maxclust') #CHANGE hard coded 14 clusters
        Z = sch.dendrogram(Y, orientation='top',labels = clusters, #CA.ind_active #clusters
                       leaf_font_size = 11,color_threshold=0)

    else:
        clusters = fcluster(Y, cel_max_d, criterion='distance')
        Z = sch.dendrogram(Y, orientation='top',labels = clusters, #CA.ind_active #clusters
                       leaf_font_size = 11,color_threshold=cel_max_d)

    # axdendro_top.set_xticks([])
    axdendro_top.set_yticks([])
    axdendro_top.spines['top'].set_visible(False)
    axdendro_top.spines['right'].set_visible(False)
    axdendro_top.spines['bottom'].set_visible(False)
    axdendro_top.spines['left'].set_visible(False)

    index_top = Z['leaves']
    D = D[:,index_top]
    clusters_sorted = clusters[index_top]
    im = axmatrix.matshow(D, aspect='auto', origin='lower',cmap='magma')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.975,0.1,0.01,0.8])
    pylab.colorbar(im, cax=axcolor)

    lesion_units_list = [None]
    for il, l in enumerate(np.unique(clusters)):
        ind_l = np.where(clusters == l)[0]
        # In original indices
        lesion_units_list += [CA.ind_active[ind_l]]

    # save cluster variables
    cluster_var = {'D':D,
                'index_top':index_top,
                'index_left':index_left,
                'tick_names':tick_names_sorted,
                'clusters':clusters_sorted,
                'lesion_units_list':lesion_units_list,
                'max_d':cel_max_d,
                'criterion':criterion,
                'method':method}

    lesion_folder = 'lesion_fps_hierarchical_'+method+'_'+criterion+'_max_d'+str(cel_max_d)
    save_dir = os.path.join(m,lesion_folder)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez(os.path.join(save_dir,'cluster_var.npz'),**cluster_var)
    plt.savefig(os.path.join(save_dir,'dynamic_modules_atlas'+'.pdf'))
    plt.savefig(os.path.join(save_dir,'dynamic_modules_atlas'+'.png'))

    # Display and save figure.
    p = get_path_names()
    figpath = os.path.join(p,'code','overleaf','multitask-nets','v1_figs','clusters')
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    plt.savefig(os.path.join(figpath,'dynamic_modules_atlas'+'.pdf'))
    plt.savefig(os.path.join(figpath,'dynamic_modules_atlas'+'.png'))

def plot_stability(m,rule = 'delaygo', epoch = 'stim1', trial_n = 0, trained = True):

    model = Model(m)
    with tf.Session() as sess:
        model.restore()
        model._sigma=0
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        hp = model.hp
        trial = generate_trials(rule, hp, mode='random', noise_on=False, batch_size =100)
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict)

    fig = plt.figure(figsize=(17, 3))

    # Plot during task
    ax1 = plt.subplot(1,3,1)
    plt.plot(h_tf[:,0,:50],alpha = .5)
    plt.xlabel('time in trial',fontsize = 18)
    plt.ylabel('activations',fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    #Plot evolution of hidden state for extended time
    n_steps = 500
    x_t = np.matlib.repmat(trial.x[-1,trial_n,:],n_steps,1)
    h0 = h_tf[-1,trial_n,:]
    h_t = vanilla_run_with_h0(params, x_t, h0, hp)

    ax1 = plt.subplot(1,3,2)
    ax1.plot(h_t[:,:50],alpha = .5)
    plt.xlabel('timesteps',fontsize = 18)
    plt.ylabel('activations',fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    #Plot eigenspectrum
    inputs = x_t[:1,:]
    state = h_tf[-1:,trial_n,:]

    dFdx = calc_jacobian_at(m,inputs,state,lesion_units = [])
    evals, _ = LA.eig(dFdx[0])

    xs = np.linspace(-1, 1, 1000)
    ys = np.sqrt(1 - xs**2)

    ax2 = plt.subplot(1,3,3)

    ax2.plot(evals.real,evals.imag,'.k',alpha = .3,markerfacecolor = 'k')
    ax2.plot(xs, ys,':k',linewidth = 1)
    ax2.plot(xs, -ys,':k',linewidth = 1)
    ax2.plot([1,1],[-1,1],'-k',alpha = .3)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    eigenspectrum_axes(epoch,ax2)
    ax2.set_aspect('equal')
    tit = hp['activation'] + '_' + hp['w_rec_init'] + '_alpha_' + str(hp['alpha'])
    plt.title(tit,fontsize = 18)

    save_dir = os.path.join(m,'stability_figs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir,'eigenspec.png'))

from network import get_perf
from task import generate_trials
        
def lesions(m,rules=[],max_d = 3.5,criterion = 'distance'):

    method = 'ward'

    if criterion == 'distance':
        lesion_folder = 'lesion_fps_hierarchical_'+method+'_max_d'+str(max_d)
    else:
        lesion_folder = 'lesion_fps_hierarchical_'+method+'_'+criterion+'_max_d'+str(max_d)

    save_dir = os.path.join(m,lesion_folder)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cluster_var = np.load(os.path.join(save_dir,'cluster_var.npz'))
    lesion_units_list = cluster_var['lesion_units_list']

    perfs_store_list = list()
    perfs_changes = list()
    cost_store_list = list()
    cost_changes = list()

    for i, lesion_units in enumerate(lesion_units_list):
        print(lesion_units)
        model = Model(m)
        hp = model.hp
        if len(rules)==0:
            rules = hp['rule_trains']
        with tf.Session() as sess:
            model.restore()
            model.lesion_units(sess, lesion_units)

            perfs_store = list()
            cost_store = list()
            for rule in rules:
                n_rep = 16
                batch_size_test = 256
                batch_size_test_rep = int(batch_size_test / n_rep)
                clsq_tmp = list()
                perf_tmp = list()
                for i_rep in range(n_rep):
                    trial = generate_trials(rule, hp, 'random',
                                            batch_size=batch_size_test_rep)
                    feed_dict = tools.gen_feed_dict(model, trial, hp)
                    y_hat_test, c_lsq = sess.run(
                        [model.y_hat, model.cost_lsq], feed_dict=feed_dict)

                    # Cost is first summed over time, and averaged across batch and units
                    # We did the averaging over time through c_mask

                    # IMPORTANT CHANGES: take overall mean
                    perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
                    clsq_tmp.append(c_lsq)
                    perf_tmp.append(perf_test)

                perfs_store.append(np.mean(perf_tmp))
                cost_store.append(np.mean(clsq_tmp))

        perfs_store = np.array(perfs_store)
        cost_store = np.array(cost_store)

        perfs_store_list.append(perfs_store)
        cost_store_list.append(cost_store)

        if i > 0:
            perfs_changes.append(perfs_store - perfs_store_list[0])
            cost_changes.append(cost_store - cost_store_list[0])

    perfs_changes = np.array(perfs_changes)
    cost_changes = np.array(cost_changes)
    
    # save cluster variables
    lesion_var = {'perfs_changes':perfs_changes,
                'cost_changes':cost_changes}

    if criterion == 'distance':
        lesion_folder = 'lesion_fps_hierarchical_'+method+'_max_d'+str(max_d)
    else:
        lesion_folder = 'lesion_fps_hierarchical_'+method+'_'+criterion+'_max_d'+str(max_d)
    save_dir = os.path.join(m,lesion_folder)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez(os.path.join(save_dir,'lesion_var.npz'),**lesion_var)

    return perfs_changes, cost_changes