# Numpy, JAX, Matplotlib and h5py should all be correctly installed and on the python path.
from __future__ import print_function, division, absolute_import
import datetime
import h5py
import jax.numpy as np
from jax import jacrev, random, vmap
from jax.experimental import optimizers
from jax.ops import index, index_add, index_update
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as onp             # original CPU-backed NumPy
import os
import sys
import time
import tensorflow as tf
from scipy import stats
#from sklearn import linear_model
from numpy import linalg as LA
import numpy.random as npr

import getpass
ui = getpass.getuser()
if ui == 'laura':
    p = '/home/laura'
elif ui == 'lauradriscoll':
    p = '/Users/lauradriscoll/Documents'
elif ui == 'lndrisco':
    p = '/home/users/lndrisco/'

sys.path.append(os.path.join(p,'code','computation-thru-dynamics'))
import fixed_point_finder.decision as decision
import fixed_point_finder.fixed_points as fp_optimize
import fixed_point_finder.rnn as rnn
import fixed_point_finder.utils as utils


########### Edit This ###########
net = 'stepnet'
model_n = 0
dir_specific_all = 'crystals/softplus/l2h00001'#'crystals/softplus/no_reg'#''stepnet/crystals/softplus/'#grad_norm_both/'#'lowD/combos'#'stepnet/lowD/tanh'#'lowD/grad_norm_l2001' #' #'lowD/armnet_noreg'#lowD/combos' ##grad_norm_l2h000001' /Documents/data/rnn/multitask/varGo/lowD/most/
model_dir_all = os.path.join(p,'data/rnn/multitask/',dir_specific_all,str(model_n))

# Fixed point optimization hyperparameters
fp_num_batches = 50000       # Total number of batches to train on.
fp_batch_size = 128          # How many examples in each batch
fp_step_size = .5          # initial learning rate
fp_decay_factor = 0.99999     # decay the learning rate this much
fp_decay_steps = 1           #
fp_adam_b1 = 0.9             # Adam parameters
fp_adam_b2 = 0.999
fp_adam_eps = 1e-5
fp_opt_print_every = 1000   # Print training information during optimziation every so often

# Fixed point finding thresholds and other HPs
fp_noise_var = 0.001      # Gaussian noise added to fixed point candidates before optimization.
fp_opt_stop_tol = 0.000001  # Stop optimizing when the average value of the batch is below this value.
fp_tol = 0.000001        # Discard fps with squared speed larger than this value.
fp_unique_tol = 0.025   # tolerance for determination of identical fixed points
fp_outlier_tol = 1.0    # Anypoint whos closest fixed point is greater than tol is an outlier.

#################################

PATH_YANGNET = os.path.join(p,'code/multitask-nets',net)

sys.path.insert(0, PATH_YANGNET)
from task import generate_trials, rule_name, rule_index_map, rules_dict
from network import Model
import tools
from tools_lnd import name_best_ckpt

ckpt_n = name_best_ckpt(model_dir_all,'multidelaydm')
ckpt_n_dir = os.path.join(model_dir_all,'ckpts/model.ckpt-' + str(int(ckpt_n)))

model = Model(model_dir_all)
with tf.Session() as sess:
    model.saver.restore(sess,ckpt_n_dir)
    model._sigma=0
    var_list = model.var_list
    params = [sess.run(var) for var in var_list]
    hparams = model.hp
    
    n_input = hparams['n_input']
    n_rnn = hparams['n_rnn']
    n_output = hparams['n_output']
    w_in = params[0]
    b_in = params[1]
    w_out = params[2]
    b_out = params[3]
    sigma_rec = 0#hparams['sigma_rec']
    dt = hparams['dt']
    tau = hparams['tau']
    alpha = dt/tau
    activation = hparams['activation']

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

def out_affine(params, h):
    return np.dot(params[2].T,h)+params[3]

def relu(x):
    f = x * (x > 0)
    return f

def rnn_vanilla(params, h, x, alpha):
    xh = np.concatenate([x,h], axis=0)
    gate_inputs = np.dot(params[0].T,xh)+params[1]
    noise = 0
    output = _activation(gate_inputs) # + noise

    h_new = (1-alpha) * h + alpha * output
    
    return h_new

batch_affine = vmap(out_affine, in_axes=(None, 0))

def vanilla_run_with_h0(params, x_t, h0, alpha):
    h = h0
    h_t = []
    h_t.append(h)
    for x in x_t:
        h = rnn_vanilla(params, np.squeeze(h), np.squeeze(x.T), alpha)
        h_t.append(np.expand_dims(h,axis=1))

    h_t = np.squeeze(np.array(h_t))  
    o_t = batch_affine(params, h_t)
    return h_t, o_t

reload(fp_optimize)

fp_tols = [1e-7] # Used for both fp_tol and opt_stop_tol

model = Model(model_dir_all)
with tf.Session() as sess:
    model.saver.restore(sess,ckpt_n_dir)
    model._sigma=0
    var_list = model.var_list
    params = [sess.run(var) for var in var_list]
    hparams = model.hp

    for rule in hparams['rule_trains']:
        
        fldr = os.path.join(model_dir_all,'fps',rule)
        if not os.path.exists(fldr):
            os.makedirs(fldr)

        trial = generate_trials(rule, hparams, mode='test', noise_on=False, batch_size = 128, delay_fac =1)
        feed_dict = tools.gen_feed_dict(model, trial, hparams)
        h_tf = sess.run(model.h, feed_dict=feed_dict) #(n_time, n_condition, n_neuron)

        for fp_epoch in ['stim1']:

                # get trial shape
            _,B,N = np.shape(h_tf)
            trial_set = range(10,B)

            # Fixed point epoch
            T = trial.epochs[fp_epoch][1], 
            if T[0] is None:
                T = np.shape(trial.x)[0]

            for trial_ind in trial_set:
                # Input during fixed point epoch
                x_star = trial.x[T[0]-1,trial_ind,:]

                # Make a one parameter function of thie hidden state, useful for jacobians.
                rnn_fun = lambda h : rnn_vanilla(params, h, x_star, alpha)
                batch_rnn_fun = vmap(rnn_fun, in_axes=(0,))

                fp_candidates = np.transpose(h_tf,(1,0,2))  # was batch x time x dim
                a = np.reshape(fp_candidates[range(0,10),trial.epochs['fix1'][1]-1,:], (-1, N)) # now batch * time x dim
                b = np.reshape(fp_candidates[trial_set,trial.epochs[fp_epoch][0]:trial.epochs[fp_epoch][1]:2,:], (-1, N)) # now batch * time x dim
                fp_candidates = np.concatenate((a, b), axis=0)

                all_fps = {}
                for tol_ind in range(len(fp_tols)):
                    tol = fp_tols[tol_ind]
                    fp_hps = {'num_batches' : fp_num_batches, 
                              'step_size' : fp_step_size, 
                              'decay_factor' : fp_decay_factor, 
                              'decay_steps' : fp_decay_steps, 
                              'adam_b1' : fp_adam_b1, 'adam_b2' : fp_adam_b2, 'adam_eps' : fp_adam_eps,
                              'noise_var' : fp_noise_var, 
                              'fp_opt_stop_tol' : 0, 
                              'fp_tol' : tol, 
                              'unique_tol' : fp_unique_tol, 
                              'outlier_tol' : fp_outlier_tol, 
                              'opt_print_every' : fp_opt_print_every}
                    fps, fp_losses, fp_idxs, fp_opt_details = \
                        fp_optimize.find_fixed_points(rnn_fun, fp_candidates, fp_hps, do_print=True)
                    if len(fp_idxs) > 0:
                        F_of_fps = batch_rnn_fun(fps)
                    else:
                        F_of_fps = onp.zeros([0,N])

                    all_fps[tol_ind] = {'fps' : fps, 'candidates' : fp_candidates[fp_idxs],
                                    'losses' : fp_losses, 'F_of_fps' : F_of_fps, 
                                    'opt_details' : fp_opt_details, 'hps' : fp_hps, 'tol' : tol, 'input': x_star}

                if len(all_fps[tol_ind]['losses'])>0:
                    onp.savez(fldr +'/'+ fp_epoch + '_' + str(int(np.log10(tol))) + '_' + str(trial_ind)+'.npz', **all_fps[tol_ind])

