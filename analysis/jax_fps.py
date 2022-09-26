# Numpy, JAX, Matplotlib and h5py should all be correctly installed and on the python path.
from __future__ import print_function, division, absolute_import
import datetime
import h5py
import jax.numpy as np
from jax import jacrev, random, vmap
from jax.experimental import optimizers
from jax.ops import index, index_add, index_update
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as onp             # original CPU-backed NumPy
import os
import sys
import time
import tensorflow as tf
from scipy import stats
from sklearn import linear_model
from numpy import linalg as LA

import getpass
ui = getpass.getuser()
if ui == 'laura':
    p = '/home/laura'
elif ui == 'lauradriscoll':
    p = '/Users/lauradriscoll/Documents'

####################################### EDIT THIS
net = 'stepnet'
epoch = 'go1' # epoch = 2 #go1 epoch
rule = 'contextdelaydm1'
model_n = 0
dir_specific_all = 'stepnet/lowD/softplus/'#grad_norm_both/'#'lowD/combos'#'stepnet/lowD/tanh'#'lowD/grad_norm_l2001' #' #'lowD/armnet_noreg'#lowD/combos' ##grad_norm_l2h000001' /Documents/data/rnn/multitask/varGo/lowD/most/
model_dir = os.path.join(p,'data/rnn/multitask/',dir_specific_all,'most',str(model_n))
#######################################

HOME_DIR = '/home/laura/code/' 

sys.path.append(os.path.join(HOME_DIR,'computation-thru-dynamics'))
import fixed_point_finder.decision as decision
import fixed_point_finder.fixed_points as fp_optimize
import fixed_point_finder.rnn as rnn
import fixed_point_finder.utils as utils

PATH_YANGNET = os.path.join(p,'code/multitask-nets',net)

sys.path.insert(0, PATH_YANGNET)
from task import generate_trials, rule_name, rule_index_map
from network import Model
import tools
from tools_lnd import plot_N, plot_FP

def out_affine(params, h):
    return np.dot(params[2].T,h)+params[3]

def relu(x):
    f = x * (x > 0)
    return f

def rnn_vanilla(params, h, x):
    xh = np.concatenate([x,h], axis=0)
    gate_inputs = np.dot(params[0].T,xh)+params[1]
    noise = 0
    output = _activation(gate_inputs) # + noise

    h_new = (1-alpha) * h + alpha * output
    
    return h_new

batch_affine = vmap(out_affine, in_axes=(None, 0))

def vanilla_run_with_h0(params, x_t, h0):
    h = h0
    h_t = []
    h_t.append(h)
    for x in x_t:
        h = rnn_vanilla(params, np.squeeze(h), np.squeeze(x.T))
        h_t.append(np.expand_dims(h,axis=1))

    h_t = np.squeeze(np.array(h_t))  
    o_t = batch_affine(params, h_t)
    return h_t, o_t

## Define regression for each stimulus period
def generate_Beta_epoch(trial):
	Beta_epoch = {}

	for epoch in trial.epochs.keys():
	    T_use = trial.epochs[epoch][1], 
	    if T_use[0] is None:
	        T_use = np.shape(h_tf)
	    inds_use = range(0,np.shape(h_tf)[1],int(np.shape(h_tf)[1]/200)) ######### Maybe edit?
	    
	    X = h_tf[T_use[0]-1,inds_use,:].T
	    
	    y1 = np.sin(trial.stim_locs[inds_use,0:1])
	    y2 = np.cos(trial.stim_locs[inds_use,0:1])
	    y = onp.concatenate((y1,y2),axis=1)
	    
	    X_zscore = stats.zscore(X, axis=1)
	    X_zscore_nonan = X_zscore
	    X_zscore_nonan[np.isnan(X_zscore)] = 0
	    r = X_zscore_nonan

	    lm = linear_model.LinearRegression()
	    model = lm.fit(y,r.T)
	    Beta = model.coef_
	    Beta_epoch[epoch,1],_ = LA.qr(Beta)
	    
	    y1 = np.sin(trial.stim_locs[inds_use,1:2])
	    y2 = np.cos(trial.stim_locs[inds_use,1:2])
	    y = onp.concatenate((y1,y2),axis=1)
	    
	    X_zscore = stats.zscore(X, axis=1)
	    X_zscore_nonan = X_zscore
	    X_zscore_nonan[np.isnan(X_zscore)] = 0
	    r = X_zscore_nonan

	    lm = linear_model.LinearRegression()
	    model = lm.fit(y,r.T)
	    Beta = model.coef_
	    Beta_epoch[epoch,2],_ = LA.qr(Beta)
	    
	    return Beta_epoch

###########################

model = Model(model_dir)
with tf.Session() as sess:
    model.restore()
    # get all connection weights and biases as tensorflow variables
    var_list = model.var_list
    # evaluate the parameters after training
    params = [sess.run(var) for var in var_list]
    # get hparams
    hparams = model.hp
    # create a trial
    trial = generate_trials(rule, hparams, mode='test', noise_on=False, batch_size = 128, delay_fac =1)
    # get feed_dict
    feed_dict = tools.gen_feed_dict(model, trial, hparams)
    # run model
    h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)

    n_input = hparams['n_input']
    n_rnn = hparams['n_rnn']
    n_output = hparams['n_output']
    w_in = params[0]
    b_in = params[1]
    w_out = params[2]
    b_out = params[3]
    sigma_rec = hparams['sigma_rec']
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

####################################### EDIT THIS
T,B,N = np.shape(h_tf)
fp_epoch = 'stim1'
T = trial.epochs[fp_epoch][1]-1

x_star = trial.x[T-1,:,:]
x_star_want = trial.x[T-1,0,:]
x_star_want[1:5] = [0,0,0,-1]

trial_set = onp.where((x_star == x_star_want).all(axis=1))[0][0]

f0 = plt.figure(figsize=(12,6))
plt.plot(x_star[trial_set,:],'.')
plt.xlabel('input number')
plt.ylabel('input value')

#######################################

# starting points for fixed point search
fp_candidates = np.reshape(h_tf[T-1:T,trial_set,:], (-1, N))
fp_candidates = fp_candidates + .0001*onp.random.randn(500,N)

# These are some preliminaries. 
x_star = x_star[trial_set,:]  # Input at the end of the trial

# Make a one parameter function of thie hidden state, useful for jacobians.
rnn_fun = lambda h : rnn_vanilla(params, h, x_star)
batch_rnn_fun = vmap(rnn_fun, in_axes=(0,))

# Fixed point loss functions
fp_loss_fun = fp_optimize.get_fp_loss_fun(rnn_fun)
total_fp_loss_fun = fp_optimize.get_total_fp_loss_fun(rnn_fun)

# Fixed point optimization hyperparameters
fp_num_batches = 10000       # Total number of batches to train on.
fp_batch_size = 128          # How many examples in each batch
fp_step_size = 0.2           # initial learning rate
fp_decay_factor = 0.9999     # decay the learning rate this much
fp_decay_steps = 1           #
fp_adam_b1 = 0.9             # Adam parameters
fp_adam_b2 = 0.999
fp_adam_eps = 1e-5
fp_opt_print_every = 100   # Print training information during optimziation every so often

# Fixed point finding thresholds and other HPs
fp_noise_var = 0#0.0001      # Gaussian noise added to fixed point candidates before optimization.
fp_opt_stop_tol = 0.000001  # Stop optimizing when the average value of the batch is below this value.
fp_tol = 0.000001        # Discard fps with squared speed larger than this value.
fp_unique_tol = 0.025   # tolerance for determination of identical fixed points
fp_outlier_tol = 1.0    # Anypoint whos closest fixed point is greater than tol is an outlier.

reload(fp_optimize)

fp_tols = [0.000001, 0.0000001, 0.00000005] # Used for both fp_tol and opt_stop_tol

all_fps = {}
for tol in fp_tols:
    fp_hps = {'num_batches' : fp_num_batches, 
              'step_size' : fp_step_size, 
              'decay_factor' : fp_decay_factor, 
              'decay_steps' : fp_decay_steps, 
              'adam_b1' : fp_adam_b1, 'adam_b2' : fp_adam_b2, 'adam_eps' : fp_adam_eps,
              'noise_var' : fp_noise_var, 
              'fp_opt_stop_tol' : tol, 
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
        
    all_fps[tol] = {'fps' : fps, 'candidates' : fp_candidates[fp_idxs],
                    'losses' : fp_losses, 'F_of_fps' : F_of_fps, 
                    'opt_details' : fp_opt_details, 'hps' : fp_hps}

    all_fps[tol]

################################################################

f1 = plt.figure(figsize=(12,6))

for tol in fp_tols: 
    plt.semilogy(all_fps[tol]['losses'],'.'); 
    plt.xlabel('Fixed point #')
    plt.ylabel('Fixed point loss');
plt.legend(fp_tols)
plt.title('Fixed point loss by fixed point (sorted) and stop tolerance')

f2 = plt.figure(figsize=(12,4))

pidx = 1
nfp_tols = len(fp_tols)
for tol_idx, tol in enumerate(fp_tols):
    plt.subplot(1, nfp_tols, pidx); pidx += 1
    plt.hist(onp.log10(fp_loss_fun(all_fps[tol]['fps'])), 50);
    plt.xlabel('log10(FP loss)')
    plt.title('Tolerance: ' + str(tol));

######################################################
# Sort the best fixed points by projection onto the readout.
if np.shape(all_fps[tol]['fps'])[0]==1:
    fp_readouts = onp.squeeze(onp.dot(params[2].T,all_fps[tol]['fps'].T)+onp.expand_dims(params[3], axis=1))
    sorted_fp_readouts = fp_readouts[1]
    sorted_fps = all_fps[tol]['fps']

    jacs = fp_optimize.compute_jacobians(rnn_fun, sorted_fps)
    eig_decomps = fp_optimize.compute_eigenvalue_decomposition(jacs, sort_by='real', do_compute_lefts=True)
else:
    fp_readouts = onp.squeeze(onp.dot(params[2].T,all_fps[tol]['fps'].T)+onp.expand_dims(params[3], axis=1))
    fp_ro_sidxs = onp.argsort(fp_readouts[1,:])
    sorted_fp_readouts = fp_readouts[1,fp_ro_sidxs]
    sorted_fps = all_fps[tol]['fps'][fp_ro_sidxs]

    downsample_fps = 1 # Use this if too many fps
    sorted_fp_readouts = sorted_fp_readouts[0:-1:downsample_fps]
    sorted_fps = sorted_fps[0:-1:downsample_fps]
    jacs = fp_optimize.compute_jacobians(rnn_fun, sorted_fps)
    eig_decomps = fp_optimize.compute_eigenvalue_decomposition(jacs, sort_by='real', do_compute_lefts=True)

###########################################
n=N
from sklearn.decomposition import PCA
cmap=plt.get_cmap('rainbow')
f3 = plt.figure(figsize=(16,16));
ax = f3.add_subplot(111, projection='3d');

T = trial.epochs['stim1'][1]-1
trial_set_plot = range(0,np.shape(h_tf)[1],int(np.shape(h_tf)[1]/10))
hiddens = np.reshape(h_tf[:T,:,:], (-1, N)) # now batch * time x dim
pca = PCA(n_components=30).fit(hiddens)

max_fps_to_plot = 100
sizes = [100, 500, 800]
for tol, size in zip(fp_tols[0:3], sizes):
    
    for t in trial_set_plot:
        hiddens = np.reshape(h_tf[:T,t,:], (-1, N)) # now batch * time x dim
        h_pca = pca.transform(hiddens)
        ax.scatter(h_pca[:,0], h_pca[:,1], h_pca[:,2], color=cmap(t/max(trial_set_plot)), s=10)

    traj = pca.transform(np.reshape(h_tf[:trial.epochs['fix1'][1],trial_set,:], (-1, N)))
    ax.plot3D(traj[:,0], traj[:,1], traj[:,2], '-', color='k')
    ax.plot3D(traj[0:1,0], traj[0:1,1], traj[0:1,2], 'd', color='k')

    hstars = np.reshape(all_fps[tol]['fps'], (-1, n))
    hstar_pca = pca.transform(hstars) 
    
    marker_style = dict(marker='*', s=size, edgecolor='gray')
    
    ax.scatter(hstar_pca[:,0], hstar_pca[:,1], hstar_pca[:,2], 
                'k', color=cmap(trial_set/max(trial_set_plot)), **marker_style);
        
plt.title('Fixed point structure and neural states.');
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3');

##############################################
f4 = plt.figure(figsize=(5, 9))

plt.subplot(3,1,1)
for decomp in eig_decomps:
    plt.scatter(decomp['evals'].real, decomp['evals'].imag, marker = '.', alpha = .01)
    plt.title('All')
    plt.ylabel('Imaginary')
    plt.xlabel('Real')

    plt.subplot(3,1,2)
    plt.scatter(decomp['evals'].real, decomp['evals'].imag, marker = '.', alpha = .5)
    plt.title('Single')
    plt.ylabel('Imaginary')
    plt.xlabel('Real')

    plt.subplot(3,1,3)
    plt.plot(onp.sort(decomp['evals'].real),'.')
    plt.title('Single')
    plt.ylabel('Real Sorted')
    plt.xlabel('Eigenvalue #')

################################################

f5 = plt.figure(figsize=(5, 5))
ldots_mod1 = []
ldots_mod2 = []
rdots = []
rdotla = []  
evals = []

x = trial.x[T,trial_set,:]
nfps = len(sorted_fps)

for jidx in range(nfps):
    fp = sorted_fps[jidx, :]
    rnn_fun_x = lambda x : rnn_vanilla(params, fp, x)
    dfdx = jacrev(rnn_fun_x)
    
    for idx in onp.argwhere(eig_decomps[jidx]['evals']>.95):
        ii = idx[0]
        r0 = onp.real(eig_decomps[jidx]['R'][:, ii])                          
        rdots.append(onp.dot(r0, params[2][:,2]))
        l0 = onp.real(eig_decomps[jidx]['L'][:, ii])
        ldots_mod1.append(onp.dot(l0, dfdx(x)[:,2]))
        ldots_mod2.append(onp.dot(l0, dfdx(x)[:,4]))
        evals.append(eig_decomps[jidx]['evals'][ii].real)

plt.figure(figsize=(4,4))
plt.subplot(111)
plt.scatter(rdots, onp.abs(ldots_mod1), np.power(np.power(100,evals),evals), c='g', label = 'mod1', alpha = .5)
plt.scatter(rdots, onp.abs(ldots_mod2), np.power(np.power(100,evals),evals), c='k', label = 'mod2', alpha = .5)
plt.legend()
plt.title(rule)
plt.ylabel('l0 dFdh . dfdx(cos@)')
plt.xlabel('r0 dFdh . W_out(cos@)')

plt.savefig('jax_fps/eig_decomp/' + rule + '_.svg')
plt.show()