from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import numpy as np
import numpy.random as npr
import tensorflow as tf
import sys
import pickle
import matplotlib.pyplot as plt
import pdb
import getpass

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
from task import generate_trials, rule_name, rule_index_map
from network import FixedPoint_Model
import tools
from tools_lnd import plot_N, plot_FP, name_best_ckpt

PATH_TO_RECURRENT_WHISPERER = p+'/code/recurrent-whisperer'#'/home/laura/code/recurrent-whisperer'#
sys.path.insert(0, PATH_TO_RECURRENT_WHISPERER)
from RecurrentWhisperer import RecurrentWhisperer

PATH_TO_FIXED_POINT_FINDER = p+'/code/fixed-point-finder' #'/home/laura/code/fixed-point-finder-experimental'#
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinder import FixedPointFinder

##################################################################
#Find right model dir
##################################################################
all_dir = 'data/rnn/sequential_multiTask_RNN/trained_nets_lea/sequential_tasks/ours/'
tit = 'sequential'
letter_set = ['a','b','c','d']
NOISE_SCALE = 0.05 #0.5 # Standard deviation of noise added to initial states
N_INITS = 1000 # The number of initial states to provide


task_list = ['delaygo', 'fdgo', 'reactgo', 'fdanti', 'reactanti', 'delayanti',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

##################################################################
def project2d(x,axes):

    if x.ndim>1:
        n_steps = x.shape[1]
    else:
        n_steps = 1

    z = np.zeros((n_steps,2))
    z[:,0] = np.dot(axes[:,0],x)
    z[:,1] = np.dot(axes[:,1],x)
    return z

def add_unique_to_inputs_list(dict_list, key, value):
    for d in range(len(dict_list)):
        if (dict_list.values()[d]==value).all():
            return False, dict_list

    dict_list.update({key : value})
    return True, dict_list

def get_filename(trial, epoch,t):
    ind_stim_loc  = 180*trial.y_loc[-1,t]/np.pi
    filename = trial.epochs.keys()[epoch]+'_'+str(round(ind_stim_loc,2))

    return filename, ind_stim_loc

for n_letter in letter_set:

    for n_number in range(5):

        if tit == 'sequential': 
            model_dir_all = os.path.join(p,all_dir,'goantiset',n_letter,str(n_number))
        else:
            model_dir_all = os.path.join(p,all_dir,'goantiset',str(n_number))

        for rule in task_list:
            model = FixedPoint_Model(model_dir_all)
            with tf.Session() as sess:
                model.restore()
                model._sigma=0
                # get all connection weights and biases as tensorflow variables
                var_list = model.var_list
                # evaluate the parameters after training
                params = [sess.run(var) for var in var_list]
                # get hparams
                hparams = model.hp
                # create a trial
                trial = generate_trials(rule, hparams, mode='test', noise_on=False, batch_size=40)# get feed_dict
                feed_dict = tools.gen_feed_dict(model, trial, hparams)
                # run model
                h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)  

                ##################################################################
                # get shapes   
                n_steps, n_trials, n_input_dim = np.shape(trial.x)
                n_rnn = np.shape(h_tf)[2]
                n_output = np.shape(y_hat_tf)[2]

                # Fixed point finder hyperparameters
                # See FixedPointFinder.py for detailed descriptions of available
                # hyperparameters.
                fpf_hps = {}
                alr_dict = ({'decrease_factor' : .95, 'initial_rate' : 1})

                n_epochs = len(trial.epochs)
                for epoch in range(n_epochs):
                    e_start = max([0, trial.epochs.values()[epoch][0]])
                    end_set = [n_steps, trial.epochs.values()[epoch][1]]
                    e_end = min(x for x in end_set if x is not None)

                    n_inputs = 0
                    input_set = {str(n_inputs) : np.zeros((1,n_input_dim))}

                    for t in [0,]:#range(0,n_trials,int(n_trials/10)):#range(1,n_trials): np.arange(0, 40, 8): #

                        inputs = np.squeeze(trial.x[e_start,t,:])
                        inputs = inputs[np.newaxis,:]
                        inputs_big = inputs[np.newaxis,:]

                        unique_input, input_set = add_unique_to_inputs_list(input_set, str(n_inputs), inputs)
                        
                        if unique_input:
                            n_inputs+=1
                            input_set[str(n_inputs)] = inputs

                            fpf = []
                            fpf = FixedPointFinder(model.cell,sess, alr_hps=alr_dict, method='joint', verbose = True, **fpf_hps) #do_compute_input_jacobians = True , q_tol = 1e-1, do_q_tol = True

                            example_predictions = {'state': np.transpose(h_tf,(1,0,2)), #[0:90,0:1,:]
                                                    'output': np.transpose(y_hat_tf,(1,0,2))}
                            
                            initial_states = fpf.sample_states(example_predictions['state'][:,:,:], #specify T inds removed e_start:e_end
                                                            n_inits=N_INITS,
                                                            noise_scale=NOISE_SCALE)
                            # Run the fixed point finder
                            unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

                            if unique_fps.xstar.shape[0]>0:

                                all_fps = {}
                                all_fps = {'xstar':unique_fps.xstar,
                                    # 'J_inputs':unique_fps.J_inputs, 
                                    'J_xstar':unique_fps.J_xstar, 
                                    'qstar':unique_fps.qstar, 
                                    'inputs':unique_fps.inputs, 
                                    'epoch_inds':range(e_start,e_end),
                                    'noise_var':NOISE_SCALE,
                                    'state_traj':example_predictions['state'],
                                    'out_dir':180*trial.y_loc[-1,t]/np.pi}

                                save_dir = os.path.join(model_dir_all,'tf_fixed_pts_all_init',rule)
                                filename, ind_stim_loc = get_filename(trial, epoch, t)

                                if not os.path.exists(save_dir):
                                    os.makedirs(save_dir)
                                np.savez(os.path.join(save_dir,filename+'.npz'),**all_fps)
