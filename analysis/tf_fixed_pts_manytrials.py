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
import pdb
import getpass

ui = getpass.getuser()
if ui == 'laura':
    p = '/home/laura'
elif ui == 'lauradriscoll':
    p = '/Users/lauradriscoll/Documents'
elif ui == 'lndrisco':
    p = '/home/users/lndrisco'

net = 'binary_inputs'
PATH_YANGNET = os.path.join(p,'code/sophie-nets',net) 

sys.path.insert(0, PATH_YANGNET)
from task import generate_trials, rule_name, rule_index_map, rules_dict
from network import FixedPoint_Model
import tools

PATH_TO_RECURRENT_WHISPERER = p+'/code/recurrent-whisperer'#'/home/laura/code/recurrent-whisperer'#
sys.path.insert(0, PATH_TO_RECURRENT_WHISPERER)
from RecurrentWhisperer import RecurrentWhisperer

PATH_TO_FIXED_POINT_FINDER = p+'/code/fixed-point-finder' #'/home/laura/code/fixed-point-finder-experimental'#
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinder import FixedPointFinder

script_name = os.path.basename(sys.argv[0])[:-3]

##################################################################
#Find right model dir
##################################################################

NOISE_SCALE = 0.05 #0.5 # Standard deviation of noise added to initial states
N_INITS = 1000 # The number of initial states to provide

# # all_dir = 'data/rnn/multitask/stepnet/crystals/softplus/'
# # # all_dir = '/Users/lauradriscoll/Documents/data/rnn/multitask/stepnet/crystals/softplus/4_tasks/'
# # # all_dir = '/Users/lauradriscoll/Documents/data/rnn/multitask/stepnet/crystals/softplus'
# file_spec = 'l2w0001'
# data_folder = 'data/rnn/multitask/stepnet/crystals/softplus'
# # file_spec = 'transfer_learn/start_from_best/14_to_1_tasks/256_n_rnn/l2_w4_h5_delayanti/'

##################################################################
#Find right model dir
##################################################################

# rule_trains = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
#           'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
#           'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

rnn_type = 'LeakyRNN'
activation = 'softplus'
w_init = 'randgauss'
ruleset = 'basic'
n_rnn = str(256)
l2w = -7.0
l2h = -7.0
l1w = 0
l1h = 0
lr = -7.0
seed = str(0)
rule_trains = [rules_dict['basic'][1],rules_dict['basic'][3]]
rule_trains_str = '_'.join(rule_trains)
sigma_rec = 1/20
sigma_x = 2/20
w_rec_coeff = .9

net_name = 'lr'+"{:.1f}".format(-lr)+'l2_w'+"{:.1f}".format(-l2w)+'_h'+"{:.1f}".format(-l2h)
net_name2 = '_sig_rec'+str(sigma_rec)+'_sig_x'+str(sigma_x)+'_w_rec_coeff'+"{:.1f}".format(w_rec_coeff)+'_'+rule_trains_str

dir_specific_all = os.path.join(ruleset,rnn_type,activation,
    w_init,str(len(rule_trains))+'_tasks',str(n_rnn)+'_n_rnn',net_name+net_name2)

m = os.path.join(p,'data','sophie-nets',net,'data',dir_specific_all,str(seed))

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
    ind_stim_loc  = str(int(180*trial.y_loc[-1,t]/np.pi))
    nonzero_stim = trial.stim_locs[t,:]<100
    stim_names = '_'.join(str(int(180*x/np.pi)) for x in trial.stim_locs[t,nonzero_stim])
    filename = trial.epochs.keys()[epoch]+'_trial'+str(t)+'_x'+stim_names+'_y'+ind_stim_loc

    return filename, ind_stim_loc

for rule in ['delaygo','delayanti']:

    model = FixedPoint_Model(m)

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
        fpf_hps = {'tol_q': 1e-6,'tol_unique': 1e-1}
        alr_dict = ({'decrease_factor' : .95, 'initial_rate' : 1})

        n_epochs = len(trial.epochs)
        for epoch in range(n_epochs):
            e_start = max([0, trial.epochs.values()[epoch][0]])
            end_set = [n_steps, trial.epochs.values()[epoch][1]]
            e_end = min(x for x in end_set if x is not None)
            e_diff = e_end - e_start
            e_ind = int(e_diff/2)

            n_inputs = 0
            input_set = {str(n_inputs) : np.zeros((1,n_input_dim))}

            for t in [0,]:#range(int(n_trials/8),int(n_trials),int(n_trials/8)):#[int(n_trials/2),]:#:2,n_trials,int(n_trials/4)

                inputs = np.squeeze(trial.x[e_start+e_ind,t,:])
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

                        save_dir = os.path.join(m,script_name,rule)
                        filename, ind_stim_loc = get_filename(trial, epoch, t)

                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        np.savez(os.path.join(save_dir,filename+'.npz'),**all_fps)
