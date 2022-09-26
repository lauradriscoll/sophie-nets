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

net = 'stepnet'
PATH_YANGNET = os.path.join(p,'code/multitask-nets',net)

sys.path.insert(0, PATH_YANGNET)
from task import generate_trials, rule_name, rule_index_map, rules_dict
from network import FixedPoint_Model
import tools
from tools_lnd import gen_trials_from_model_dir, make_cat_h_rules, same_mov_inds

PATH_TO_RECURRENT_WHISPERER = p+'/code/recurrent-whisperer'#'/home/laura/code/recurrent-whisperer'#
sys.path.insert(0, PATH_TO_RECURRENT_WHISPERER)
from RecurrentWhisperer import RecurrentWhisperer

PATH_TO_FIXED_POINT_FINDER = p+'/code/fixed-point-finder' #'/home/laura/code/fixed-point-finder-experimental'#
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinder import FixedPointFinder

def add_unique_to_inputs_list(dict_list, key, value):
    for d in range(len(dict_list)):
        if (dict_list.values()[d]==value).all():
            return False, dict_list

    dict_list.update({key : value})
    return True, dict_list

def get_filename(trial1,trial2,epoch_list,t_set):
    n_stim_per_ring = int(np.shape(trial1.y)[2]-1)
    stim_size = int(2*n_stim_per_ring+1)

    rule1 = rules_dict['all'][np.argmax(trial1.x[0,0,stim_size:])]
    rule2 = rules_dict['all'][np.argmax(trial2.x[0,0,stim_size:])]
    ind_stim_loc1  = 180*trial1.y_loc[-1,t_set[0]]/np.pi
    ind_stim_loc2  = 180*trial2.y_loc[-1,t_set[1]]/np.pi
    filename = rule1+'_'+rule2+'_'+'_'.join(epoch_list)+'_x'+str(round(ind_stim_loc1,2))+'_x'+str(round(ind_stim_loc2,2))

    return filename
##################################################################
#Find right model dir
##################################################################


# rule_trains = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
#           'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
#           'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']##################

rnn_type = 'LeakyRNN'
activation = 'softplus'
w_init = 'diag'
ruleset = 'all'
rule_trains = rules_dict[ruleset]
n_tasks = str(len(rule_trains))
rule_trains_str = '_'.join(rule_trains)
n_rnn = str(128)
l2w = -6
l2h = -6
l1w = 0
l1h = 0
seed = '1'
lr = -6
sigma_rec = 1/20
sigma_x = 2/20
w_rec_coeff  = 8/10
data_folder_all = 'data/rnn/multitask/stepnet/'

net_name = 'lr'+"{:.1f}".format(-lr)+'l2_w'+"{:.1f}".format(-l2w)+'_h'+"{:.1f}".format(-l2h)
net_name2 = '_sig_rec'+str(sigma_rec)+'_sig_x'+str(sigma_x)+'_w_rec_coeff'+"{:.1f}".format(w_rec_coeff)+'_'+rule_trains_str

dir_specific_all = os.path.join('final1',ruleset,rnn_type,activation,
    w_init,str(len(rule_trains))+'_tasks',str(n_rnn)+'_n_rnn',net_name+net_name2)

model_dir_all = os.path.join(p,'data','rnn','multitask',net,dir_specific_all,str(seed))

epoch_list = ['delay1','go1']#['delay1','delay1'] #
r1 = 5
r2 = 5

rule1 = rule_trains[r1]
rule2 = rule_trains[r2]


t_set = [0,0]
h_cat = make_cat_h_rules(model_dir_all,task_set = [rule1,rule2]).T

##################################################################

##################################################################
#Run fixed pt finder
##################################################################

'''Initial states are sampled from states observed during realistic behavior
of the network. Because a well-trained network transitions instantaneously
from one stable state to another, observed networks states spend little if any
time near the unstable fixed points. In order to identify ALL fixed points,
noise must be added to the initial states before handing them to the fixed
point finder.'''
NOISE_SCALE = 0.01 #0.01 #0.5 # Standard deviation of noise added to initial states
N_INITS = 1000 # The number of initial states to provide
n_interp = 20 # number of steps between input conditions

trial1 = gen_trials_from_model_dir(model_dir_all,rule1,mode='test',noise_on = False)
trial2 = gen_trials_from_model_dir(model_dir_all,rule2,mode='test',noise_on = False)
trial2 = same_mov_inds(trial1, trial2) 
trial1 = gen_trials_from_model_dir(model_dir_all,rule1,mode='test',noise_on = False)

model = FixedPoint_Model(model_dir_all)
with tf.Session() as sess:
    print(model_dir_all)
    model.restore()
    model._sigma=0
    # get all connection weights and biases as tensorflow variables
    var_list = model.var_list
    # evaluate the parameters after training
    params = [sess.run(var) for var in var_list]
    # get hparams
    hparams = model.hp

    feed_dict1 = tools.gen_feed_dict(model, trial1, hparams)
    h_tf1, y_hat_tf1 = sess.run([model.h, model.y_hat], feed_dict=feed_dict1) #(n_time, n_condition, n_neuron) 

    feed_dict2 = tools.gen_feed_dict(model, trial2, hparams)
    h_tf2, y_hat_tf2 = sess.run([model.h, model.y_hat], feed_dict=feed_dict2) #(n_time, n_condition, n_neuron) 

    ##################################################################
    # get shapes   
    n_steps, n_trials, n_input_dim = np.shape(trial1.x)
    n_rnn = np.shape(h_tf1)[2]
    n_output = np.shape(y_hat_tf1)[2]

    # Fixed point finder hyperparameters
    # See FixedPointFinder.py for detailed descriptions of available
    # hyperparameters.
    fpf_hps = {'tol_q': 1e-6,'tol_unique': 1e-2}
    alr_dict = ({'decrease_factor' : .95, 'initial_rate' : 1})

    trial_set = [trial1, trial2]
    e_lims = np.zeros((2,len(trial_set)))

    for ti in range(len(trial_set)):
        trial = trial_set[ti]
        epoch = epoch_list[ti]
        e_start = max([0, trial.epochs[epoch][0]])
        e_lims[0,ti] = int(e_start)
        end_set = [np.shape(trial.x)[0], trial.epochs[epoch][1]]
        e_end = min(x for x in end_set if x is not None)
        e_lims[1,ti] = int(e_end)

    n_inputs = 0
    input_set = {str(n_inputs) : np.zeros((1,n_input_dim))}

    example_predictions = {'state': h_cat[np.newaxis,:,:]}

    fp_predictions = []

    inputs_1 = trial1.x[int(e_lims[0,0]+1),t_set[0],:]
    inputs_2 = trial2.x[int(e_lims[0,1]+1),t_set[1],:]
    del_inputs = inputs_2 - inputs_1

    for step_i in range(n_interp):

        step_inputs = inputs_1[np.newaxis,:]+del_inputs[np.newaxis,:]*(step_i/n_interp)
        inputs = step_inputs

        unique_input, input_set = add_unique_to_inputs_list(input_set, str(n_inputs), inputs)
        
        n_inputs+=1
        input_set[str(n_inputs)] = inputs

        fpf = []
        fpf = FixedPointFinder(model.cell, sess, alr_hps=alr_dict, method='joint', verbose = False, **fpf_hps)
        
        if np.shape(fp_predictions)[0]==0:
            fp_predictions = fpf.sample_states(example_predictions['state'],
                                        n_inits=N_INITS,
                                        noise_scale=NOISE_SCALE)

        unique_fps, all_fps = fpf.find_fixed_points(fp_predictions, inputs)


        if unique_fps.xstar.shape[0]>0:

            cat_fp_h = np.concatenate((h_cat[np.newaxis,:,:],unique_fps.xstar[np.newaxis,:,:]),axis=1)
            fp_predictions = fpf.sample_states(cat_fp_h,n_inits=N_INITS,noise_scale=NOISE_SCALE)

            script_name = os.path.basename(sys.argv[0])[:-3]
            save_dir = os.path.join(model_dir_all,script_name,rule1+'_'+rule2,'tol_q_e_'+str(-np.log10(fpf_hps['tol_q'])))
            filename = get_filename(trial1,trial2,epoch_list,t_set)

            all_fps = {}
            all_fps = {'xstar':unique_fps.xstar,
                'J_xstar':unique_fps.J_xstar, 
                'qstar':unique_fps.qstar, 
                'inputs':unique_fps.inputs, 
                'epoch_inds':range(e_start,e_end),
                'noise_var':NOISE_SCALE,
                'trial_num':t_set,
                'state_traj':example_predictions['state']}

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.savez(os.path.join(save_dir,filename+'_step_'+str(step_i)+'.npz'),**all_fps)
