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
from task import generate_trials, rule_name, rule_index_map, rules_dict
from network import FixedPoint_Model
import tools
from tools_lnd import gen_trials_from_model_dir, make_cat_h_rules

PATH_TO_RECURRENT_WHISPERER = p+'/code/recurrent-whisperer'#'/home/laura/code/recurrent-whisperer'#
sys.path.insert(0, PATH_TO_RECURRENT_WHISPERER)
from RecurrentWhisperer import RecurrentWhisperer

PATH_TO_FIXED_POINT_FINDER = p+'/code/fixed-point-finder' #'/home/laura/code/fixed-point-finder-experimental'#
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinder import FixedPointFinder

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

def get_filename(trial,epoch,t_set):
    n_stim_per_ring = int(np.shape(trial.y)[2]-1)
    stim_size = int(2*n_stim_per_ring+1)

    rule = rules_dict['all'][np.argmax(trial.x[0,0,stim_size:])]

    ind_stim_loc1  = 180*trial.y_loc[-1,t_set[0]]/np.pi
    ind_stim_loc2  = 180*trial.y_loc[-1,t_set[1]]/np.pi

    filename = rule+'_'+epoch+'_'+str(round(ind_stim_loc1,2))+'_'+str(round(ind_stim_loc2,2))

    return filename
##################################################################
#Find right model dir
##################################################################
for model_n in [0,]:
    # dir_specific_all = 'stepnet/crystals/softplus/4_tasks/l2w0001_fdgo_fdanti_delaygo_delayanti'#'crystals_no_noise/softplus/l2w0001/'#'crystals/softplus/no_reg'#''crystals/softplus/l2h00001'#'stepnet/crystals/softplus/'#grad_norm_both/'#'lowD/combos'#'stepnet/lowD/tanh'#'lowD/grad_norm_l2001' #' #'lowD/armnet_noreg'#lowD/combos' ##grad_norm_l2h000001' /Documents/data/rnn/multitask/varGo/lowD/most/
    # dir_specific_all = 'stepnet/crystals/softplus/two_tasks/l2w0001_delaygo_delayanti/'
    dir_specific_all = 'stepnet/crystals/softplus/l2w0001'
    model_dir_all = os.path.join(p,'data/rnn/multitask/',dir_specific_all,str(model_n))

    h_cat = make_cat_h_rules(model_dir_all).T
    task_list = ['fdgo',]

    for ri in range(1):
        rule = task_list[ri]
        epoch = 'go1'

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

        trial = gen_trials_from_model_dir(model_dir_all,rule,mode='test',noise_on = False)

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

            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron) 

            ##################################################################
            # get shapes   
            n_steps, n_trials, n_input_dim = np.shape(trial.x)
            n_rnn = np.shape(h_tf)[2]
            n_output = np.shape(y_hat_tf)[2]

            # Fixed point finder hyperparameters
            # See FixedPointFinder.py for detailed descriptions of available
            # hyperparameters.
            fpf_hps = {'tol_q': 1e-9}
            alr_dict = ({'decrease_factor' : .95, 'initial_rate' : 1})

            
            e_start = max([0, trial.epochs[epoch][0]])
            end_set = [n_steps, trial.epochs[epoch][1]]
            e_end = min(x for x in end_set if x is not None)

            for offset in [180+3*36,]:#range(0,n_trials,int(n_trials/10)):
                
                # t_set = [int(offset)%n_trials,int(offset+n_trials/2)%n_trials]
                # t_set = [20,28]
                # t_set = [(180+offset)%360, (180+offset+36)%360]
                t_set = [int(offset/4.5),int(8+offset/4.5)]

                n_inputs = 0
                input_set = {str(n_inputs) : np.zeros((1,n_input_dim))}

                example_predictions = {'state': h_cat[np.newaxis,:,:]}

                fp_predictions = []

                inputs_1 = trial.x[int(e_start),t_set[0],:]
                inputs_2 = trial.x[int(e_start),t_set[1],:]
                del_inputs = inputs_2 - inputs_1

                for step_i in range(n_interp):

                    step_inputs = inputs_1[np.newaxis,:]+del_inputs[np.newaxis,:]*(step_i/n_interp)
                    inputs = step_inputs
                    inputs_big = inputs[np.newaxis,:]

                    unique_input, input_set = add_unique_to_inputs_list(input_set, str(n_inputs), inputs)
                    
                    n_inputs+=1
                    input_set[str(n_inputs)] = inputs

                    fpf = []
                    fpf = FixedPointFinder(model.cell, sess, alr_hps=alr_dict, method='joint', verbose = False, **fpf_hps) #do_compute_input_jacobians = True , q_tol = 1e-1, do_q_tol = True
                    
                    if np.shape(fp_predictions)[0]==0:
                        fp_predictions = fpf.sample_states(example_predictions['state'],
                                                    n_inits=N_INITS,
                                                    noise_scale=NOISE_SCALE)

                    unique_fps, all_fps = fpf.find_fixed_points(fp_predictions, inputs)


                    if unique_fps.xstar.shape[0]>0:

                        cat_fp_h = np.concatenate((h_cat[np.newaxis,:,:],unique_fps.xstar[np.newaxis,:,:]),axis=1)
                        fp_predictions = fpf.sample_states(cat_fp_h,n_inits=N_INITS,noise_scale=NOISE_SCALE)

                        script_name = os.path.basename(sys.argv[0])[:-3]
                        save_dir = os.path.join(model_dir_all,script_name,rule)#,'random_trials'
                        filename = get_filename(trial,epoch,t_set)

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