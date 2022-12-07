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

##################################################################
#Find right model dir
##################################################################
net = 'stepnet'
epoch = 'go1' # epoch = 2 #go1 epoch
model_n = 0
dir_specific_all = 'crystals/softplus/l2w0001' #'varGo/lowD/'#lowD/combos' ##grad_norm_l2h000001' /Documents/data/rnn/multitask/varGo/lowD/most/
model_dir = os.path.join(p,'data/rnn/multitask/',dir_specific_all,'most',str(model_n))
task_list = ['delaygo', 'delayanti'] #['fdgo', 'reactgo', 'fdanti', 'reactanti',  #'delaygo', 'delayanti',
            #   'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
            #   'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
##################################################################

PATH_YANGNET = os.path.join(p,'code/multitask-nets',net)
sys.path.insert(0, PATH_YANGNET)

from task import generate_trials, rule_name, rule_index_map
from network import Model
import tools

PATH_TO_RECURRENT_WHISPERER = os.path.join(p,'code/recurrent-whisperer')#'/home/laura/code/recurrent-whisperer'#
sys.path.insert(0, PATH_TO_RECURRENT_WHISPERER)
from RecurrentWhisperer import RecurrentWhisperer

PATH_TO_FIXED_POINT_FINDER = os.path.join(p,'code/fixed-point-finder') #'/home/laura/code/fixed-point-finder-experimental'#
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

def get_filename(trial, epoch, rule, t):
    if 'context' in rule or 'multi' in rule:
        n_stim_loc, n_stim_mod1_strength, n_stim_mod2_strength = batch_shape = 20, 5, 5
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod1_strength, ind_stim_mod2_strength = np.unravel_index(range(batch_size),batch_shape)

        stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
        stim2_locs = (stim1_locs+np.pi)%(2*np.pi)
        stim1_mod1_strengths = 0.4*ind_stim_mod1_strength/n_stim_mod1_strength+0.8
        stim2_mod1_strengths = 2 - stim1_mod1_strengths
        stim1_mod2_strengths = 0.4*ind_stim_mod2_strength/n_stim_mod2_strength+0.8
        stim2_mod2_strengths = 2 - stim1_mod2_strengths
        filename = trial.epochs.keys()[epoch]+'_'+str(round(stim1_locs[t],2))+'_'+str(round(stim2_locs[t],2))+'_'+str(round(stim1_mod1_strengths[t]-stim2_mod1_strengths[t],2))+'_'+str(round(stim1_mod2_strengths[t]-stim2_mod2_strengths[t],2))
    
    elif 'fd' in rule or 'react' in rule or 'delaygo' in rule or 'delayanti' in rule:
        n_stim_loc, n_stim_mod = batch_shape = 200, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size),batch_shape)
        stim_locs  = 2*np.pi*ind_stim_loc/n_stim_loc
        filename = trial.epochs.keys()[epoch]+'_'+str(round(stim_locs[t],2))
        filename = epoch+'_'+str(round(stim_locs[t],2))

    return filename, ind_stim_loc

def plot_input_dims(fpf_dict, rule, t):

    FIG_WIDTH = 6 # inches
    FIG_HEIGHT = 6 # inches
    FONT_WEIGHT = 'bold'

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT),tight_layout=True)
    ax = fig.add_subplot(121)
    ax.set_xlabel('Input 1', fontweight=FONT_WEIGHT)
    ax.set_ylabel('Input 2', fontweight=FONT_WEIGHT)

    # if 'context' in rule or 'multi' in rule:
    #     n_stim_loc, n_stim_mod1_strength, n_stim_mod2_strength = batch_shape = 20, 5, 5
    #     batch_size = np.prod(batch_shape)
    #     ind_stim_loc, ind_stim_mod1_strength, ind_stim_mod2_strength = np.unravel_index(range(batch_size),batch_shape)

    #     stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
    #     stim2_locs = (stim1_locs+np.pi)%(2*np.pi)
    #     stim1_mod1_strengths = 0.4*ind_stim_mod1_strength/n_stim_mod1_strength+0.8
    #     stim2_mod1_strengths = 2 - stim1_mod1_strengths
    #     stim1_mod2_strengths = 0.4*ind_stim_mod2_strength/n_stim_mod2_strength+0.8
    #     stim2_mod2_strengths = 2 - stim1_mod2_strengths
    #     coh1 = stim1_mod1_strengths-stim2_mod1_strengths
    #     coh2 = stim1_mod2_strengths-stim2_mod2_strengths

    #     colors1 = plt.cm.coolwarm(np.linspace(0,1,len(np.unique(coh1))))
    #     colors2 = plt.cm.PRGn(np.linspace(0,1,len(np.unique(coh2))))
    #     color1 = colors1[coh1[t]==np.unique(coh1),:]
    #     color2 = colors2[coh2[t]==np.unique(coh1),:]
    # elif 'fd' in rule or 'react' in rule or 'delaygo' in rule or 'delayanti' in rule:
    #     n_stim_loc, n_stim_mod = batch_shape = 20, 2
    #     batch_size = np.prod(batch_shape)
    #     ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size),batch_shape)
    #     stim_locs  = 2*np.pi*ind_stim_loc/n_stim_loc
    #     color1 = plt.cm.coolwarm(np.linspace(0,1,len(np.unique(stim_locs))))
    #     color2 = plt.cm.coolwarm(np.linspace(0,1,len(np.unique(stim_locs))))
  
    # n_dfdu_v = np.matrix.transpose(fpf_dict['J_inputs'][0,:,(1+ind_stim_loc[t], 1+(16+ind_stim_loc[t])%32)]) #make these constants variables
    # q = np.linalg.qr(n_dfdu_v)[0]
    # fp_ind = np.argmin(fpf_dict['qstar'])

    for t_ind in range(0,400,40):#15*(t//15),-15*((-t)//15)):
        z_state_traj = project2d(np.squeeze(fpf_dict['state_traj'][t_ind,e_start:e_end,:]).T,q)
        ax.plot(z_state_traj[:,0], z_state_traj[:,1], '-',color = color1[0],linewidth=6) 
        ax.plot(z_state_traj[:,0], z_state_traj[:,1], '--',color = color2[0],linewidth=6) 

        z_xstar = project2d(fpf_dict['xstar'][fp_ind],q)
        ax.plot(z_xstar[:,0], z_xstar[:,1], '.m', markersize=25)
        plt.axis('equal')
        plt.axis('square')
        plt.title(rule)

        ax2 = fig.add_subplot(122)
        ax2.set_xlabel('Real Part', fontweight=FONT_WEIGHT)
        ax2.set_ylabel('Imaginary Part', fontweight=FONT_WEIGHT)
        e_vals, e_vecs = np.linalg.eig(fpf_dict['J_xstar'])
        if 'context' in rule or 'multi' in rule:
            plt.text(.1,.3,'coh1 : ' + str(round(coh1[t],2)) + ' coh2 : ' + str(round(coh2[t],2)))
            plt.text(.1,.4,'stim1 : ' + str(round(stim1_locs[t],2)) + ' stim2 : ' + str(round(stim2_locs[t],2)))
        elif 'fd' in rule or 'react' in rule or 'delaygo' in rule or 'delayanti' in rule:
            plt.text(.1,.3,'stim : ' + str(round(stim_locs[t],2))) 
        ax2.scatter(e_vals.real, e_vals.imag)
        ax2.plot([1,1],[-.5,.5],'-k')
        ax2.set_xticks(range(-4,4))
        ax2.set_yticks(range(-4,4))
        plt.axis('equal')
        plt.axis('square')
        
        plt.ion()
        plt.show()

    return plt


##################################################################
#Run fixed pt finder
##################################################################

'''Initial states are sampled from states observed during realistic behavior
of the network. Because a well-trained network transitions instantaneously
from one stable state to another, observed networks states spend little if any
time near the unstable fixed points. In order to identify ALL fixed points,
noise must be added to the initial states before handing them to the fixed
point finder.'''
NOISE_SCALE = 0 # 0.01 #0.5 # Standard deviation of noise added to initial states
N_INITS = 500 # The number of initial states to provide

for rule in task_list:
    model = Model(model_dir)
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
        trial = generate_trials(rule, hparams, mode='test', noise_on=False)# get feed_dict
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
        fpf_dict = {}
        alr_dict = ({'decrease_factor' : .95, 'initial_rate' : 1})

        n_epochs = len(trial.epochs)
        # print(trial.epochs.keys()[epoch])
        # pdb.set_trace()
        # for epoch in range (0,n_epochs):
        # e_start = max([0, trial.epochs.values()[epoch][0]])
        # end_set = [n_steps, trial.epochs.values()[epoch][1]]
        e_start = max([0, trial.epochs[epoch][0]])
        end_set = [n_steps, trial.epochs[epoch][1]]
        e_end = min(x for x in end_set if x is not None)

        n_inputs = 0
        input_set = {str(n_inputs) : np.zeros((1,n_input_dim))}

        for t in range(1):#range(1,n_trials): np.arange(0, n_trials, int(n_trials/10)): #only 1 input for go period

            inputs = np.squeeze(trial.x[e_end-1,t,:])
            inputs = inputs[np.newaxis,:]

            unique_input, input_set = add_unique_to_inputs_list(input_set, str(n_inputs), inputs)
            
            if unique_input:
                n_inputs+=1
                input_set[str(n_inputs)] = inputs

                fpf = []
                fpf = FixedPointFinder(model.cell,sess, tol=1e-25, max_iters=1e5, alr_hps=alr_dict, method='joint', verbose = False, **fpf_hps) #do_compute_input_jacobians = True , do_q_tol = True

                example_predictions = {'state': np.transpose(h_tf,(1,0,2)), #[0:90,0:1,:]
                                        'output': np.transpose(y_hat_tf,(1,0,2))}
                
                initial_states = fpf.sample_states(example_predictions['state'][:,e_end-1:e_end,:],#[:,e_start:e_end,:],
                                                n_inits=N_INITS,
                                                noise_scale=NOISE_SCALE)

                # pdb.set_trace()
                # Run the fixed point finder
                fpf.find_fixed_points(initial_states, inputs)
                # Run the fixed point finder
                unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

                # Visualize identified fixed points with overlaid RNN state trajectories
                # All visualized in the 3D PCA space fit the the example RNN states.
                # unique_fps.plot(example_predictions['state'],
                #     plot_batch_idx=range(0, n_trials, int(n_trials/40)),
                #     plot_start_time=e_start)
                # pdb.set_trace()

                # print('Entering debug mode to allow interaction with objects and figures.')
                # pdb.set_trace()

                if unique_fps.xstar.shape[0]>0:

                    fpf_dict.update({'xstar':unique_fps.xstar,
                        # 'J_inputs':unique_fps.J_inputs, 
                        'J_xstar':unique_fps.J_xstar, 
                        'qstar':unique_fps.qstar, 
                        'inputs':unique_fps.inputs, 
                        'epoch_inds':range(e_start,e_end),
                        'state_traj':example_predictions['state']})

                    save_dir = os.path.join(model_dir,'fixed_pts',rule)
                    # filename, ind_stim_loc = get_filename(trial, epoch, rule, t)
                    filename = rule+'_'+epoch

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    np.save(os.path.join(save_dir,filename),fpf_dict)

                    # plot_input_dims(fpf_dict, rule, t)
                    # plt.title(trial.epochs.keys()[epoch])
                    # plt.savefig(os.path.join(save_dir,filename + trial.epochs.keys()[epoch] + '.png'))


                    # Visualize identified fixed points with overlaid RNN state trajectories
                    # All visualized in the 3D PCA space fit the the example RNN states.
                    # t_set = range(5*(t//5),-5*(-(t+1)//5)) #get multiples of 5 trials
                    #t_set = range(0,40)

                    #fpf.plot_summary(example_predictions['state'][t_set,e_start:e_end,:])
                    #plt.savefig(os.path.join(save_dir,filename +'3D.png'))
                    # pdb.set_trace()

