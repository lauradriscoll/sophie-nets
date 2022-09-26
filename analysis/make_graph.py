from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import numpy as np
import numpy.random as npr
from numpy import linalg as LA
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
from task import generate_trials, rules_dict
from network import FixedPoint_Model
import tools
from tools_lnd import get_T_inds

PATH_TO_RECURRENT_WHISPERER = p+'/code/recurrent-whisperer'#'/home/laura/code/recurrent-whisperer'#
sys.path.insert(0, PATH_TO_RECURRENT_WHISPERER)
from RecurrentWhisperer import RecurrentWhisperer

PATH_TO_FIXED_POINT_FINDER = p+'/code/fixed-point-finder' #'/home/laura/code/fixed-point-finder-experimental'#
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinder import FixedPointFinder


####
model_n = 0
task_list = ['fdgo','fdanti','delaygo','delayanti']
rule = task_list[0]
which_net = 'l2w0001'
fldr = '4_tasks'
supp = []#'long_train'

if fldr == '4_tasks':
	s = '_'
	rule_trains_str = s.join(task_list)
	file_spec = os.path.join(fldr,which_net+'_'+rule_trains_str)
else:
	file_spec = which_net
	
dir_specific_all = os.path.join('crystals','softplus',file_spec)#,supp)
	
m = os.path.join(p,'data/rnn/multitask/',net,dir_specific_all,str(model_n))
####

model = FixedPoint_Model(m)
with tf.Session() as sess:
	model.restore()
	model._sigma=0
	var_list = model.var_list
	params = [sess.run(var) for var in var_list]
	hparams = model.hp
	trial = generate_trials(rule, hparams, mode='random', noise_on=False, batch_size = 128, delay_fac =1)
	feed_dict = tools.gen_feed_dict(model, trial, hparams)
	h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
	
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

def vanilla_run_with_h0(params, x_t, h0, alpha):
	h = h0
	h_t = []
	h_t.append(np.expand_dims(h0,axis=1))
	for x in x_t:
		h = rnn_vanilla(params, np.squeeze(h), np.squeeze(x.T), alpha)
		h_t.append(np.expand_dims(h,axis=1))

	h_t = np.squeeze(np.array(h_t))  
	return h_t

def get_filename(trial,epoch,t):
	ind_stim_loc  = 180*trial.y_loc[-1,t]/np.pi
	filename = epoch+'_'+str(round(ind_stim_loc,2))
	return filename

fp_epoch = 'stim1'
offset = 0 #set stimulus angle
isanti = ['anti' in x for x in task_list] #fp saved by output, this is for aligning stim
fp_output_ang = (180*np.double(isanti)+offset)%360
NUM_TRIALS = 80
t_num = int(NUM_TRIALS*offset/360) #ONLY WORKS FOR tasks w 80 test trials (this includes 1st 6 tasks)
eps_ball_coef = hparams['alpha']/2
n_jit = 2
n_steps = 100
lim = LA.norm(np.ones((1,n_rnn))*eps_ball_coef)

for ri in range(len(task_list)):
	elist = []

	#for each trained rule
	rule = task_list[ri]

	#load rule specific trial
	model = FixedPoint_Model(m)
	with tf.Session() as sess:
		model.restore()
		hparams = model.hp
		model._sigma=0
		trial = generate_trials(rule, hparams, mode='test',noise_on=False, batch_size = 100,delay_fac = 1)

	#load FPs (these were saved according to ouput angle)
	f = os.path.join(m,'tf_fixed_pts_all_init',rule,fp_epoch+'_'+str(fp_output_ang[ri])+'.npz')
	fp_struct = np.load(f)
	sorted_fps = fp_struct['xstar']
	fp_inds = range(len(sorted_fps))
	qvals = fp_struct['qstar']

	for ii in fp_inds:
		
		jit_loc = sorted_fps[ii,:]

		T_inds = get_T_inds(trial, fp_epoch)
		x_t = np.matlib.repmat(trial.x[T_inds[1]-1,offset,:],n_steps,1)
		
		for jit in range(n_jit):
			h0 = jit_loc + eps_ball_coef*npr.randn(n_rnn)
			h_t = vanilla_run_with_h0(params, x_t, h0, alpha)
			
			X = np.squeeze(h_t).astype(np.float64)
			D = np.zeros((n_steps,len(sorted_fps)))
			for step_ii in range(n_steps):
				for fp_ii in fp_inds:
					D[step_ii,fp_ii] = LA.norm(X[step_ii,:]-sorted_fps[fp_ii,:])
					

			if np.min(D[-1,:])< lim:
				elist.append((np.argmin(D[0,:]), np.argmin(D[-1,:])))

	graph = {}
	graph = {'elist':elist,
		'xstar':fp_struct['xstar'], 
		'qstar':fp_struct['qstar'], 
		'input':trial.x[T_inds[1]-1,t_num,:], 
		'eps_ball_coef':eps_ball_coef,
		'n_jit':n_jit,
		'rule':rule,
		'epoch':fp_epoch}

	save_dir = os.path.join(m,'graph',rule,fp_epoch)
	filename = get_filename(trial, fp_epoch, t_num)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	np.savez(os.path.join(save_dir,filename+'.npz'),**graph)
