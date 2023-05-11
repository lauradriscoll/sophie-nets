from __future__ import division

import os
import sys
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import task
from task import generate_trials
from network import Model, get_perf
import tools
import train

# delay match to category = dmcgo
# delay match to category anti = dmcnogo

all_rules = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']



###### EDIT THESE PARAMETERS #ca#######
rule_trains = ['delaygo']  # pick what rules to train

# select hyperparameters (leave fixed)
l2w = -7
l2h = -7
l1w = 0
l1h = 0
activation = 'softplus'
rnn_type = 'LeakyRNN'
init = 'diag'
n_rnn = 256
lr = -7
sigma_rec = 1/20
sigma_x = 2/20
pop_rule = 5
ruleset = 'all'
w_rec_coeff  = 9

#############

s = '_'
rule_trains_str = s.join(rule_trains)

for x in range(1):
	folder = str(x)
	
	net_name = 'lr'+str(-lr)+'l2_w'+str(-l2w)+'_h'+str(-l2h)+'_sig_rec'+str(sigma_rec)+'_sig_x'+str(sigma_x)+'_w_rec_coeff'+str(w_rec_coeff)+'_'+rule_trains_str
	filedir = os.path.join('data',ruleset,rnn_type,activation,init,str(len(rule_trains))+'_tasks',str(n_rnn)+'_n_rnn',net_name,folder)

	train.train(filedir, seed=x,  max_steps=6e6, ruleset = 'basic',rule_trains = rule_trains,
		hp = { 'activation' : activation,
				'w_rec_init': init,
				'n_rnn': n_rnn,
				'l1_h': np.min([-l1h, 10**l1h]),
				'l2_h': np.min([-l2h, 10**l2h]),
	            'l1_weight': np.min([-l1w, 10**l1w]),
	            'l2_weight': np.min([-l2w, 10**l2w]),
	            'l2_weight_init': 0,
            	'sigma_rec': sigma_rec,
	            'sigma_x': sigma_x,
	            'rnn_type': rnn_type,
			    'use_separate_input': False,
			    'w_rec_coeff': w_rec_coeff/10,
			    'learning_rate': 10**(lr/2)},
		display_step=1000, rich_output=False)
