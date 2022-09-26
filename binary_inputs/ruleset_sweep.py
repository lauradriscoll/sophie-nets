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

rules_dict = \
    {'basic' : ['fdgo','delaygo','fdanti','delayanti'],

    'mante' : ['contextdm1', 'contextdm2'],

    'oicdmc' : ['oic', 'dmc']}

# parse input arguments as:
rnn_type = str(sys.argv[1])
activation = str(sys.argv[2])
init = str(sys.argv[3])
n_rnn = int(sys.argv[4])
l2w = float(sys.argv[5])
l2h = float(sys.argv[6])
l1w = float(sys.argv[7])
l1h = float(sys.argv[8])
seed = int(sys.argv[9])
lr = float(sys.argv[10])
sigma_rec = float(sys.argv[11])/20
sigma_x = float(sys.argv[12])/20
pop_rule = int(sys.argv[13])
ruleset = str(sys.argv[14])
w_rec_coeff = float(sys.argv[15])/10

rule_trains = rules_dict[ruleset]
rule_trains = [rule_trains[pop_rule],rule_trains[(pop_rule+2)%len(rules_dict[ruleset])]]
s = '_'
rule_trains_str = s.join(rule_trains)
folder = str(seed)
net_name = 'lr'+str(-lr)+'l2_w'+str(-l2w)+'_h'+str(-l2h)+'_sig_rec'+str(sigma_rec)+'_sig_x'+str(sigma_x)+'_w_rec_coeff'+str(w_rec_coeff)+'_'+rule_trains_str
filedir = os.path.join('data',ruleset,rnn_type,activation,init,str(len(rule_trains))+'_tasks',str(n_rnn)+'_n_rnn',net_name,folder)
train.train(filedir, seed=seed,  max_steps=6e6, ruleset = 'basic',rule_trains = rule_trains,
    hp = { 'batch_size_train' : 64,
	       'activation' : activation,
            'w_rec_init' : init,
            'w_rec_coeff' : w_rec_coeff,
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
	        'learning_rate': 10**(lr/2)},
    display_step=1000, rich_output=False)
