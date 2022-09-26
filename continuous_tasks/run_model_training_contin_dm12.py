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

for x in range(1,5):
	folder = str(x)
	filedir = os.path.join('data','crystals','softplus','int20_mod12diff',folder)
	train.train(filedir, seed=x,  max_steps=6e7, ruleset = 'all', rule_trains = 
		['contin_delaydm12',],
		hp = { 'activation' : 'softplus',
				'l1_h': 0,
				'l2_h': 0,
	            'l1_weight': 0,
	            'l2_weight': 0.0001,
	            'l2_weight_init': 0,
	            'n_eachring' : 2,
	            'n_output' : 1+2,
	            'n_input' : 1+2*2+20,
			    'delay_fac' : 1,
            	'sigma_rec': 0.05,
	            'sigma_x': 0.1,
			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
		display_step=1000, rich_output=False)

