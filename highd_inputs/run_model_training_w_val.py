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

all_rules = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']

n_eachring = 32      

for x in range(1):
	folder = str(x)
	filedir = os.path.join('data','crystals','highd_inputs','all_rules'+'_'+str(n_eachring),'softplus','l2h000001_l2w00001',folder)
	train.train(filedir, seed=x,  max_steps=9e6, ruleset = 'all',rule_trains = all_rules,
		hp = { 'activation' : 'softplus',
				'l1_h': 0,
				'l2_h': 0.000001,
	            'l1_weight': 0,
	            'l2_weight': 0.00001,
	            'l2_weight_init': 0,
	            'n_eachring' : n_eachring,
	            'n_output' : 1+n_eachring,
	            'n_input' : 1+n_eachring*2+20,
			    'delay_fac' : 1,
            	'sigma_rec': 0.05,
	            'sigma_x': 0.1,
			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
		display_step=1000, rich_output=False)

for x in range(1):
	folder = str(x)
	filedir = os.path.join('data','crystals','highd_inputs','all_rules'+'_'+str(n_eachring),'softplus','l2h00001_l2w0001',folder)
	train.train(filedir, seed=x,  max_steps=9e6, ruleset = 'all',rule_trains = all_rules,
		hp = { 'activation' : 'softplus',
				'l1_h': 0,
				'l2_h': 0.00001,
	            'l1_weight': 0,
	            'l2_weight': 0.0001,
	            'l2_weight_init': 0,
	            'n_eachring' : n_eachring,
	            'n_output' : 1+n_eachring,
	            'n_input' : 1+n_eachring*2+20,
			    'delay_fac' : 1,
            	'sigma_rec': 0.05,
	            'sigma_x': 0.1,
			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
		display_step=1000, rich_output=False)

# for x in range(1):
# 	folder = str(x)
# 	filedir = os.path.join('data','crystals','highd_inputs','all','softplus','no_reg'+'_'+str(n_eachring),folder)
# 	train.train(filedir, seed=x,  max_steps=3e7, ruleset = 'all',rule_trains = 
# 		['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
#               'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
#               'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
#               'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
# 		hp = { 'activation' : 'softplus',
# 				'l1_h': 0,
# 				'l2_h': 0,
# 	            'l1_weight': 0,
# 	            'l2_weight': 0,
# 	            'l2_weight_init': 0,
# 	            'n_eachring' : n_eachring,
# 	            'n_output' : 1+n_eachring,
# 	            'n_input' : 1+n_eachring*2+20,
# 			    'delay_fac' : 1,
#             	'sigma_rec': 0.05,
# 	            'sigma_x': 0.1,
# 			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# 		display_step=1000, rich_output=False)

# for x in range(1):
# 	folder = str(x)
# 	filedir = os.path.join('data','crystals','highd_inputs','contextdm1_contextdm2_multidm','softplus','no_reg'+'_'+str(n_eachring),folder)
# 	train.train(filedir, seed=x,  max_steps=3e7, ruleset = 'all',rule_trains = 
# 		['contextdm1', 'contextdm2', 'multidm'],
# 		hp = { 'activation' : 'softplus',
# 				'l1_h': 0,
# 				'l2_h': 0,
# 	            'l1_weight': 0,
# 	            'l2_weight': 0,
# 	            'l2_weight_init': 0,
# 	            'n_eachring' : n_eachring,
# 	            'n_output' : 1+n_eachring,
# 	            'n_input' : 1+n_eachring*2+20,
# 			    'delay_fac' : 1,
#             	'sigma_rec': 0.05,
# 	            'sigma_x': 0.1,
# 			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# 		display_step=1000, rich_output=False)
