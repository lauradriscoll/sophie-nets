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


##dec 17,2008 with continuous delaygo
# for x in range(20):
# x = 3
# folder = str(x)
# filedir = os.path.join('data','n8_uniform_stim','all',folder)
# train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all', #rule_trains = [rule_trains],
# 	hp = { 'activation' : 'relu',
# 			'l1_h': 0,
# 			'l2_h': 0,
#             'l1_weight': 0,
#             'l2_weight': 0,
#             'l2_weight_init': 0,
#             'n_eachring' : 8,
# 		    'delay_fac' : .25},#hp=hp, rule_trains=['fdgo']
# 	display_step=1000, rich_output=False)


# ##dec 17,2008 with continuous delaygo
# for x in range(20):
# 	folder = str(x)
# 	for task in range(20):
# 		rule_trains = all_rules[task]
# 		filedir = os.path.join('data','n8',rule_trains,folder)
# 		train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = [rule_trains],
# 			hp = {'l1_h': 0,
# 					'l2_h': 0,
# 		            'l1_weight': 0,
# 		            'l2_weight': 0,
# 		            'l2_weight_init': 0,
# 	            	'n_eachring' : 8},#hp=hp, rule_trains=['fdgo']
# 			display_step=500, rich_output=False)


# for x in range(1,4):
# 	folder = str(0)
# 	l1reg = 10**-x
# 	# for task in range(19,20):
# 	task = 2
# 	rule_trains = all_rules[task]
# 	filedir = os.path.join('data','sub_training_regularize_l1'+str(x),rule_trains,folder)
# 	train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = [rule_trains],
# 		hp = {'l1_h': l1reg,
# 				'l2_h': 0,
#                 'l1_weight': 0,
#                 'l2_weight': 0,
#                 'l2_weight_init': 0},#hp=hp, rule_trains=['fdgo']
# 		display_step=500, rich_output=False)

# for model_n in range(0,20):
# 	delay_fac = 10
# 	folder = str(model_n)
# 	l2reg = 0

# 	filedir = os.path.join('data','sub_training_regularize_delay_fac_'+str(delay_fac),'all',folder)
# 	train.train(filedir, seed=model_n,  max_steps=1e6, ruleset = 'all', #rule_trains = all_rules,
# 		hp = {'l1_h': 0,
# 				'l2_h': l2reg,
# 	            'l1_weight': 0,
# 	            'l2_weight': 0,
# 	            'l2_weight_init': 0},#hp=hp, rule_trains=['fdgo']
# 		display_step=500, rich_output=False)

# 	for task in [0,2,6]:
# 		rule_trains = all_rules[task]
# 		filedir = os.path.join('data','sub_training_regularize_delay_fac_'+str(delay_fac),rule_trains,folder)
# 		train.train(filedir, seed=model_n,  max_steps=1e6, ruleset = 'all', rule_trains = [rule_trains],
# 			hp = {'l1_h': 0,
# 					'l2_h': l2reg,
# 		            'l1_weight': 0,
# 		            'l2_weight': 0,
# 		            'l2_weight_init': 0},#hp=hp, rule_trains=['fdgo']
# 			display_step=500, rich_output=False)

	# filedir = os.path.join('data','sub_training','contextdm1_contextdm2',folder)
	# train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = [ 'contextdm1','contextdm2'], #hp=hp, rule_trains=['fdgo']
	# 	display_step=500, rich_output=False)

	# for task in range(3,6):
	# 		rule_trains = all_rules[task]
	# 		filedir = os.path.join('data','sub_training',rule_trains,folder)
	# 		train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = [rule_trains], #hp=hp, rule_trains=['fdgo']
	# 			display_step=500, rich_output=False)

	# filedir = os.path.join('data','pro',folder)
	# train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = ['fdgo','reactgo', 'delaygo'], #hp=hp, rule_trains=['fdgo']
	# 	display_step=500, rich_output=False)

	# filedir = os.path.join('data', 'anti',folder)
	# train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = ['fdanti','reactanti', 'delayanti'], #hp=hp, rule_trains=['fdgo']
	# display_step=500, rich_output=False)

	# filedir = os.path.join('data', '1_6',folder)
	# train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = ['fdgo','reactgo', 'delaygo','fdanti','reactanti', 'delayanti'], #hp=hp, rule_trains=['fdgo']
	# 	display_step=500, rich_output=False)

	# filedir = os.path.join('data', 'delaygo_delayanti',folder)
	# train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = [ 'delaygo','delayanti'], #hp=hp, rule_trains=['fdgo']
	# 	display_step=500, rich_output=False)

	# filedir = os.path.join('data', 'fdgo_fdanti',folder)
	# train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = [ 'fdgo','fdanti'], #hp=hp, rule_trains=['fdgo']
	# 	display_step=500, rich_output=False)

	# filedir = os.path.join('data', 'reactgo_reactanti',folder)
	# train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = [ 'reactgo','reactanti'], #hp=hp, rule_trains=['fdgo']
	# 	display_step=500, rich_output=False)

	# filedir = os.path.join('data', 'delaygo_delayanti_reactgo_reactanti',folder)
	# train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = [ 'reactgo','reactanti','delaygo','delayanti'], #hp=hp, rule_trains=['fdgo']
	# 	display_step=500, rich_output=False)


	# filedir = os.path.join('data','sub_training','dmcgo',folder)
	# train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = [ 'dmcgo'], #hp=hp, rule_trains=['fdgo']
	# 	display_step=500, rich_output=False)

	# filedir = os.path.join('data','sub_training','dmsnogo',folder)
	# train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = [ 'dmsnogo'], #hp=hp, rule_trains=['fdgo']
	# 	display_step=500, rich_output=False)

# filedir = os.path.join('data','sub_training','dmsgo',folder)
# train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = [ 'dmsnogo'], #hp=hp, rule_trains=['fdgo']
# 	display_step=500, rich_output=False)

##dec 20,2008
# delay_tasks = np.array([2,5,11,12,13,14,15,16,17,18,19])
# for d in range(4,len(delay_tasks)):
# 	rule_trains = [all_rules[i] for i in delay_tasks[0:d+1]]
# 	for x in range(3):
# 		folder = str(x)
# 		filedir = os.path.join('data','n8_ntasks',str(d),folder)
# 		train.train(filedir, seed=x,  max_steps=1e6, ruleset = 'all', rule_trains = rule_trains, # reduced max steps 1e7 to 1e6 jan 3 2019
# 			hp = { 'activation' : 'relu',
# 					'l1_h': 0,
# 					'l2_h': 0,
# 		            'l1_weight': 0,
# 		            'l2_weight': 0,
# 		            'l2_weight_init': 0,
# 		            'n_eachring' : 8,
# 		            'delay_fac' : 1},#hp=hp, rule_trains=['fdgo']
# 			display_step=1000, rich_output=False)


# # commented out january 8th 2019
# for delay_fac in [1.5, 2]: #.25, .5, .75, 
# 	for x in range(5):
# 		folder = str(x)
# 		filedir = os.path.join('data','n8_delay_fac','delay_fac_'+str(100*delay_fac),folder)
# 		train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all',
# 			hp = { 'activation' : 'relu',
# 					'l1_h': 0,
# 					'l2_h': 0,
# 		            'l1_weight': 0,
# 		            'l2_weight': 0,
# 		            'l2_weight_init': 0,
# 		            'n_eachring' : 8,
# 		            'delay_fac' : delay_fac},#hp=hp, rule_trains=['fdgo']
# 			display_step=1000, rich_output=False)

# for x in range(5):
# 	folder = str(x)
# 	filedir = os.path.join('data','n8_uniform_stim','all',folder)
# 	train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all', #rule_trains = [rule_trains],
# 		hp = { 'activation' : 'relu',
# 				'l1_h': 0,
# 				'l2_h': 0,
# 	            'l1_weight': 0,
# 	            'l2_weight': 0,
# 	            'l2_weight_init': 0,
# 	            'n_eachring' : 8,
# 			    'delay_fac' : 1},#hp=hp, rule_trains=['fdgo']
# 		display_step=1000, rich_output=False)

# for x in range(5):
# 	for task in range(20):
# 		rule_trains = all_rules[task]
# 		filedir = os.path.join('data','n8_uniform_stim',rule_trains,folder)
# 		train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all', rule_trains = [rule_trains],
# 			hp = { 'activation' : 'relu',
# 					'l1_h': 0,
# 					'l2_h': 0,
# 		            'l1_weight': 0,
# 		            'l2_weight': 0,
# 		            'l2_weight_init': 0,
# 		            'n_eachring' : 8,
# 				    'delay_fac' : 1},#hp=hp, rule_trains=['fdgo']
# 			display_step=1000, rich_output=False)

