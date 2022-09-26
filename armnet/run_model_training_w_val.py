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

for x in range(5,20):
	folder = str(x)
	filedir = os.path.join('data','lowDin','grad_norm_both','most',folder)
	train.train(filedir, seed=x,  max_steps=3e7, ruleset = 'all',rule_trains = 
		['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
		'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
		'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
		hp = { 'activation' : 'relu',
				'l1_h': 0,
				'l2_h': 0.00000001,
	            'l1_weight': 0,
	            'l2_weight': 0.00001,
	            'l2_weight_init': 0,
	            'n_eachring' : 2,
	            'n_output' : 1+2,
	            'n_input' : 1+2*2+20,
			    'delay_fac' : 1,
			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
		display_step=1000, rich_output=False)

# for x in range(20):
# 	folder = str(x)
# 	for task in range(20):
# 		rule_trains = all_rules[task]
# 		filedir = os.path.join('data','lowDin','grad_norm_l2h000001',rule_trains,folder)
# 		train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all',rule_trains = [rule_trains],
# 			hp = { 'activation' : 'relu',
# 					'l1_h': 0,
# 					'l2_h': 0.000001,
# 		            'l1_weight': 0,
# 		            'l2_weight': 0,
# 		            'l2_weight_init': 0,
# 		            'n_eachring' : 2,
# 		            'n_output' : 1+2,
# 		            'n_input' : 1+2*2+20,
# 				    'delay_fac' : 1,
# 				    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# 			display_step=1000, rich_output=False)

# for x in range(20):
# 	folder = str(x)
# 	for task in range(20):
# 		rule_trains = all_rules[task]
# 		filedir = os.path.join('data','lowDin','grad_norm_l2001',rule_trains,folder)
# 		train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all',rule_trains = [rule_trains],
# 			hp = { 'activation' : 'relu',
# 					'l1_h': 0,
# 					'l2_h': 0,
# 		            'l1_weight': 0,
# 		            'l2_weight': 0.001,
# 		            'l2_weight_init': 0,
# 		            'n_eachring' : 2,
# 		            'n_output' : 1+2,
# 		            'n_input' : 1+2*2+20,
# 				    'delay_fac' : 1,
# 				    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# 			display_step=1000, rich_output=False)

# for x in range(20):
# 	folder = str(x)
# 	filedir = os.path.join('data','lowDin','grad_norm_none','most',folder)
# 	train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all',rule_trains = 
# 		['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
# 		'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
# 		'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
# 		hp = { 'activation' : 'relu',
# 				'l1_h': 0,
# 				'l2_h': 0,
# 	            'l1_weight': 0,
# 	            'l2_weight': 0,
# 	            'l2_weight_init': 0,
# 	            'n_eachring' : 2,
# 	            'n_output' : 1+2,
# 	            'n_input' : 1+2*2+20,
# 			    'delay_fac' : 1,
# 			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# 		display_step=1000, rich_output=False)

# for x in range(20):
# 	folder = str(x)
# 	filedir = os.path.join('data','lowDin','grad_norm_l2h000001','most',folder)
# 	train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all',rule_trains = 
# 		['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
# 		'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
# 		'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
# 		hp = { 'activation' : 'relu',
# 				'l1_h': 0,
# 				'l2_h': 0.000001,
# 	            'l1_weight': 0,
# 	            'l2_weight': 0,
# 	            'l2_weight_init': 0,
# 	            'n_eachring' : 2,
# 	            'n_output' : 1+2,
# 	            'n_input' : 1+2*2+20,
# 			    'delay_fac' : 1,
# 			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# 		display_step=1000, rich_output=False)

# for x in range(20):
# 	folder = str(x)
# 	filedir = os.path.join('data','lowDin','grad_norm_l2001','most',folder)
# 	train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all',rule_trains = 
# 		['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
# 		'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
# 		'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
# 		hp = { 'activation' : 'relu',
# 				'l1_h': 0,
# 				'l2_h': 0,
# 	            'l1_weight': 0,
# 	            'l2_weight': 0.001,
# 	            'l2_weight_init': 0,
# 	            'n_eachring' : 2,
# 	            'n_output' : 1+2,
# 	            'n_input' : 1+2*2+20,
# 			    'delay_fac' : 1,
# 			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# 		display_step=1000, rich_output=False)

# for x in range(20):
# 	folder = str(x)
# 	for task in range(20):
# 		rule_trains = all_rules[task]
# 		filedir = os.path.join('data','lowDin','grad_norm_l1h000001',rule_trains,folder)
# 		train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all', rule_trains = [rule_trains],
# 			hp = { 'activation' : 'relu',
# 					'l1_h': 0.000001,
# 					'l2_h': 0,
# 		            'l1_weight': 0,
# 		            'l2_weight': 0,
# 		            'l2_weight_init': 0,
# 		            'n_eachring' : 2,
# 		            'n_output' : 1+2,
# 		            'n_input' : 1+2*2+20,
# 				    'delay_fac' : 1,
# 				    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# 			display_step=1000, rich_output=False)

# for x in range(20):
# 	folder = str(x)
# 	for task in range(20):
# 		rule_trains = all_rules[task]
# 		filedir = os.path.join('data','lowDin','grad_norm_l1001',rule_trains,folder)
# 		train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all', rule_trains = [rule_trains],
# 			hp = { 'activation' : 'relu',
# 					'l1_h': 0,
# 					'l2_h': 0,
# 		            'l1_weight': 0.001,
# 		            'l2_weight': 0,
# 		            'l2_weight_init': 0,
# 		            'n_eachring' : 2,
# 		            'n_output' : 1+2,
# 		            'n_input' : 1+2*2+20,
# 				    'delay_fac' : 1,
# 				    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# 			display_step=1000, rich_output=False)


# # for x in range(1,2):
# # 	folder = str(x)
# # 	for task in range(14,15):
# # 		rule_trains = all_rules[task]
# # 		filedir = os.path.join('data','lowDin','grad_norm_l2h000001',rule_trains,folder)
# # 		train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all', rule_trains = [rule_trains],
# # 			hp = { 'activation' : 'relu',
# # 					'l1_h': 0,
# # 					'l2_h': 0.000001,
# # 		            'l1_weight': 0,
# # 		            'l2_weight': 0,
# # 		            'l2_weight_init': 0,
# # 		            'n_eachring' : 2,
# # 		            'n_output' : 1+2,
# # 		            'n_input' : 1+2*2+20,
# # 				    'delay_fac' : 1,
# # 				    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# # 			display_step=1000, rich_output=False)

# # for x in range(1):
# # 	folder = str(x)
# # 	for task in range(11,20):
# # 		rule_trains = all_rules[task]
# # 		filedir = os.path.join('data','lowDin','grad_norm_l2001',rule_trains,folder)
# # 		train.train(filedir, seed=x,  max_steps=1e7, ruleset = 'all', rule_trains = [rule_trains],
# # 			hp = { 'activation' : 'relu',
# # 					'l1_h': 0,
# # 					'l2_h': 0,
# # 		            'l1_weight': 0,
# # 		            'l2_weight': 0.001,
# # 		            'l2_weight_init': 0,
# # 		            'n_eachring' : 2,
# # 		            'n_output' : 1+2,
# # 		            'n_input' : 1+2*2+20,
# # 				    'delay_fac' : 1,
# # 				    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# # 			display_step=1000, rich_output=False)



# # # january 8th 2019
# # for x in range(1,5):
# # 	folder = str(x)
# # 	filedir = os.path.join('data','lowDin','most',folder)
# # 	train.train(filedir, seed=x,  max_steps=1e8, ruleset = 'all', 
# # 		rule_trains = ['fdgo','reactgo', 'delaygo','fdanti','reactanti', 'delayanti',
# # 		'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
# # 		'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],
# # 		hp = { 'activation' : 'relu',
# # 				'l1_h': 0,
# # 				'l2_h': 0,
# # 	            'l1_weight': 0,
# # 	            'l2_weight': 0,
# # 	            'l2_weight_init': 0,
# # 	            'n_eachring' : 2,
# # 	            'n_output' : 1+2,
# # 	            'n_input' : 1+2*2+20,
# # 			    'delay_fac' : 1,
# # 			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# # 		display_step=1000, rich_output=False)


# # for x in range(1):
# # 	folder = str(x)
# # 	filedir = os.path.join('data','lowDin','delaygo_contextdelaydm1',folder)
# # 	train.train(filedir, seed=x,  max_steps=1e8, ruleset = 'all', 
# # 		rule_trains = ['delaygo','contextdelaydm1'],
# # 		hp = { 'activation' : 'relu',
# # 				'l1_h': 0,
# # 				'l2_h': 0,
# # 	            'l1_weight': 0,
# # 	            'l2_weight': 0,
# # 	            'l2_weight_init': 0,
# # 	            'n_eachring' : 2,
# # 	            'n_output' : 1+2,
# # 	            'n_input' : 1+2*2+20,
# # 			    'delay_fac' : 1,
# # 			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# # 		display_step=1000, rich_output=False)


# # # feb 7th 2019
# # for x in range(1):
# # 	folder = str(x)
# # 	filedir = os.path.join('data','lowDin','delay_4tasks',folder)
# # 	train.train(filedir, seed=x,  max_steps=1e8, ruleset = 'all', 
# # 		rule_trains = ['delaygo','delaydm1','contextdelaydm1','dmsgo'],
# # 		hp = { 'activation' : 'relu',
# # 				'l1_h': 0,
# # 				'l2_h': 0,
# # 	            'l1_weight': 0,
# # 	            'l2_weight': 0,
# # 	            'l2_weight_init': 0,
# # 	            'n_eachring' : 2,
# # 	            'n_output' : 1+2,
# # 	            'n_input' : 1+2*2+20,
# # 			    'delay_fac' : 1,
# # 			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# # 		display_step=1000, rich_output=False)




# # # january 8th 2019
# # for x in range(1):
# # 	filedir = os.path.join('data','lowDin','all',str(x))
# # 	train.train(filedir, seed=x,  max_steps=1e8, ruleset = 'all',
# # 		hp = { 'activation' : 'relu',
# # 				'l1_h': 0,
# # 				'l2_h': 0,
# # 	            'l1_weight': 0,
# # 	            'l2_weight': 0,
# # 	            'l2_weight_init': 0,
# # 	            'n_eachring' : 2,
# # 	            'n_output' : 1+2,
# # 	            'n_input' : 1+2*2+20,
# # 			    'delay_fac' : 1,
# # 			    'use_separate_input': False},#hp=hp, rule_trains=['fdgo']
# # 		display_step=1000, rich_output=False)


