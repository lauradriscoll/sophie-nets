from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import pdb
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import parallel_for as pfor

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from brewer2mpl import qualitative

from task import generate_trials, rule_name, Trial
from network import Model
import tools

PATH_TO_dPCA = '/home/laura/code/analysis/randy data/'
sys.path.insert(0, PATH_TO_dPCA)
import analysis tools

from numpy.random import rand, randn, randint
PATH_TO_dPCA = '/home/laura/code/dPCA'
sys.path.insert(0, PATH_TO_dPCA)
from dPCA import dPCA

class Model(object):
    """The model."""

    def __init__(self,
                 model_dir,
                 cmap='jet',
                 rule = 'delaygo',
                 n_trials = 100,
                 plotting = False):

	trial = gen_trials_from_model_dir(model_dir,rule)
	model = Model(model_dir)
	with tf.Session() as sess:
		model.restore()
		model._sigma=0
		# get all connection weights and biases as tensorflow variables
		var_list = model.var_list
		# evaluate the parameters after training
		hparams = model.hp
		feed_dict = tools.gen_feed_dict(model, trial, hparams)
		# run model
		h_tf, y_hat_tf = sess.run([model.h, model.y_hat], feed_dict=feed_dict) #(n_time, n_condition, n_neuron)
		self.x = np.transpose(h_tf,(2,1,0))

	# number of neurons, time-points and stimuli
	N,S,T = np.shape(x)

	# build two latent factors
	zt = (np.arange(T)/float(T))
	zs = trial.stim_locs/np.max(trial.stim_locs)

	# build trial-by trial data
	trialR = np.zeros((2,N,int(S/2),T))
	trialR[0,:,:,:] = x[:,range(0,400,2),:]
	trialR[1,:,:,:] = x[:,range(1,400,2),:]

	# trial-average data
	self.R = np.mean(trialR,0)

	# center data
	self.R -= np.mean(R.reshape((N,-1)),1)[:,None,None]

	self.dpca = dPCA.dPCA(labels='st',regularizer='auto')
	self.dpca.protect = ['t']

	self.Z = self.dpca.fit_transform(R,trialR)

	if plotting:
		plt.figure(figsize=(16,24))

		for s in range(S):
			c = cmap(s/S)
			plt.subplot(331)
			plt.plot(time,Z['t'][0,s],c = c)
			plt.title('1st time component')

			plt.subplot(334)
			plt.plot(time,Z['t'][1,s],c = c)
			plt.title('2nd time component')

			plt.subplot(337)
			plt.plot(time,Z['t'][2,s],c = c)
			plt.title('3rd time component')

			plt.subplot(332)
			plt.plot(time,Z['s'][0,s],c = c)
			plt.title('1st stimulus component')

			plt.subplot(335)
			plt.plot(time,Z['s'][1,s],c = c)
			plt.title('2nd stimulus component')

			plt.subplot(338)
			plt.plot(time,Z['s'][2,s],c = c)
			plt.title('3rd stimulus component')

			plt.subplot(333)
			plt.plot(time,Z['st'][0,s],c = c)
			plt.title('1st mixing component')

			plt.subplot(336)
			plt.plot(time,Z['st'][1,s],c = c)
			plt.title('2nd mixing component')

			plt.subplot(339)
			plt.plot(time,Z['st'][2,s],c = c)
			plt.title('3rd mixing component')
		plt.show()