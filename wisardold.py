# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:17:58 2015

@author: canalli
"""

import numpy as np

class Wisard:
	
	def __init__(self):
		self.X_train = None
		self.y_train = None
		self.n_samples = None
		self.n_features = None
		self.n_classes = None
		self.discriminators = []
		self.bleaching_threshold = None

	def fit(self, X, y, address_size=5, bleaching_threshold=20):
		self.X_train = X.astype(int)
		self.n_samples = self.X_train.shape[0]
		self.n_features = self.X_train.shape[1]

		self.bleaching_threshold = bleaching_threshold

		self.y_train = y.astype(int)
		self.n_classes = np.unique(self.y_train).shape[0]

		for i in range(self.n_classes):
			ram = []
			for j in range(self.n_features):
				ram.append(dict())
			self.discriminators.append(ram)


		for i in range(self.n_samples):
			sample = self.X_train[i]
			sample_class = self.y_train[i]

			for j in range(self.n_features):
				address = sample[j]
				d = self.discriminators
				if address in self.discriminators[sample_class][j]:
					self.discriminators[sample_class][j][address] += 1
				else:
					self.discriminators[sample_class][j][address] = 1

	def predict(self, X):
		activations = np.zeros( (X.shape[0], self.n_classes) )
		i = 0
		for ram in self.discriminators:
			j = 0
			for sample in X:
				k = 0
				for address in sample:
					if (address in ram[k]) and (ram[k][address] > self.bleaching_threshold):
						activations[j,i] += 1
					k += 1
				j += 1
			i += 1

		y = np.argsort(activations)
		return y[:,self.n_classes - 1]
