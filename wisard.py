# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:17:58 2015

@author: canalli
"""

import numpy as np

class Wisard:
	
	def __init__(self, disable_randomness=False):
		self.X_train = None
		self.y_train = None
		self.n_samples = None
		self.n_features = None
		self.n_classes = None
		self.n_rams = None
		self.address_size = None
		self.discriminators = []
		self.permutation_array = None
		self.disable_randomness = disable_randomness

		
	def fit(self, X, y, address_size=4):

		if (self.disable_randomness):
			self.X_train = X.astype(int)
		else:
			self.permutation_array = np.random.permutation(X.shape[1])
			self.X_train = X[:,self.permutation_array].astype(int)

		self.n_samples = self.X_train.shape[0]
		self.n_features = self.X_train.shape[1]

		if (self.X_train.shape[1] % address_size) == 0:
			self.address_size = address_size
		else:
			self.address_size = address_size - (self.X_train.shape[1] % address_size)

		self.n_rams = self.n_features // self.address_size

		self.y_train = y.astype(int)
		self.n_classes = np.max(self.y_train) + 1

		for i in range(self.n_classes):
			ram = []
			for j in range(self.n_rams):
				ram.append(dict())
			self.discriminators.append(ram)

		for i in range(self.n_samples):
			sample = self.X_train[i]
			sample_class = self.y_train[i]

			addresses = [''.join(x) for x in sample.astype(str).reshape(self.n_rams,self.address_size)]

			for j in range(self.n_rams):
				address = addresses[j]
				if address in self.discriminators[sample_class][j]:
					self.discriminators[sample_class][j][address] += 1
				else:
					self.discriminators[sample_class][j][address] = 1
				
	def predict(self, X):

		if (self.disable_randomness):
			X_test = X.astype(int)
		else:
			X_test = X[:,self.permutation_array].astype(int)

		activations = np.zeros( (X_test.shape[0], self.n_classes) )
		
		bleaching_threashold = 0
		max_bleaching = 20
		while bleaching_threashold < max_bleaching and self._needs_bleaching(activations):
			
			i = 0
			for ram in self.discriminators:
				j = 0
				for sample in X_test:
					addresses = [''.join(x) for x in sample.astype(str).reshape(self.n_rams,self.address_size)]
					k = 0
					for address in addresses:
						if (address in ram[k]) and (ram[k][address] > bleaching_threashold):
							activations[j,i] += 1
						k += 1
						
					j += 1
				i += 1
			bleaching_threashold += 1

		y = np.argsort(activations)
		return y[:,self.n_classes - 1]
		
	def _needs_bleaching(self, activations):
		sorted_activations = np.sort(activations)
		diffs_from_greaters = sorted_activations[:,-1] - sorted_activations[:,-2]
		
		#zeros means draw
		if (np.count_nonzero(diffs_from_greaters) != diffs_from_greaters.shape[0]):
			return True
			
		return False