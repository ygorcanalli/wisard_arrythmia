# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:15:41 2015

@author: canalli
"""

import arrhythmia_reader as reader
import numpy as np
from sklearn.cross_validation import KFold
from wisard import Wisard

output_path = 'results.csv'

X = reader.read_numerical_features()
classes, y = reader.read_simplified_classifications()

def evaluate(address_size, bleaching_threshold, disable_ransomness):
	
	n_folds = 20
	precision = 0
	
	kf = KFold(X.shape[0], n_folds=n_folds)
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		w = Wisard(disable_randomness=disable_ransomness)
		w.fit(X_train, y_train, address_size=address_size, bleaching_threshold=bleaching_threshold)
		y_predicted = w.predict(X_test)
		
		false_positives = np.count_nonzero(y_test - y_predicted)
		precision += (1 - (false_positives / y_test.shape[0]))
	
	precision = precision/n_folds
	return precision

address_size_options = [16]#[2*x for x in range (1,20)]
bleaching_threshold_options = [5]#[x for x in range (40)] 
disable_ransomness_options = [False]#[True, False]

evaluations = np.zeros( (len(address_size_options), len(bleaching_threshold_options), len(disable_ransomness_options)) )

for i in range(evaluations.shape[0]):
	for j in range(evaluations.shape[1]):
		for k in range(evaluations.shape[2]):
			evaluations[i,j,k] = evaluate(address_size_options[i], bleaching_threshold_options[j], disable_ransomness_options[k])
			with open(output_path, "a") as output_file:
				print("%d,%d,%s,%f\n" % (address_size_options[i], bleaching_threshold_options[j], str(disable_ransomness_options[k]), evaluations[i,j,k]))
				output_file.write("%d,%d,%s,%f\n" % (address_size_options[i], bleaching_threshold_options[j], str(disable_ransomness_options[k]), evaluations[i,j,k]))

