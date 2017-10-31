# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:15:41 2015

@author: canalli
"""

import arrhythmia_reader as reader
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from wisard import Wisard

def find_n_divisors(number, n):
	divisors = []
	for i in range(2,number+1):
		if number % i == 0:
			divisors.append(i)
			if len(divisors) == n:
				return divisors
	return divisors		

def split_gender(X, y, genders):
	X_man = X[genders == 0]
	X_woman = X[genders == 1]
	
	y_man = y[genders == 0]
	y_woman = y[genders == 1]
	
	return X_man, X_woman, y_man, y_woman

def evaluate(X, y, genders, address_size, n_folds=10):
	
	X_man, X_woman, y_man, y_woman = split_gender(X, y, genders)
	y_test_man = []
	y_predicted_man = []
	y_test_woman = []
	y_predicted_woman = []
	accuracys = np.zeros(n_folds)
	
	kf = KFold(X_man.shape[0], n_folds=n_folds)
	for train_index, test_index in kf:
		X_train, X_test = X_man[train_index], X_man[test_index]
		y_train, y_test = y_man[train_index], y_man[test_index]
		
		w = Wisard()
		w.fit(X_train, y_train, address_size=address_size)
		y_predicted = w.predict(X_test)
		
		y_test_man.append(y_test)
		y_predicted_man.append(y_predicted)
		
	
	kf = KFold(X_woman.shape[0], n_folds=n_folds)
	for train_index, test_index in kf:
		X_train, X_test = X_woman[train_index], X_woman[test_index]
		y_train, y_test = y_woman[train_index], y_woman[test_index]
		
		w = Wisard()
		w.fit(X_train, y_train, address_size=address_size)
		y_predicted = w.predict(X_test)
		
		y_test_woman.append(y_test)
		y_predicted_woman.append(y_predicted)
		
	for i in range(n_folds):
		y_test = np.hstack( (y_test_man[i],y_test_woman[i]) )
		y_predicted = np.hstack( (y_predicted_man[i],y_predicted_woman[i]) )
		accuracys[i] = accuracy_score(y_test,y_predicted)

	return np.mean(accuracys)
	
	
output_path = 'results.csv'

n_address_size_options = 5
thermometer_size_options = [x for x in range (1,2)]
evaluations = np.zeros( (len(thermometer_size_options), n_address_size_options) )

with open(output_path, "a") as output_file:
		output_file.write("\n====Begin evaluation====")
		
for i in range(len(thermometer_size_options)):
	
	X = reader.read_categorical_features(thermometer_size=thermometer_size_options[i])
	genders = reader.read_categorical_features()[:,1]
	classes, y = reader.read_classifications()
	
	address_size_options = find_n_divisors(X.shape[1], n_address_size_options)
	
	with open(output_path, "a") as output_file:
		output_file.write("\nThermometer size: %s" % str(thermometer_size_options[i]))
		output_file.write("\nShape: %s" % str(X.shape))
		output_file.write("\nAddress size options: %s" % str(address_size_options))
	
	for j in range(len(address_size_options)):
			evaluations[i,j] = evaluate(X, y, genders, address_size_options[j])
			with open(output_path, "a") as output_file:
				output_file.write("\naddress_size=%d: %f" % (address_size_options[j], evaluations[i,j]))
				

