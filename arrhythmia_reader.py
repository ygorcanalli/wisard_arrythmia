# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:43:31 2015

@author: canalli
"""

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

path = "arrhythmia.data"
numerical_features = [0] + [x for x in range(2,21)] + [x for x in range(27,33)] + [x for x in range(39,45)] + [x for x in range(51,57)] + [x for x in range(63,69)] + [x for x in range(75,81)] + [x for x in range(87,93)] + [x for x in range(99,105)] + [x for x in range(111,117)] + [x for x in range(123,129)] + [x for x in range(135,141)] + [x for x in range(147,153)] + [x for x in range(159,279)]
categorical_features = [1] + [x for x in range(21,27)] + [x for x in range(33,39)] + [x for x in range(45,51)] + [x for x in range(57,63)] + [x for x in range(69,75)] + [x for x in range(81,87)] + [x for x in range(93,99)] + [x for x in range(105,111)] + [x for x in range(117,123)] + [x for x in range(129,135)] + [x for x in range(141,147)] + [x for x in range(153,159)]
classes_column = [279]
delimiter = ","
missing = "?"
header_size = 0

'''
def thermometer_encoding2(data, size):
	encoded_data = np.chararray( data.shape, itemsize=size)
	for j in range(data.shape[1]):
		c = data[:,j]
  
		occurrences, bins = np.histogram(c, size)
		levels = np.digitize(c, bins)

		for i in range(levels.shape[0]):
			encoded_data[i,j] = ("1"*levels[i]).zfill(size)

	return encoded_data
'''
def thermometer_encoding(data, size):
	encoded_data = []
	for j in range(data.shape[1]):
		c = data[:,j]
		encoded_column = np.zeros( (c.shape[0], size) )
		
		occurrences, bins = np.histogram( c, size )
		levels = np.digitize(c, bins)

		for i in range(levels.shape[0]):
			encoded_column[i,-levels[i]:-1] = 1

		encoded_data.append(encoded_column)
		
	return np.hstack(encoded_data).astype(int)

def parse_categorical_missing_values(data, missing_values="?"):
	
	if len(np.shape(data)) == 1:
		unique,pos = np.unique(data,return_inverse=True) #Finds all unique elements and their positions
		counts = np.bincount(pos)			 #Count the number of each unique element
		maxpos = counts.argmax()
		most_freq = unique[maxpos]
		data = np.core.defchararray.replace(data, missing_values, most_freq)
	else:	
		m = data.shape[1]
		for j in range(m):
			#Finds all unique elements and their positions
			unique,pos = np.unique(data[:,j],return_inverse=True)
			#Count the number of each unique element
			counts = np.bincount(pos)
			maxpos = counts.argmax()
			most_freq = unique[maxpos]
			data[:,j] = np.core.defchararray.replace(data[:,j], missing_values, most_freq)

def read_numerical_features(thermometer_size=8):
	# read numerical features
	numerical = np.genfromtxt(path, dtype=np.float, usecols=numerical_features,delimiter=delimiter,skip_header=header_size)
	
	# Count NaN
	nan_count = np.sum(np.isnan(numerical))
	print("Numerical original shape: %s" % str(numerical.shape))
	print("NaN count: %d" % nan_count)
	print("NaN percentage: %f" % ((100*nan_count)/(numerical.shape[0]*numerical.shape[1])))
	
	# replace missing values with mean value
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	numerical = imp.fit_transform(numerical)
	#normalize numerical features to non-negative
	min_max_scaler = preprocessing.MinMaxScaler( (0,1) )
	normalized_numerical = min_max_scaler.fit_transform(numerical)
	# encode numerical features into thermometer unary code
	encoded_numerical = thermometer_encoding(normalized_numerical, thermometer_size)

	return encoded_numerical

def read_categorical_features(thermometer_size=8):
	# read categorical features
	categorical = np.genfromtxt(path, dtype=np.unicode, usecols=categorical_features,delimiter=delimiter,skip_header=header_size)
	
	# Count NaN
	nan_count = np.sum(categorical == '?')
	print("Categorical original shape: %s" % str(categorical.shape))
	print("NaN count: %d" % nan_count )
	print("NaN percentage: %f" % ((100*nan_count)/(categorical.shape[0]*categorical.shape[1])))	
	
	# replace missing values to most frequent value
	parse_categorical_missing_values(categorical, missing_values=missing)
	
	# transform categorical features in sequential integer categorical features, for each collumn
	numbered_categories = np.zeros(np.shape(categorical))
	m = np.shape(categorical)[1]
	for j in range(m):
		le = preprocessing.LabelEncoder()
		numbered_categories[:,j] = le.fit_transform(categorical[:,j])
		
	# one hot categorical encoding
	ohe = preprocessing.OneHotEncoder()
	normalized_categorical = ohe.fit_transform(numbered_categories).toarray()
		
	return normalized_categorical

def read_all_features(thermometer_size=8):
	numerical = read_numerical_features(thermometer_size)
	categorical = read_categorical_features()
	
	# join features
	features = np.hstack( (categorical, numerical) )
	return features

def read_classifications():
	# read classes
	classifications = np.genfromtxt(path, dtype=np.unicode, usecols=classes_column,delimiter=delimiter,skip_header=header_size)
	le = preprocessing.LabelEncoder()
	numbered_classifications = le.fit_transform(classifications)
	
	return le.classes_, numbered_classifications.astype(int)

def read_simplified_classifications():
	# read classes
	classifications = np.genfromtxt(path, dtype=np.unicode, usecols=classes_column,delimiter=delimiter,skip_header=header_size)
	classifications = classifications != "1"	
	le = preprocessing.LabelEncoder()
	numbered_classifications = le.fit_transform(classifications)
	
	return le.classes_, numbered_classifications.astype(int)
