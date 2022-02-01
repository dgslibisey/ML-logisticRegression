#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 01:22:50 2022

@author: damlagizemsabur
"""

import numpy as np    #importing the library for scientific computation
import pandas as pd   #importing the library for data manipulation and storage 
from contextlib import suppress #for the libraries Exceptions supress
with suppress(Exception):

    from LogisticRegression import LogisticRegression  #to reach LogisticRegression class from the other .py file
    
    training_data = pd.read_csv('data_ready_for_ML_GENIE.txt', delimiter=' ')
    test_data = pd.read_csv('data_ready_for_ML_TCGA.txt', delimiter=' ')
    
    
    training_data['labels_driver_mutation'] = training_data['labels_driver_mutation'].replace(['passenger'], 0)
    training_data['labels_driver_mutation'] = training_data['labels_driver_mutation'].replace(['driver'], 1)
    training_data = training_data.replace(to_replace=['yes', 'no'], value=[1,0])
    
    test_data['labels_driver_mutation'] = test_data['labels_driver_mutation'].replace(['passenger'], 0)
    test_data['labels_driver_mutation'] = test_data['labels_driver_mutation'].replace(['driver'], 1)
    test_data = test_data.replace(to_replace=['yes', 'no'], value=[1,0])
    
    #print(df.isnull().sum())  #is there any missing/null values
    #print(test.isnull().sum())  #is there any missing/null values
    
    
    # filling the missing value with column's mean'
    updated_test = test_data
    updated_test['SIFT']=updated_test['SIFT'].fillna(training_data['SIFT'].mean())
    updated_test['PolyPhen']=updated_test['PolyPhen'].fillna(training_data['PolyPhen'].mean())
    updated_test['Condel']=updated_test['Condel'].fillna(training_data['Condel'].mean())
    updated_test['average_rank']=updated_test['average_rank'].fillna(training_data['average_rank'].mean())
    #updated_test.info()
    #print(df.isnull().sum())  #is there any missing/null values
    #print(test.isnull().sum())  #is there any missing/null values
    
    
    
    # for data normalization ---  lambda function is a small anonymous function, can take any number of arguments, but can only have one expression.
    # in this line, lambda used for data normalization formula.
    X_train = training_data.iloc[:,:-1].apply(lambda x: (x-x.mean())/ x.std(), axis=0).values
    y_train = training_data.iloc[:, -1].values
        
    X_test = test_data.iloc[:,:-1].apply(lambda x: (x-x.mean())/ x.std(), axis=0).values
    y_test = test_data.iloc[:, -1].values
    
    # print(X_train.shape) (186931, 51)
    X_train=np.transpose(X_train)
    # print(X_train.shape)

    X_test=np.transpose(X_test)
    
    
    def performance_metrics(y_test, y_pred):
             
        # True Positive (TP): we predict 1(driver), and the true label is 1(driver).
        TP = np.sum(np.logical_and(y_pred == 1, y_test == 1))
         
        # True Negative (TN): we predict 0(passenger), and the true label is 0(passenger).
        TN = np.sum(np.logical_and(y_pred == 0, y_test == 0))
         
        # False Positive (FP): we predict a label of 1 (driver), but the true label is 0(passenger).
        FP = np.sum(np.logical_and(y_pred == 1, y_test == 0))
         
        # False Negative (FN): we predict a label of 0(passenger), but the true label is 1(driver).
        FN = np.sum(np.logical_and(y_pred == 0, y_test == 1))
        
        
        # accuracy : True predict/ All prediction
        accuracy = 100*((TP+TN)/(TP+TN+FP+FN))
        
        #Recall (Sensitivity) : TP/(TP+FN)
        recall = 100*(TP/(TP+FN))
        
        precision = 100*(TP/(TP+FP))
        
        specificity = 100*(TN/(TN+FP))
        
        confusion_matrix = np.matrix([[TN,FN], [FP,TP]])
         
        print('\nAccuracy: %f %%\nTrue Positive: %i\nFalse Positive: %i\nTrue Negative: %i\nFalse Negative: %i\nRecall (Sensitivity): %f %%\nPrecision: %f %%\nSpecificity: %f %%\n' %(accuracy, TP,FP,TN,FN, recall, precision, specificity))

        print('Confusion_matrix:\n ',np.matrix([[TN,FN], [FP,TP]]))
        
        #for check the values size is equal to each other.
        #print('total:', np.size(y_pred), np.size(y_test))

        
        return TP, TN, FP, FN, accuracy, recall, precision, specificity, confusion_matrix
    
    
    
    driver = LogisticRegression(learning_rate=0.005, number_of_iteration=10000)
    a = driver.train_model(X_train, y_train, X_test, y_test)
    predictions = driver.predict(X_test)
    weight_list = a["w"]
    
    
    feature_names=[]
    for i in training_data.head(0):
        feature_names.append(i)
    
    feature_names.pop() # labels pop

    
    new_weight = []
    for i in weight_list:
        new_weight.append(i)
   
        
    weights_with_feature_names = zip(feature_names, new_weight)
    print ('\nWeigths in descending order with feature names \n', sorted(weights_with_feature_names, key=lambda x:x[1][0], reverse = True))
        
    
    performance_metrics(y_test, predictions)
        

