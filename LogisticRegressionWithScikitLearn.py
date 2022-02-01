#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 20:46:47 2022

@author: damlagizemsabur
"""

import pandas as pd

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


X_train = training_data.iloc[:,:-1].values
Y_train = training_data.iloc[:, -1].values
    
X_test = test_data.iloc[:,:-1].values
Y_test = test_data.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# model create- log reg

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(random_state=0)
log_model.fit(X_train,Y_train)
    
    
# pred
y_pred = log_model.predict(X_test)

# control
from sklearn.metrics import confusion_matrix, accuracy_score
print("accuracy_score: ",accuracy_score(Y_test, y_pred))
print("confusion_matrix: \n",confusion_matrix(Y_test, y_pred))
    
    

    
    
    
    
    
    
    
    