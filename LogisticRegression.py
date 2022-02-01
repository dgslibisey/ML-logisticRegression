#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 01:22:50 2022

@author: damlagizemsabur
"""

import numpy as np    #importing the library for scientific computation
import matplotlib.pyplot as plt     #importing the library used for plots



class LogisticRegression:
    
    # init function with learning rate and number of iteration parameters
    def __init__(self, learning_rate, number_of_iteration):
       self.learning_rate = learning_rate
       self.number_of_iteration = number_of_iteration
   
    #for weight and bias' shapes determine, 
    def weight_bias_function(self, dim): 
        w = np.zeros((dim,1))
        b = 0
        return w, b
    
    # sigmoid function return 1/(1 + np.exp(-z))
    def _sigmoid(self,z):
        s = 1/(1 + np.exp(-z)) 
        return s

    # loss(cost) function
    def _cost(self,A,y,m):     
        
        cost = -np.sum(y*np.log(A)+ (1-y)*np.log(1-A))/m 
        # cost = np.squeeze(cost)   
        return cost
    
    # gradient descent 
    def fit(self, w, b, X, y): 
        cost_list=[] # for plot cost vs number of iteration 
        iter_list=[] # for plot cost vs number of iteration 
        for i in range(self.number_of_iteration):
            m = X.shape[1] # m is the number of observation
            A = self._sigmoid(np.dot(w.T,X)+b) #y_hat
            cost = self._cost(A,y,m) # it calls cost method and calculate the cost for each iteration
            dw = np.dot(X,(A-y).T)/m # derivate for w
            db = np.sum(A-y)/m # derivate for w
 

            w = w - (self.learning_rate * dw) # at each iteration weight be updated
            b = b - (self.learning_rate * db) # at each iteration bias be updated
    
            cost_list.append(cost)
            iter_list.append(i)
            
            # Print the cost every 100 training iterations
            if(i % 1000 == 0):
                print(f'Loss(cost) for {i}. iterations : {cost}')
 
 
            params = {"w": w,
                  "b": b}
 
            grads = {"dw": dw,
                "db": db}
        
        #to plot cost vs number of iterations figure
        plt.plot(iter_list, cost_list) 
        plt.ylabel('Loss(cost)')
        plt.xlabel('number_of_iteration')
        plt.title('number_of_iteration')
        plt.title('Learning rate = ' + str(self.learning_rate))
        plt.show() 
        
        

            
        return params, grads, cost_list
   


    def train_model(self, X, y, X_test, y_test):

        # initialize parameters with zeros 
        dim = np.shape(X)[0]
        w, b = self.weight_bias_function(dim)
        # Gradient descent 
        parameters, grads, costs = self.fit(w, b, X, y)
 
        # Retrieve parameters w and b from dictionary "parameters"
        self.w = parameters["w"]
        self.b = parameters["b"]
 
        # Predict test/train set examples 
        self.y_prediction = self.predict(X_test)
        
        
        return parameters


    def predict(self,X):
        X = np.array(X)
        m = X.shape[1] # m is the number of observation
        
        # from x test data, we create a y_prediction shape
        y_prediction = np.zeros((1,m))
  
        w = self.w.reshape(X.shape[0], 1)
        b = self.b
        A = self._sigmoid(np.dot(w.T,X)+b) 
        
        # for prediction each line of all data, we use A[0,i], and try to predicted data make equal to 0 or 1.
        for i in range(A.shape[1]):
            if A[0,i] > 0.5: #driver mutation
                y_prediction[0,i] = 1
            else: 
                y_prediction[0,i] = 0 #passenger mutation
   
        return y_prediction
 
