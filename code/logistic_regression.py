# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:11:10 2019

@author: Wong Kwan Ho
"""
# Required imports for all tasks
from __future__ import print_function
import numpy as np
import seaborn as sns

# Imports for presentation
import matplotlib.pyplot as plt

# For confusion matrix
from sklearn import metrics

# Import libraries for Logistic Regression using gradient descent
from sklearn.linear_model import SGDClassifier

# Import for accuracy for Logistic regression
from sklearn.metrics import accuracy_score

# For measuring time
import time

"""
    Load the datasets
"""
# FIFA data
fifa = np.load('fifa.npz')

# Finance data
finance = np.load('finance.npz')

# Orbits data
orbits = np.load('orbits.npz')

# Set the seaborn module
sns.set()

"""
    Plots the confusion matrix for logistic regression.
    
    @param cm - the confusion matrix
    @param title - the title of the metrix
    @param score - the accuracy score of the logistic regression
    @param training - a boolean flag indicates whether the confusion matrix is training or test. True when training, False when test
    @param name - the name of the dataset
    @return an pyplot instance containing the drawn figure
"""
def plot_conf_matrix(cm, title, score, training, name):
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    title += ', Accuracy Score: {:.4f}'.format(score) 
    plt.title(title, size = 15);
    if training:
        plt.savefig("logistic_regression_conf_matrix_training_" + name +".png")
    else:
        plt.savefig("logistic_regression_conf_matrix_test_" + name +".png")
    return plt

"""
    Performs the logistic regression task.
    
    @param dataset - the dataset concerned
    @param name - the name of the dataset
    @param eta0 - the initial learning rate of the logistic regression
"""
def logistic_regression_task(dataset, name, eta0):
    # Output to a file
    log_reg_output = open("logistic_regression_output_"+name+".txt", 'w+')
    
    # start the timer
    start = time.clock()
    
    # build the model with given starting learning rate
    clf_sgd = SGDClassifier(loss = 'log', max_iter = 1000, tol= 1e-3, verbose=1, learning_rate='adaptive', eta0=eta0, early_stopping=True)
    
    # fit the model with the given dataset
    clf_sgd.fit(dataset['train_X'], dataset['classification_train_Y'])
    
    # Predict the model output (training and test sets) with the given inputs
    predict_train = clf_sgd.predict(dataset['train_X'])
    predict_test = clf_sgd.predict(dataset['test_X'])
    
    # Obtain the accuracy score for the training and test sets
    clf_sgd_train_score = clf_sgd.score(dataset['train_X'], dataset['classification_train_Y'])
    clf_sgd_test_score = clf_sgd.score(dataset['test_X'], dataset['classification_test_Y'])

    log_reg_output.write("Accuracy for Logistic Regression with Stochastic Gradient Descent, "+ name +" training set: " + str(clf_sgd_train_score) + "\n")
    log_reg_output.write("Accuracy for Logistic Regression with Stochastic Gradient Descent, "+ name +" test set: " + str(clf_sgd_test_score) + "\n")
    
    # Set up the confusion matrices
    cm_training = metrics.confusion_matrix(dataset['classification_train_Y'], predict_train)
    cm_test = metrics.confusion_matrix(dataset['classification_test_Y'], predict_test)
    
    # Normalize the confusion matrices
    cm_normalized_training = cm_training.astype('float') / cm_training.sum(axis=1)[:, np.newaxis]
    cm_normalized_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
    
    # plot the confusion matrices and save as figure
    plt_train = plot_conf_matrix(cm_normalized_training, name + " training data Gradient Descent", clf_sgd_train_score, True, name)
    
    plt_test = plot_conf_matrix(cm_normalized_test, name + " test data Gradient Descent", clf_sgd_test_score, False, name)
    
    # stop the timer
    end = time.clock()
    
    # record the time
    log_reg_output.write("Elapsed time: " + str(end-start) + " seconds." + "\n" )
    
    # flush changes
    log_reg_output.close()

# Logistic Regression for fifa dataset
logistic_regression_task(fifa, "FIFA", 0.9)

# Logistic Regression for finance dataset
logistic_regression_task(finance, "FINANCE", 0.9)

# Logistic Regression for orbits dataset
logistic_regression_task(orbits, "ORBITS", 0.9)