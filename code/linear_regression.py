# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:31:19 2019

Performs Linear Regression Tasks

@author: Wong Kwan Ho
"""

# Required imports for all tasks
from __future__ import print_function
import numpy as np
import seaborn as sns

# Imports for presentation
import matplotlib.pyplot as plt

# To calculate R^2 score for linear regression
from sklearn import metrics

# Import libraries for linear regression
from sklearn.linear_model import LinearRegression

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
    Performs a linear regression task.
    
    @param dataset : the dataset to be performed.
    @param name : the name of the dataset. Must be type <str>
"""
def linear_regression_task(dataset, name):
    # For output to file
    lin_reg_output = open("linear_regression_output_"+ name +".txt", 'w+')
    
    # start the timer
    start = time.clock()
    
    # Build the model
    lin_regression_model = LinearRegression()
    
    # fit the model with data
    lin_regression_model.fit(dataset['train_X'], dataset['regression_train_Y'])
    
    # Predicts the trainng and test set with fitted regression
    predictTrain = lin_regression_model.predict(dataset['train_X'])
    predictTest = lin_regression_model.predict(dataset['test_X'])
    
    # Obtains the R^2 scores for the regression
    r2Train = metrics.r2_score(dataset['regression_train_Y'], predictTrain)
    r2Test = metrics.r2_score(dataset['regression_test_Y'], predictTest)
    
    lin_reg_output.write("Coefficient of determination for "+ name +" training set using linear regression: " + str(r2Train) +"\n")
    lin_reg_output.write("Coefficient of determination for "+ name +" test set using linear regression: " + str(r2Test) + "\n")
    
    # Obtains the mean square error for training and test sets
    mse_train = metrics.mean_squared_error(dataset['regression_train_Y'], predictTrain)
    mse_test = metrics.mean_squared_error(dataset['regression_test_Y'], predictTest)
    
    lin_reg_output.write("Mean squared error for "+ name +" training set using linear regression: " + str(mse_train) +"\n")
    lin_reg_output.write("Mean squared error for "+ name +" test set using linear regression: " + str(mse_test) +"\n")
    
    # Calculates the squared error for each data point in test set
    # First calculate the linear error for each data point
    linear_error_test = np.subtract(predictTest, dataset['regression_test_Y'])
    
    # Square each error
    squared_error_test = np.square(linear_error_test)
    
    # Set the parameters for the histogram
    # Set the size of figure
    plt.figure(num=None, figsize=(5, 5), dpi=160, facecolor='w')
    
    sns.distplot(squared_error_test, hist=True, kde=False)
    
    # Plot the histogram
    plt.xlabel("Squared Error Value")
    plt.ylabel("Count")
    plt.title("Distribution of the Squared Error Values : "+ name +" dataset")
    plt.savefig("linear_reg_histogram_"+name+".png")
    
    # stop the timer
    end = time.clock()
    
    # record the time
    lin_reg_output.write("Elapsed time: " + str(end-start) + " seconds." + "\n" )
    
    # flush changes
    lin_reg_output.close()

# Perform task for dataset fifa
linear_regression_task(fifa, "FIFA")

# Perform task for dataset finance
linear_regression_task(finance, "FINANCE")

# Perform task for dataset orbits
linear_regression_task(orbits, "ORBITS")