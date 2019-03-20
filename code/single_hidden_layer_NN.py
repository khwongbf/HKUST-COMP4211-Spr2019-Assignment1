# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:32:39 2019

@author: Wong Kwan Ho
"""

# Required imports for all tasks
from __future__ import print_function
import numpy as np
import seaborn as sns

# Imports for presentation
import matplotlib.pyplot as plt

# Import for splitting the data, if needed
from sklearn.model_selection import train_test_split

# To calculate R^2 score for linear regression
from sklearn import metrics

# Import libraries for single layer neural network
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Import for accuracy for Logistic regression
from sklearn.metrics import accuracy_score

# Import for learning curve
from sklearn.model_selection import learning_curve

# For measuring time
import time

# FIFA data
fifa = np.load('fifa.npz')

# Finance data
finance = np.load('finance.npz')

# Orbits data
orbits = np.load('orbits.npz')

"""
    Creates a new single layer neural network model
    
    @param hidden_units : The number of hidden units in the hidden layer
    @param learning_rate_init : The initial learning rate used. It controls the step-size in updating the weights.
    
    @return a new instance of the model
"""

def create_single_layer_classifier(hidden_units, learning_rate, learning_rate_init):
    return MLPClassifier(hidden_layer_sizes=(hidden_units, ), activation='logistic', solver='sgd', learning_rate=learning_rate, learning_rate_init=learning_rate_init, verbose=True, early_stopping=True, validation_fraction=0.2)

"""
    Creates a new set of parameters based on the number of units in the single layered NN model
    
    @param max_hidden_unit_count = 10 : The maximum number or hidden units in a layer
"""
def set_tuned_parameters(max_hidden_unit_count = 10):
    tuned_parameters = []
    
    learning_rate_inits = (0.9, 0.1)
    
    for i in range (1 , max_hidden_unit_count + 1):
        for learning_rate_init in learning_rate_inits:
            tuned_parameters.append({'hidden_layer_sizes': [(i,)], 'learning_rate_init' : [learning_rate_init]})
                
    return tuned_parameters

"""
    Creates a function for hyperparameter tuning.
    
    @param X : The inputs (and features) for the training and validation set combined
    @param y : The outputs (and features) for the training and validation set combined
    @param validation_set_fraction : The fraction of the validation set. This must be a float in the interval [0,1)
    @param name : The name of the dataset, used for printing only.
    
    @return : The classifier object after tuning
"""
def hyperparameter_tuning(X, y , file, validation_set_fraction = 0.2, name=""):
    # Split the dataset in two parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_set_fraction, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = set_tuned_parameters(max_hidden_unit_count = 10)

    file.write("# Tuning hyper-parameters for " + name + " ... \n")
    file.write("\n")
    
    mlp = MLPClassifier(activation='logistic', max_iter=1000, solver='sgd', verbose=False, tol=1e-4, learning_rate = 'constant', random_state=1)
    print ("Grid Search Start")
    clf = GridSearchCV(mlp, tuned_parameters, cv=5, verbose = 5)
    
    print("Grid Search Done!")
    clf.fit(X_train, y_train)

    print("Fitting done!")
    file.write("Best parameters set found on development set for " + name + " : \n")
    file.write("\n")
    file.write(str(clf.best_params_))
    file.write("\n\n")
    file.write("Grid scores on development set for " + name + " : \n")
    file.write("\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        file.write("%0.3f (+/-%0.03f) for %r \n"
              % (mean, std * 2, params))
    file.write("\n")

    file.write("Detailed classification report for " + name + " : \n")
    file.write("\n")
    file.write("The model is trained on the full development set.\n")
    file.write("The scores are computed on the full evaluation set.\n")
    file.write("\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    file.write(str(metrics.classification_report(y_true, y_pred)))
    file.write("\n\n")
    
    return clf, file
    
    """
    Plots the learning curve against samples, using the training error and the Cross-validation Error.
    
    @param estimator : the classifier that was used for training.
    @param title : the title of the figure of the learning curve
    @param X : The training input.
    @param y : The expected training output.
    
    @return : the pyplot instance of the learning curve.
"""
def plot_learning_curve(estimator, title, X, y, name):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training samples")
    plt.ylabel("Error")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, scoring = 'neg_log_loss', cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training error")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation error")

    plt.legend(loc="best")
    plt.savefig("single_hidden_layer_NN_"+name+".png")
    return plt

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
        plt.savefig("single_hidden_layer_NN_conf_matrix_training_" + name +".png")
    else:
        plt.savefig("single_hidden_layer_NN_conf_matrix_test_" + name +".png")
    return plt

"""
    Performs the neural network task.
    
    @param dataset : the dataset used.
    @param name : the name of the dataset. Used for I/O only.
    
"""
def do_single_layer_neural_network(dataset, name):
    
    #Open the file
    single_hidden_layer_file = open("single_hidden_layer_NN_output_"+name+".txt", 'w+')
    
    # start the timer
    start = time.clock()
    
    # search by hyperparameter and cross validation
    cls, single_hidden_layer_file = hyperparameter_tuning(X = dataset['train_X'], y = dataset['classification_train_Y'], file = single_hidden_layer_file, name=name )
    
    # Predict the training sets 
    predict_train = cls.predict(dataset['train_X'])
    predict_test = cls.predict(dataset['test_X'])
    
    # Obtains the log-loss for both datasets
    log_loss_train = metrics.log_loss(dataset['classification_train_Y'], predict_train)
    log_loss_test = metrics.log_loss(dataset['classification_test_Y'], predict_test)
    
    single_hidden_layer_file.write("Cross-entropy loss for "+ name +" training data set, Neural Networks model = " + str(log_loss_train) + "\n")
    single_hidden_layer_file.write("Cross-entropy loss for "+ name +" test data set, Neural Networks model = " + str(log_loss_test) + "\n")
    
    # Compute the best log-loss as the minimum of both log-losses
    best_loss = min(log_loss_train, log_loss_test)
    
    single_hidden_layer_file.write("Hence, the best loss for "+ name +" data set using Neural Networks model = " + str(best_loss) + "\n")
    
    # Plot the learning curve for training and validation sets, to visualize the performance over number of samples (time)
    plot_learning_curve(cls, name + " dataset learning curve, Neural Networks model", dataset['train_X'], dataset['classification_train_Y'], name)
    
    # Plot the normalized confusion matrix
    cm_training = metrics.confusion_matrix(dataset['classification_train_Y'], predict_train)
    cm_test = metrics.confusion_matrix(dataset['classification_test_Y'], predict_test)
    
    cm_normalized_training = cm_training.astype('float') / cm_training.sum(axis=1)[:, np.newaxis]
    cm_normalized_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
    
    cls_train_score = cls.score(dataset['train_X'], dataset['classification_train_Y'])
    cls_test_score = cls.score(dataset['test_X'], dataset['classification_test_Y'])
    
    plot_conf_matrix(cm_normalized_training, name + " training set, Neural Networks", cls_train_score, True, name)
    plot_conf_matrix(cm_normalized_test, name + " test set, Neural Networks", cls_test_score, False, name)
    
    # stop the timer
    end = time.clock()
    
    # record the time
    single_hidden_layer_file.write("Elapsed time: " + str(end-start) + " seconds." + "\n" )
    
    # flush changes
    single_hidden_layer_file.close()
    
    return

do_single_layer_neural_network(fifa, "FIFA")

do_single_layer_neural_network(finance, "FINANCE")

do_single_layer_neural_network(orbits, "ORBITS")