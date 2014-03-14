#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
Uses Principle Componenet Analysis and Grid Search
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import decomposition
from sklearn import cross_validation
from sklearn import grid_search as gs
from sklearn import metrics


def decomposition_pca(train, test):
    """ Linear dimensionality reduction """
    pca = decomposition.PCA(n_components=12, whiten=True)
    train_pca = pca.fit_transform(train)
    test_pca = pca.transform(test)
    return train_pca, test_pca


def split_data(X_data, y_data):
    """ Split the dataset in train and test """
    split_data = cross_validation.train_test_split(X_data, y_data, 
                                                   test_size=0.1, 
                                                   random_state=0
                                                   )
    return split_data


def grid_search(y_data):
    c_range = 10.0 ** np.arange(6.5,7.5,.25)
    gamma_range = 10.0 ** np.arange(-1.5,0.5,.25)
    params = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': c_range}]

    cvk = cross_validation.StratifiedKFold(y_data, n_folds=5)
    return gs.GridSearchCV(svm.SVC(), params, cv=cvk)


def train(features, result):
    """ Use features and result to train Support Vector Machine"""
    clf = grid_search(result)
    clf.fit(features, result)
    return clf


def predict(clf, features):
    """ Predict labels from trained CLF """
    return clf.predict(features).astype(np.int)


def show_score(clf, X_test, y_test):
    """ Scores are computed from the test set """
    y_pred = predict(clf, X_test)
    print metrics.classification_report(y_test.astype(np.int), y_pred)


'''
Id,Solution
1,0
2,1
3,1
...
9000,0
Your prediction should be a 9000 x 1 vector of ones or zeros.
 You also need an Id column (1 to 9000) and should include a header. 
'''


def main():

    X_data = pd.read_csv('data/train.csv', header = None)
    test_data = pd.read_csv('data/test.csv', header = None)
    y_data = pd.read_csv('data/trainLabels.csv', header = None)

    # create columns and index for output pandas dataframe
    columns = ['Solution']
    # Index as per the requirements:
    index = np.arange(1, 9001, 1)
    
    # Change from pandas dataframes to numpy arrays
    # You can use df.single_column.values or df['single_column'].values 
    # to get the underlying numpy array of your series
    X_data = np.array(X_data)
    test_data = np.array(test_data)
    # 1-d arrays don't work this way:
    #y_data = np.array(y_data) 

    print 'changed to numpy arrays'

    X_data, test_data = decomposition_pca(X_data, test_data)

    print 'done with pca'

    X_train, X_test, y_train, y_test = split_data(X_data, y_data[0].values)
    clf = train(X_train, y_train)

    print 'creating score'

    show_score(clf, X_test, y_test)

    print 'starting prediction'

    submission = predict(clf, test_data)

    print submission.shape

    df = pd.DataFrame(data=submission.T, index=index, columns=columns)
    #df = pd.DataFrame(index=test_colors.iloc[:,0], columns=solutions.columns)

    # write dataframe to .csv file
    df.to_csv('pca_grid_submission.csv', header='true', index='true', index_label= 'Id')

if __name__ == "__main__":
    main()
