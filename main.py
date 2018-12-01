import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import metrics
import numpy.linalg as la
import matplotlib.pyplot as pp
from sklearn.cross_validation import cross_val_score, cross_val_predict
import random
# global define X_train, Y_train, X_test, Y_test
global X_train
global Y_train
global X_test
global Y_test

# define the feature
feature = ['sqft_living']
features1 = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']


# define the linear kernel model
alg = SVC(C=1.0, kernel='linear')



def linreg(X,y):
    return np.dot(la.pinv(X),y)

def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)

def kford(k, X, y):
    (n, d) = np.shape(X)
    z = np.zeros((k, 1))
    for i in range(0, k):
        T = set(range(int(np.floor((n * i) / k)), int(np.floor(((n * (i + 1)) / k) - 1)) + 1))
        S = set(range(0, n)) - T
        thetaHat = linreg(X[list(S)], y[list(S)])
        summ = 0
        for t in T:
            summ += (y[t] - np.dot(X[t], thetaHat)) ** 2
        z[i] = (1.0 / len(T)) * summ
    return z

def kernel(X_train, Y_train, X_test, Y_test):
    alg.fit(X_train, Y_train)
    Y_pred = alg.predict(X_test)

    visualizationTesting(X_test, Y_pred)

def visualizationTraining(X_train, Y_train):
    # Visualizing the training Test Results
    pp.scatter(X_train, Y_train, color='red')                   #
    pp.plot(X_train, regressor.predict(X_train), color='blue')
    pp.title("Visualization of Training Dataset")
    pp.xlabel("Space")
    pp.ylabel("Price")
    #pp.show()
    pp.close()

def visualizationTesting(X_test, Y_test):
    # Visualizing the Test Results
    pp.scatter(X_test, Y_test, color='blue')
    pp.plot(X_test, regressor.predict(X_test), color='red')
    pp.title("Visualization of testing Dataset")
    pp.xlabel("Space")
    pp.ylabel("Price")
    #pp.show()
    pp.close()

def loadData(filename=''):
    df = pd.read_csv(filename)
    data = df.values

    indices = np.random.randint(0, data.shape[0], 500)
    random_sample = data[indices]

    random_sample_df = pd.DataFrame(random_sample, columns=list(df))
    df.to_csv("500.txt", index=False)

    # separate train, validation and test data into 8:1:1, to make it convenient, get first 500 hundred data for now
    df_data = df.head(500)

    train_data, test_data = train_test_split(df_data, test_size=0.2)


    return [train_data, test_data, df_data]


def TwoCrossValidation(X, Y):
    print("Two Cross Validation ")

    sample1 = list()
    sample2 = list()
    n = np.shape(X)[0]

    y_pred = np.zeros((n, 1))
    for i in range(int(n / 2)):
        sample1.append(int(i))
        sample2.append(int(i + n / 2))
    X_train = X[sample1]
    Y_train = Y[sample1]
    theta_sample1 = LinearRegression()
    theta_sample1.fit(X_train, Y_train)
    for i in sample2:
        y_pred[i] = (regressor.predict(X[i][0]))


    X_train = X[sample2]
    Y_train = Y[sample2]

    theta_sample2 = LinearRegression()
    theta_sample2.fit(X_train, Y_train)
    for i in sample1:
        y_pred[i] = (regressor.predict(X[i][0]))

    # visualizaing
    pp.scatter(Y, y_pred)
    #pp.show()
    pp.close()

def TwoCrossValidationKernel(X, Y):
    print("Two Cross Validation for linear kernel")

    sample1 = list()
    sample2 = list()
    n = np.shape(X)[0]

    y_pred = np.zeros((n, 1))
    for i in range(int(n / 2)):
        sample1.append(int(i))
        sample2.append(int(i + n / 2))

    alg = SVC(C=1.0, kernel='linear')
    alg.fit(X[sample1], Y[sample1])
    for i in sample2:
        y_pred[i] = (alg.predict(X[i][0]))
    alg = SVC(C=1.0, kernel='linear')
    alg.fit(X[sample2], Y[sample2])
    for i in sample1:
        y_pred[i] = (alg.predict(X[i][0]))
    pp.scatter(Y, y_pred)
    pp.show()
    pp.close()

def bootstrapping(B, X_subset, y_subset, C):
    n = len(X_subset)
    bs_err = np.zeros(B)
    for b in range(B):
        train_samples = list(np.random.randint(0, n, n))
        test_samples = list(set(range(n)) - set(train_samples))
        alg = SVC(C=C, kernel='linear')
        alg.fit(X_subset[train_samples], y_subset[train_samples])
        err = 0
        for i in test_samples:
            err = err + abs(Y_whole[i] - y_pred[i])
        err = err / len(test_samples)
        bs_err[b] = err
    err_res = np.mean(bs_err)
    return err_res



if __name__ == '__main__':

    # load data
    [train_data, test_data, df_data] = loadData('kc_house_data.csv')
    # train, validation, and test data
    X_train = np.array(train_data[feature])
    Y_train = np.array(train_data['price']).reshape(-1, 1)
    X_test = np.array(test_data[feature])
    Y_test = np.array(test_data['price']).reshape(-1, 1)
    X_whole = np.array(df_data[feature])
    Y_whole = np.array(df_data['price']).reshape((-1, 1))

    # define LinearRegression
    regressor = LinearRegression()

    # fit X_train and Y_train
    regressor.fit(np.array(train_data[feature]), np.array(train_data['price']).reshape(-1, 1))
    print('Intercept: {}'.format(regressor.intercept_))
    print('Coefficients: {}'.format(regressor.coef_))
    pred = regressor.predict(test_data[feature])
    # Mean Squared Error (MSE)
    msecm1 = format(np.sqrt(metrics.mean_squared_error(Y_test, pred)), '.3f')
    # Adjusted R-squared (training)
    artrcm1 = format(adjustedR2(regressor.score(train_data[feature], train_data['price']), train_data.shape[0],
                                len(feature)), '.3f')
    # visualization
    visualizationTraining(np.array(train_data[feature]), np.array(train_data['price']).reshape(-1, 1))
    visualizationTesting(np.array(test_data[feature]), pred)

    # twoCrossValidation
    TwoCrossValidation(X_whole, Y_whole)
    '''
    # K-fold cross validation
    # Perform 6-fold cross validation
    scores = cross_val_score(regressor, X_whole, Y_whole, cv=6)
    print("Cross-validated scores: ", scores)

    # Make cross validated predictions
    predictions = cross_val_predict(regressor, X_whole, Y_whole, cv=6)
    pp.scatter(Y_whole, predictions)
    pp.show()'''
    # ------------------------------#
    #                               #
    # Multiple Linear Regression    #
    #                               #
    # ------------------------------#
    complex_model_1 = LinearRegression()
    complex_model_1.fit(np.array(train_data[features1]), np.array(train_data['price']).reshape(-1, 1))

    print('Intercept: {}'.format(complex_model_1.intercept_))
    print('Coefficients: {}'.format(complex_model_1.coef_))

    # Prediction
    pred1 = complex_model_1.predict(test_data[features1])
    # Mean Squared Error (MSE)
    msecm1 = format(np.sqrt(metrics.mean_squared_error(Y_test, pred1)), '.3f')
    # Adjusted R-squared (training)
    artrcm1 = format(adjustedR2(complex_model_1.score(train_data[features1], train_data['price']), train_data.shape[0],
                                len(features1)), '.3f')
    # visualization for Multiple Linear Regression
    print("visualization for Multiple Linear Regression")
    print(np.shape(Y_whole))
    print(np.shape(pred1))
    pp.scatter(np.array(test_data['price']).reshape(-1, 1), pred1)
    #pp.show()
    pp.close()
    # ----------------------------- #
    #                               #
    # Kernel function               #
    #                               #
    # ------------------------------#

    # Training on the whole dataset

    from sklearn.svm import SVC

    alg = SVC(C=1.0, kernel='linear')
    alg.fit(X_whole, Y_whole)
    y_pred = alg.predict(X_whole)
    print("Linear Kernel visualization on whole dataset")
    pp.scatter(Y_whole, y_pred)

    err = 0
    for i in range(len(X_whole)):
        err = err + abs(Y_whole[i] - y_pred[i])
    err = err / len(X_whole)
    print(err)
    #pp.show()

    # Two-fold cross validation
    print("Two Cross Validation for linear kernel")

    sample1 = list()
    sample2 = list()
    n = np.shape(X_whole)[0]

    y_pred = np.zeros((n, 1))
    for i in range(int(n / 2)):
        sample1.append(int(i))
        sample2.append(int(i + n / 2))

    alg = SVC(C=1.0, kernel='linear')
    alg.fit(X_whole[sample1], Y_whole[sample1])
    for i in sample2:
        y_pred[i] = (alg.predict(X_whole[i][0]))
    alg = SVC(C=1.0, kernel='linear')
    alg.fit(X_whole[sample2], Y_whole[sample2])
    for i in sample1:
        y_pred[i] = (alg.predict(X_whole[i][0]))
    pp.scatter(Y_whole, y_pred)
    #pp.show()
    pp.close()

    # Hyperparameter tuning with nested cross validation

    C_list = [10.0]
    B = 30
    best_err = 99999999
    best_C = 0.0
    y_pred = np.zeros(len(X_whole), int)
    for C in C_list:
        err = bootstrapping(B, X_whole[sample1], Y_whole[sample1], C)
        print("C=", C, ", err=", err)
        if(err < best_err):
            best_err = err;
            best_C = C
    print("Best_C =", C)

    alg = SVC(C=best_C, kernel='linear')
    alg.fit(X_whole[sample2],Y_whole[sample2])
    err = 0
    for i in sample1:
        y_pred[i] = (alg.predict(X_whole[i][0]))
        err = err + abs(Y_whole[i] - y_pred[i])

    err = err / len(sample1)
    print(err)