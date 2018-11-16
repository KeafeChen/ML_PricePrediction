import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import metrics
import numpy.linalg as la
import matplotlib.pyplot as pp

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
    pp.show()
    pp.close()

def visualizationTesting(X_test, Y_test):
    # Visualizing the Test Results
    pp.scatter(X_test, Y_test, color='blue')
    pp.plot(X_test, regressor.predict(X_test), color='red')
    pp.title("Visualization of testing Dataset")
    pp.xlabel("Space")
    pp.ylabel("Price")
    pp.show()
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

    train_data, val_test_data = train_test_split(df_data, test_size=0.2)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5)

    return [train_data, val_data, test_data]
if __name__ == '__main__':

    # load data
    [train_data, val_data, test_data] = loadData('kc_house_data.csv')
    # train and test
    X_train = np.array(train_data[feature])
    Y_train = np.array(train_data['price']).reshape(-1, 1)
    X_test = np.array(test_data[feature])
    Y_test = np.array(test_data['price']).reshape(-1, 1)

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
    visualizationTesting(np.array(test_data)[feature], pred)

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

    # visualization
    visualizationTraining(np.array(train_data[features1]), np.array(train_data['price']).reshape(-1, 1))
    visualizationTesting(np.array(test_data)[features1], pred1)

    # ----------------------------- #
    #                               #
    # Kernel function               #
    #                               #
    # ------------------------------#
    kernel(X_train, Y_train, X_test, Y_test)













