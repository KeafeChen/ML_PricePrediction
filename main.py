import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy.linalg as la
import matplotlib.pyplot as pp


def linreg(X,y):
    return np.dot(la.pinv(X),y)


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


if __name__ == '__main__':
    # load the data and randomly get 500 pieces of data
    df = pd.read_csv('./kc_house_data.csv')
    data = df.values
    indices = np.random.randint(0, data.shape[0], 500)
    random_sample = data[indices]
    random_sample_df = pd.DataFrame(random_sample, columns=list(df))

    df.to_csv("500.txt", index=False)


    # separate train, validation and test data into 8:1:1, to make it convenient, get first 500 hundred data for now
    df_data = df.head(500)
    train_data, val_test_data = train_test_split(df_data, test_size=0.2)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5)

    # train and test
    feature = ['sqft_living15']
    X_train = np.array(train_data[feature])
    Y_train = np.array(train_data['price']).reshape(-1, 1)
    X_test = np.array(test_data[feature])
    Y_test = np.array(test_data['price']).reshape(-1, 1)

    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)

    print('Intercept: {}'.format(regressor.intercept_))
    print('Coefficients: {}'.format(regressor.coef_))

    # Visualizing the training Test Results
    pp.scatter(X_train, Y_train, color='red')
    pp.plot(X_train, regressor.predict(X_train), color='blue')
    pp.title("Visualization of Training Dataset")
    pp.xlabel("Space")
    pp.ylabel("Price")
    pp.show()

    # Visualizing the Test Results
    pp.scatter(X_test, Y_test, color='red')
    pp.plot(X_test, regressor.predict(X_test), color='blue')
    pp.title("Visualization of testing Dataset")
    pp.xlabel("Space")
    pp.ylabel("Price")
    pp.show()










