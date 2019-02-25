import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# import data
X_train = np.load('regression_Xtrain.npy')
y_train = np.load('regression_ytrain.npy')
X_test = np.load('regression_Xtest.npy')
y_test = np.load('regression_ytest.npy')


lr = LinearRegression()
MSE_collection = []

# polynomial fit
degree_range = range(1,11)
for degree in degree_range:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    xPoly = poly.fit_transform(X_train.reshape(-1,1))
    lr = LinearRegression()
    lr.fit(xPoly, y_train)
    x_range = np.linspace(-1, 5.5, 100)
    predicted = lr.predict(poly.fit_transform(x_range.reshape(-1,1)))
    mean_square_error = sum((y_test - lr.predict(poly.fit_transform(X_test.reshape(-1,1))))**2)/len(y_test)
    print degree, 'degree polynomial fit: mean squared error = ', mean_square_error
    MSE_collection.append(mean_square_error)

    plt.ylabel('y')
    plt.xlabel('x')
    plt.scatter(X_test, y_test, label="test set")
    plt.scatter(X_train, y_train, c='red', label="train set")
    plt.plot(x_range.reshape(-1, 1), predicted, label="fit")
    plt.legend()
    plt.title(str(degree) + ' degree polynomial fit\nMSE = ' + str(mean_square_error))
    plt.show()

plt.ylabel('MSE')
plt.xlabel('Polynomial degree')
plt.title('Mean square error of used degrees ')
degree_range.pop(), MSE_collection.pop() #10 degree error is too high
plt.plot(degree_range, MSE_collection)
plt.show()

