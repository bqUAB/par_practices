# The pickle module implements a fundamental, but powerful algorithm for
# serializing and de-serializing a Python object structure.
# "Pickling" is the process whereby a Python object hierarchy is converted
# into a byte stream that we can use e.g. to save data/load into/from a file.
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load data for the regression example
with open('./P0data1.pkl', 'rb') as f:
    (X, y) = pickle.load(f)

# Our dataset is composed by input data (X) and output data (y) pairs.
X.shape, y.shape

# This dataset is known as the "house prices" dataset
# the task to be done is to predict the price of a house given some 'feautres'
# of the house our input data (X) contains two features per sample (size of the
# house, and number of rooms) the output data (y) contains the price of each
# sample. for simplification we are going to use only one feature
# (the size of the house i.e. X[:,0])


# Plot sample points.
# plot dots ('o') for each sample (house size, house price)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], y)

ax.set_xlabel('House Size')
# ax.set_xlabel('House Size (normalized)')
ax.set_ylabel('Number of Rooms')
# ax.set_ylabel('Number of Rooms (normalized)')
ax.set_zlabel('Price($)')
ax.set_title('House Prices')

plt.show()


def GradientDescent(X, y, max_iterations=100, alpha=1):
    m = X.shape[0]  # number of samples

    # y must be a column vector
    y = y.reshape(m, 1)

    # initialize the parameters to zero
    theta = np.zeros(shape=(3, 1))
    # three parameters in this particular case

    # Repeat for max_iterations (it would be nice to also check convergence...)
    for iteration in range(max_iterations):
        grad = np.dot(X.T, (np.dot(X, theta) - y)) / m
        theta = theta - alpha*grad
    return theta


# prepare x with a column of ones (this is the x_0 for the bias term)
x = np.ones(shape=(X.shape[0], 3))
x[:, 1] = X[:, 0]  # include the 1st feature (size)
x[:, 2] = X[:, 1]  # include the 2nd feature (number of rooms)

# Scale features and set them to zero mean (standarize)
mu = np.mean(x, 0)
sigma = np.std(x, 0, ddof=1)
x[:, 1] = (x[:, 1] - mu[1]) / sigma[1]
x[:, 2] = (x[:, 2] - mu[2]) / sigma[2]

theta = GradientDescent(x, y)
print 'theta result:'
print theta

# Plot sample points.
# plot dots ('o') for each sample (house size, house price)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x[:, 1], x[:, 2], y)

x1, x2 = np.meshgrid(range(-2, 4), range(-3, 3))
yy = (-theta[0] * x1 - theta[1] * x2) * 1. / theta[2]
ax.plot_surface(x1, x2, yy)

ax.set_xlabel('House Size (normalized)')
ax.set_ylabel('Number of Rooms (normalized)')
ax.set_zlabel('Price($)')
ax.set_title('House Prices')

plt.show()
