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

ax.set_xlabel('House Size (normalized)')
ax.set_ylabel('Number of Rooms (normalized)')
ax.set_zlabel('Price($)')
ax.set_title('House Prices')

plt.show()
