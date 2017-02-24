import numpy as np
# Load the synthetic dataset
import pickle
# Load the plot library
import matplotlib.pyplot as plt


def sigmoid(X):
    """
    Computes the Sigmoid function of the input argument X.
    """
    return 1.0/(1+np.exp(-X))


# Function to tweak for the first experiment
def GradientDescent(x, y, max_iterations=2500, alpha=0.1):

    m, n = x.shape  # number of samples, number of features

    # y must be a column vector
    y = y.reshape(m, 1)

    # initialize the parameters
    theta = np.ones(shape=(n, 1))

    # Repeat until convergence (or max_iterations)
    for iteration in range(max_iterations):
        h = sigmoid(np.dot(x, theta))
        error = (h-y)
        gradient = np.dot(x.T, error) / m
        theta = theta - alpha*gradient
    return theta


def classifyVector(X, theta):
    """
    Evaluate the Logistic Regression model h(x) with theta parameters,
    and returns the predicted label of x.
    """
    prob = sigmoid(sum(np.dot(X, theta)))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


with open('./PR1data1.pkl', 'rb') as f:
    (X, y) = pickle.load(f)

print X.shape, y.shape

# Plot the data
%matplotlib inline

plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', c='b')  # positive samples
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='x', c='r')  # negative samples
plt.show()
