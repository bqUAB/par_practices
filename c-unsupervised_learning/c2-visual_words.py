"""Effect of the number of Visual Words."""
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt


# Function (modules) declarations
def sigmoid(X):
    """Return the Sigmoid of X.

    Computes the Sigmoid function of the input argument X.
    """
    return 1.0 / (1 + np.exp(-X))


def GradientDescent(x, y, max_iterations=2500, alpha=0.1):
    """Return theta."""
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


def classifyVector(X, theta, probability):
    """Return predicted label of X.

    Evaluate the Logistic Regression model h(x) with theta parameters,
    and returns the predicted label of x.
    """
    prob = sigmoid(sum(np.dot(X, theta)))
    if prob > probability:
        return 1.0
    else:
        return 0.0


# ---------------------------------------------
# Variables initialization
step_size = 8
X = np.zeros(0)
Y = np.zeros(0)


# Create a varying vocabulary size
while step_size <= 256:
    # Data creation
    X = np.append(X, step_size)
    # Import the train/test data (raw image pixels).
    with gzip.open('./BoW-' + str(step_size) + '-8.pklz', 'rb') as f:
        (train_labels, train_images, test_labels, test_images) = pickle.load(f)

    # Logistic Regression gradient descent optimization
    w_bow = GradientDescent(train_images, train_labels)

    H_bowTr = ([
        classifyVector(train_images[i, :], w_bow, 0.5)
        for i in range(train_images.shape[0])])

    # Calculate classification Accuracy in test data
    H_bowTe = ([
        classifyVector(test_images[i, :], w_bow, 0.5)
        for i in range(test_images.shape[0])])

    accuracy = (str(
        float(np.sum(H_bowTe == test_labels)) / test_labels.shape[0]))

    Y = np.append(Y, accuracy)
    print step_size, "BoW Test Accuracy : ", accuracy

    # increase step size and counter
    step_size = step_size * 2  # Powers of two

# Plot the Test Accuracy (y) as a function [y = f(x)] of  the vocabulary
# size (x).
plt.plot(X, Y)
plt.title('Test Accuracy')
plt.xlabel('Vocabulary Size')
plt.show()
