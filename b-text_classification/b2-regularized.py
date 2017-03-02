import matplotlib.pyplot as plt
import numpy as np
import pickle


# Function (modules) declarations
def sigmoid(X):
    """
    Computes the Sigmoid function of the input argument X.
    """
    return 1.0/(1+np.exp(-X))


def map_feature(x1, x2):
    """
    Maps 2D features to quadratic features.
    Returns a new feature vector with more features, comprising of
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc...
    """
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(shape=(x1[:, 0].size, 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)

    return out


def GradientDescent(x, y, max_iterations=2500, alpha=0.1, accuracy=0.83):
    m, n = x.shape  # number of samples, number of features
    # y must be a column vector
    y0 = y
    y = y.reshape(m, 1)
    # initialize the parameters
    theta = np.ones(shape=(n, 1))
    # Repeat until convergence (or max_iterations)
    i = 0  # iteration
    t_accuracy = 0  # Training Accuracy
    while t_accuracy <= accuracy and i < max_iterations:
        i += 1
        h = sigmoid(np.dot(x, theta))
        error = (h-y)
        gradient = np.dot(x.T, error) / m
        theta = theta - alpha*gradient
        H = [classifyVector(x[i2, :], theta) for i2 in range(x.shape[0])]
        t_accuracy = float(np.sum(H == y0)) / y0.shape[0]

    print "Iterations done:", i
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


# Load the synthetic data set
with open('./PR1data1.pkl', 'rb') as f:
    (X, y) = pickle.load(f)
print X.shape, y.shape

# Append the x_0 column (for the bias term theta_0)
x = np.ones(shape=(X.shape[0], 1))
x = np.append(x, X, axis=1)  # append as columns -> axis=1

features = map_feature(X[:, 0], X[:, 1])

# Logistic Regression gradient descent optimization
w = GradientDescent(features, y, 1000, 0.2, 0.8)

H = [classifyVector(features[i, :], w) for i in range(features.shape[0])]
print "Training Accuracy : "+str(float(np.sum(H == y)) / y.shape[0])

# Plot data
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', c='b')  # positive samples
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='x', c='r')  # negative samples

# Plot Boundary
u = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 50)
v = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 50)
z = np.zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i, j] = sigmoid(map_feature(np.array(u[i]), np.array(v[j])).dot(w))

z = z.T
cs = plt.contour(u, v, z, levels=[0.25, 0.5, 0.75])
plt.clabel(cs, inline=1, fontsize=10)
plt.show()
