# Import NumPy
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
# %matplotlib inline

# plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', c='b')
# positive samples
# plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='x', c='r')
# negative samples
# plt.show()

# Append the x_0 column (for the bias term theta_0)
x = np.ones(shape=(X.shape[0], 1))
x = np.append(x, X, axis=1)

# Logistic Regression gradient descent optimization
w = GradientDescent(x, y)
# Evaluate the classifier accuracy in the training data
H = [classifyVector(x[i, :], w) for i in range(x.shape[0])]
print "Training Accuracy : "+str(float(np.sum(H == y)) / y.shape[0])

# Plot data
# plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', c='b')
# positive samples
# plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='x', c='r')
# negative samples


# Plot Decision Boundary
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i, j] = sigmoid(np.dot(np.array([1, u[i], v[j]]), w))

z = z.T

# cs = plt.contour(u, v, z, levels=[0.5])
# plt.clabel(cs, inline=1, fontsize=10)


def map_feature(x1, x2):
    """
    Maps 2D features to quadratic features.
    Returns a new feature vector with more features, comprising of
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc...
    """
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6  # Value to change the complexity default = 6, test = 2
    out = np.ones(shape=(x1[:, 0].size, 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)

    return out


features = map_feature(X[:, 0], X[:, 1])
features.shape

# Logistic Regression gradient descent optimization
w = GradientDescent(features, y)

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
cs = plt.contour(u, v, z, levels=[0.5])  # levels default = 0.5
plt.clabel(cs, inline=1, fontsize=10)

plt.show()
