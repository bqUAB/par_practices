import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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


# load the data
with gzip.open('./raw_pixels_dataset_5980.pklz', 'rb') as f:
    (train_labels, train_features, test_labels, test_features) = pickle.load(f)

print train_features.shape
print test_features.shape

# For each example we compute the histogram of grey intensity values

new_train_features = np.zeros([train_features.shape[0], 8])
for i in range(train_features.shape[0]):
    # Count how many values are in a range
    new_train_features[i, :] = np.histogram(train_features[i, :], 8)[0]
    # Histogram normalization
    new_train_features[i, :] /= np.sum(new_train_features[i, :])

new_test_features = np.zeros([test_features.shape[0], 8])
for i in range(test_features.shape[0]):
    new_test_features[i, :] = np.histogram(test_features[i, :], 8)[0]
    # Histogram normalization
    new_test_features[i, :] /= np.sum(new_test_features[i, :])

print new_train_features.shape, new_test_features.shape

# Visualize the histograms of positive/negative samples

plt.subplot(2, 2, 1)
plt.imshow(np.reshape(train_features[1, :], [32, 32]), cmap=plt.cm.gray)

plt.subplot(2, 2, 2)
plt.imshow(np.reshape(train_features[5981, :], [32, 32]), cmap=plt.cm.gray)

bins = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 bits making each pixel grayscale value

plt.subplot(2, 2, 3)
plt.bar(bins, new_train_features[1, :], align='center')

plt.subplot(2, 2, 4)
plt.bar(bins, new_train_features[5981, :], align='center')

plt.show()

# print train_labels[1],train_labels[5981]

# Logistic Regression gradient descent optimization
w1 = GradientDescent(new_train_features, train_labels)

H = ([classifyVector(new_train_features[i, :], w1)
     for i in range(new_train_features.shape[0])])
print "Train Accuracy :", (
    str(float(np.sum(H == train_labels)) / train_labels.shape[0]))

# Calculate classification Accuracy in test data
H = ([classifyVector(new_test_features[i, :], w1)
     for i in range(new_test_features.shape[0])])
print "Test Accuracy : ", (
    str(float(np.sum(H == test_labels)) / test_labels.shape[0]))


# confusion matrix on test data
cm = confusion_matrix(test_labels, H)
print 'Confusion matrix:'
print (cm)

# Plot confusion matrix
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
