"""Logistic Regression classification with BoW features."""
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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

    print "Number of iterations:", iteration
    return theta


def classifyVector(X, theta):
    """Return predicted label of X.

    Evaluate the Logistic Regression model h(x) with theta parameters,
    and returns the predicted label of x.
    """
    prob = sigmoid(sum(np.dot(X, theta)))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


# Import the train/test data (raw image pixels).
with gzip.open('../raw_pixels_dataset_5980.pklz', 'rb') as f:
    (train_labels, train_images, test_labels, test_images) = pickle.load(f)

print("train_images shape" + str(train_images.shape))
print("test_images shape" + str(test_images.shape))

# For each example we compute the histogram of grey intensity values

old_train_images = np.zeros([train_images.shape[0], 8])
for i in range(train_images.shape[0]):
    # Count how many values are in a range
    old_train_images[i, :] = np.histogram(train_images[i, :], 8)[0]
    # Histogram normalization
    old_train_images[i, :] /= np.sum(old_train_images[i, :])

old_test_images = np.zeros([test_images.shape[0], 8])

for i in range(test_images.shape[0]):
    old_test_images[i, :] = np.histogram(test_images[i, :], 8)[0]
    # Histogram normalization
    old_test_images[i, :] /= np.sum(old_test_images[i, :])

print old_train_images.shape, old_test_images.shape

# print train_labels[1],train_labels[5981]

# Logistic Regression gradient descent optimization
w1 = GradientDescent(old_train_images, train_labels)

H = ([classifyVector(old_train_images[i, :], w1)
     for i in range(old_train_images.shape[0])])
print "Train Accuracy :", (
    str(float(np.sum(H == train_labels)) / train_labels.shape[0]))

# Calculate classification Accuracy in test data
H = ([classifyVector(old_test_images[i, :], w1)
     for i in range(old_test_images.shape[0])])
print "Test Accuracy : ", (
    str(float(np.sum(H == test_labels)) / test_labels.shape[0]))


# confusion matrix on test data
cm = confusion_matrix(test_labels, H)
print 'Confusion matrix:'
print (cm)

# Precision
precision = float(cm[1, 1]) / float((cm[1, 1] + cm[0, 1]))
print "Precision:", precision

# Recall
recall = float(cm[1, 1]) / float((cm[1, 0] + cm[1, 1]))
print "Recall:", recall

# Plot confusion matrix
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
