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


# ----------------------------------------------
# ------------------- BoW  ---------------------
# ----------------------------------------------
# Import the train/test data (raw image pixels).
with gzip.open('./BoW_train_features.pklz', 'rb') as f:
    (prepro_train_labels, prepro_train_images) = pickle.load(f)

with gzip.open('./BoW_test_features.pklz', 'rb') as f:
    (prepro_test_labels, prepro_test_images) = pickle.load(f)

print("Preprocessed train images shape = " + str(prepro_train_images.shape))
print("Preprocessed test images shape = " + str(prepro_test_images.shape))

# Logistic Regression gradient descent optimization
w_bow = GradientDescent(prepro_train_images, prepro_train_labels)

H_bowTr = ([
    classifyVector(prepro_train_images[i, :], w_bow, 0.5)
    for i in range(prepro_train_images.shape[0])])

# Calculate classification Accuracy in test data
H_bowTe = ([
    classifyVector(prepro_test_images[i, :], w_bow, 0.5)
    for i in range(prepro_test_images.shape[0])])

# confusion matrix on test data
cm_bow = confusion_matrix(prepro_test_labels, H_bowTe)
print 'BoW Confusion matrix:'
print (cm_bow)

# ----------------------------------------------
# ------------ Grey Level Histograms -----------
# ----------------------------------------------
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

print "old_train_images shape" + str(old_train_images.shape)
print "old_test_images shape" + str(old_test_images.shape)

# print train_labels[1],train_labels[5981]

# Logistic Regression gradient descent optimization
w1 = GradientDescent(old_train_images, train_labels)

H_train = ([
    classifyVector(old_train_images[i, :], w1, 0.5)
    for i in range(old_train_images.shape[0])])

# Calculate classification Accuracy in test data
H_test = ([
    classifyVector(old_test_images[i, :], w1, 0.5)
    for i in range(old_test_images.shape[0])])

# confusion matrix on test data
cm = confusion_matrix(test_labels, H_test)
print 'Previous Confusion matrix:'
print (cm)

# ----------------------------------------------
# ------------ Print Accuracy Values -----------
# ----------------------------------------------
print "BoW Train Accuracy:", (str(float(
    np.sum(H_bowTr == prepro_train_labels)) / prepro_train_labels.shape[0]))
print "BoW Test Accuracy : ", (str(float(
    np.sum(H_bowTe == prepro_test_labels)) / prepro_test_labels.shape[0]))

print "Previous Train Accuracy :", (
    str(float(np.sum(H_train == train_labels)) / train_labels.shape[0]))
print "Previous Test Accuracy : ", (
    str(float(np.sum(H_test == test_labels)) / test_labels.shape[0]))

# ----------------------------------------------
# ------- Plotting Confusion Matrices ----------
# ----------------------------------------------
# Plot BoW confusion matrix
plt.matshow(cm_bow)
plt.title('BoW Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Plot confusion matrix
plt.matshow(cm)
plt.title('Previous Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# ----------------------------------------------
# ----------- Precision and Recall -------------
# ----------------------------------------------
# BoW Precision
bow_precision = float(cm_bow[1, 1]) / float((cm_bow[1, 1] + cm_bow[0, 1]))
print "BoW Precision:", bow_precision

# BoW Recall
bow_recall = float(cm_bow[1, 1]) / float((cm_bow[1, 0] + cm_bow[1, 1]))
print "BoW Recall:", bow_recall

# Precision
precision = float(cm[1, 1]) / float((cm[1, 1] + cm[0, 1]))
print "Previous Precision:", precision

# Recall
recall = float(cm[1, 1]) / float((cm[1, 0] + cm[1, 1]))
print "Previous Recall:", recall

# ----------------------------------------------
# ----------- BoW (No preprocessing) -----------
# ----------------------------------------------
# Import the train/test data (raw image pixels).
with gzip.open('./BoW-no_preprocess.pklz', 'rb') as f:
    (train_labels, train_images, test_labels, test_images) = pickle.load(f)

print("Unprocessed train images shape = " + str(train_images.shape))
print("Unprocessed test images shape = " + str(test_images.shape))

# Logistic Regression gradient descent optimization
w_bow = GradientDescent(train_images, train_labels)

H_bowTr = ([
    classifyVector(train_images[i, :], w_bow, 0.5)
    for i in range(train_images.shape[0])])

# Calculate classification Accuracy in test data
H_bowTe = ([
    classifyVector(test_images[i, :], w_bow, 0.5)
    for i in range(test_images.shape[0])])

# confusion matrix on test data
cm_bow = confusion_matrix(test_labels, H_bowTe)
print 'Unprocessed BoW Confusion matrix:'
print (cm_bow)

print "Unprocessed BoW Train Accuracy:", (str(float(
    np.sum(H_bowTr == train_labels)) / train_labels.shape[0]))
print "Unprocessed BoW Test Accuracy : ", (str(float(
    np.sum(H_bowTe == test_labels)) / test_labels.shape[0]))

# Plot BoW confusion matrix
plt.matshow(cm_bow)
plt.title('Unprocessed BoW Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Unprocessed BoW Precision
bow_precision = float(cm_bow[1, 1]) / float((cm_bow[1, 1] + cm_bow[0, 1]))
print "Unprocessed BoW Precision:", bow_precision

# Unprocessed BoW Recall
bow_recall = float(cm_bow[1, 1]) / float((cm_bow[1, 0] + cm_bow[1, 1]))
print "Unprocessed BoW Recall:", bow_recall
