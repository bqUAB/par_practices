"""Effect of vocabulary size in Bow."""
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
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


# Import the train/test data (raw image pixels).
with gzip.open('../raw_pixels_dataset_5980.pklz', 'rb') as f:
    (train_labels, train_img, test_labels, test_img) = pickle.load(f)

print("train_img before reshape" + str(train_img.shape))
print("test_img before reshape" + str(test_img.shape))

# -----------------------------------------------------------------
# Extract patches (of 8x8 pixels) from train images and test images

# reshape images to 32x32 and scale values to (0, 1)
train_img = train_img.reshape((-1, 32, 32)).astype('float32') / 255
test_img = test_img.reshape((-1, 32, 32)).astype('float32') / 255

print("train_img shape " + str(train_img.shape))
print("test_img shape " + str(test_img.shape))

# Collect image patches with sliding window (8x8) in each train image sample
PATCH_SIZE = 8
STEP_SIZE = 8

# Create a container for the patches
# 0 = initialization of container   <- This value is going to be updated to the
#                                      number of images of the training set.
# PATCH_SIZE = 8 = x (width) dimensions of the image
# PATCH_SIZE = 8 = y (height) dimensions of the image
# As a result, in this case, the resolution of the patch is 8 x 8 = 64 px
train_patches = np.zeros((0, PATCH_SIZE, PATCH_SIZE))
test_patches = np.zeros((0, PATCH_SIZE, PATCH_SIZE))

# Sliding window process
for x in range(0, 32 - PATCH_SIZE + 1, STEP_SIZE):
    for y in range(0, 32 - PATCH_SIZE + 1, STEP_SIZE):
        train_patches = (
            np.concatenate(  # increase the container each iteration
                (train_patches,
                 train_img[:, x: x + PATCH_SIZE, y: y + PATCH_SIZE]),
                axis=0))
        test_patches = (
            np.concatenate(  # increase the container each iteration
                (test_patches,
                 test_img[:, x: x + PATCH_SIZE, y: y + PATCH_SIZE]),
                axis=0))

# Store the patch as a vector
train_patches = train_patches.reshape((train_patches.shape[0], -1))
test_patches = test_patches.reshape((test_patches.shape[0], -1))
print("train_patches shape " + str(train_patches.shape))
print("test_patches shape " + str(test_patches.shape))

# ---------------------------------------------------
# Preprocessing (I) Contrast Normalization of Patches

mu = train_patches.mean(axis=1)  # mean values

# standard deviation (plus a small value)
sigma = train_patches.std(axis=1) + 0.0001

# subtract the mean and divide by the standard deviation
train_patches = (
    train_patches - mu.reshape([-1, 1])) / (sigma.reshape([-1, 1]))

# Set NaN (not a number) values (if exist) to 0
w = np.isnan(train_patches)
train_patches[w] = 0

# ------------------------------------------------------
# Preprocessing (II) ZCA Whitening of normalized patches

eig_values, eig_vec = np.linalg.eig(np.cov(train_patches.T))
epsilon = 0.01
pca = eig_vec.dot(np.diag((eig_values + epsilon) ** -0.5).dot(eig_vec.T))

M = train_patches.mean(axis=0)
train_patches = train_patches - M  # substract average value
train_patches = np.dot(train_patches, pca)  # perform pca whitening

# ----------------------------------------------------
# K-means clustering to learn K visual_words from data
NUM_VISUAL_WORDS = 256

km = KMeans(n_clusters=NUM_VISUAL_WORDS, max_iter=50, n_init=1, verbose=False)
km.fit(train_patches)

visual_words = km.cluster_centers_
print("visual_words shape " + str(visual_words.shape))

# -------------------------------------------------------------
# Learn a KNN classifier, each visual word represents one class
# In this case KNN is used simply as a way to search the nearest neighbor
# visual word in our vocabulary

neig = KNeighborsClassifier(n_neighbors=1)
neig.fit(visual_words, range(0, NUM_VISUAL_WORDS))

# Extract features from train and test images
train_features = np.zeros((train_img.shape[0], visual_words.shape[0]))
test_features = np.zeros((test_img.shape[0], visual_words.shape[0]))

for i in range(0, train_img.shape[0]):  # for each train image
    # Do sliding window (8x8) in each image to extract patches and build the
    # Bag of Words histogram
    for x in range(0, 32 - PATCH_SIZE + 1, STEP_SIZE):
        for y in range(0, 32 - PATCH_SIZE + 1, STEP_SIZE):
            train_patches = train_img[i, x: x + PATCH_SIZE, y: y + PATCH_SIZE]
            train_patches = train_patches.reshape(1, -1)

            # Preprocessing (I): normalize
            mu = train_patches.mean(axis=1)  # mean values
            sigma = (  # std + small value
                train_patches.std(axis=1) +
                max(np.ptp(train_patches, axis=1) / 20.0, 0.0001))
            train_patches = (
                train_patches-(mu[np.newaxis, :]).T)/(sigma[np.newaxis, :]).T

            # Set NaN values to 0
            w = np.isnan(train_patches)
            train_patches[w] = 0

            # Preprocessing (II): ZCA whitening
            train_patches = train_patches - M  # substract average value
            train_patches = np.dot(train_patches, pca)  # perform pca whitening

            # BoW
            nn = neig.predict(train_patches)
            train_features[i, nn] = train_features[i, nn] + 1

    # Histogram Normalization
    train_features[i, :] = train_features[i, :] / max(train_features[i, :])

for i in range(0, test_img.shape[0]):  # for each test image
    # Do sliding window (8x8) in each image to extract patches and build the
    # Bag of Words histogram
    for x in range(0, 32 - PATCH_SIZE + 1, STEP_SIZE):
        for y in range(0, 32 - PATCH_SIZE + 1, STEP_SIZE):
            test_patches = test_img[i, x: x + PATCH_SIZE, y: y + PATCH_SIZE]
            test_patches = test_patches.reshape(1, -1)

            # PreProcessing (I): Normalize
            mu = test_patches.mean(axis=1)  # mean values
            sigma = (  # std + small value
                test_patches.std(axis=1) +
                max(np.ptp(test_patches, axis=1) / 20.0, 0.0001))
            test_patches = (
                test_patches-(mu[np.newaxis, :]).T) / (sigma[np.newaxis, :]).T

            # Set NaN values to 0
            w = np.isnan(test_patches)
            test_patches[w] = 0

            # PreProcessing (II): ZCA whitening
            test_patches = test_patches - M  # subtract average value
            test_patches = np.dot(test_patches, pca)  # perform pca whitening

            # BoW
            nn = neig.predict(test_patches)
            test_features[i, nn] = test_features[i, nn] + 1

    # Histogram Normalization
    test_features[i, :] = test_features[i, :] / max(test_features[i, :])

# Save your test features
with gzip.open('./BoW-256-8.pklz', 'wb') as f:
    pickle.dump(
        (train_labels, train_features, test_labels, test_features),
        f, pickle.HIGHEST_PROTOCOL)

# Import the train/test data (raw image pixels).
with gzip.open('./BoW-256-8.pklz', 'rb') as f:
    (train_labels, train_images, test_labels, test_images) = pickle.load(f)

print("train_images shape = " + str(train_images.shape))
print("test_images shape = " + str(test_images.shape))

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
print 'BoW Confusion matrix:'
print (cm_bow)

print "BoW Train Accuracy:", (str(float(
    np.sum(H_bowTr == train_labels)) / train_labels.shape[0]))
print "BoW Test Accuracy : ", (str(float(
    np.sum(H_bowTe == test_labels)) / test_labels.shape[0]))

# Plot BoW confusion matrix
plt.matshow(cm_bow)
plt.title('BoW Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Unprocessed BoW Precision
bow_precision = float(cm_bow[1, 1]) / float((cm_bow[1, 1] + cm_bow[0, 1]))
print "BoW Precision:", bow_precision

# Unprocessed BoW Recall
bow_recall = float(cm_bow[1, 1]) / float((cm_bow[1, 0] + cm_bow[1, 1]))
print "BoW Recall:", bow_recall
