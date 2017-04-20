"""Feature Learning.

Feature Learning with K-means clustering. Nearest Neighbours, and Bag of Words.
"""
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

# Import the train/test data (raw image pixels).
with gzip.open('../raw_pixels_dataset_5980.pklz', 'rb') as f:
    (train_labels, train_images, test_labels, test_images) = pickle.load(f)

print("train_images shape" + str(train_images.shape))
print("test_images shape" + str(test_images.shape))

# Show a few samples of the positive and negative classes.
num_text = sum(train_labels == 0)
# The dataset is symmetric which means that half of data is positive.
# That means that the following code will also work:
# ** np.random.randint(train_labels.shape[0]/2, train_labels.shape[0]), :],

fig = plt.figure()
for i in range(1, 6):
    ax = fig.add_subplot(2, 5, i)
    ax.imshow(
        np.reshape(
            train_images[
                # np.random.randint(0, train_labels.shape[0]/2), :],
                np.random.randint(0, num_text), :],
            [32, 32]),
        cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = fig.add_subplot(2, 5, i+5)
    ax.imshow(
        np.reshape(
            train_images[
                # can be replaced with **
                np.random.randint(num_text, train_labels.shape[0]), :],
            [32, 32]),
        cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.show()

# -------------------------------------------------
# Extract patches (of 8x8 pixels) from train images

# reshape images to 32x32 and scale values to (0, 1)
images = train_images.reshape((-1, 32, 32)).astype('float32') / 255
print("Images shape " + str(images.shape))

# Collect image patches with sliding window (8x8) in each train image sample
PATCH_SIZE = 8
STEP_SIZE = 8

patches = np.zeros((0, PATCH_SIZE, PATCH_SIZE))

for x in range(0, 32 - PATCH_SIZE + 1, STEP_SIZE):
    for y in range(0, 32 - PATCH_SIZE + 1, STEP_SIZE):
        patches = np.concatenate(
                    (patches, images[:, x: x + PATCH_SIZE, y: y + PATCH_SIZE]),
                    axis=0)

patches = patches.reshape((patches.shape[0], -1))
print("Patches shape " + str(patches.shape))

# Visualize a few patches
fig = plt.figure()
for i in range(1, 6):
    ax = fig.add_subplot(1, 5, i)
    ax.imshow(
        np.reshape(patches[i], [PATCH_SIZE, PATCH_SIZE]),
        cmap=plt.cm.gray, vmin=0, vmax=1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.show()

# ---------------------------------------------------
# Preprocessing (I) Contrast Normalization of Patches

mu = patches.mean(axis=1)  # mean values

# standard deviation (plus a small value)
sigma = patches.std(axis=1) + 0.0001

# subtract the mean and divide by the standard deviation
patches = (patches - mu.reshape([-1, 1])) / (sigma.reshape([-1, 1]))

# Set NaN (not a number) values (if exist) to 0
w = np.isnan(patches)
patches[w] = 0

# Visualize a few Constrast Normalized patches
fig = plt.figure()
for i in range(1, 6):
    ax = fig.add_subplot(1, 5, i)
    ax.imshow(
        np.reshape(patches[i], [PATCH_SIZE, PATCH_SIZE]),
        cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.show()

# ------------------------------------------------------
# Preprocessing (II) ZCA Whitening of normalized patches

eig_values, eig_vec = np.linalg.eig(np.cov(patches.T))
epsilon = 0.01
pca = eig_vec.dot(np.diag((eig_values + epsilon) ** -0.5).dot(eig_vec.T))

M = patches.mean(axis=0)
patches = patches - M  # substract average value
patches = np.dot(patches, pca)  # perform pca whitening

# Visualize a few preprocessed patches
fig = plt.figure()
for i in range(1, 6):
    ax = fig.add_subplot(1, 5, i)
    ax.imshow(
        np.reshape(patches[i], [PATCH_SIZE, PATCH_SIZE]),
        cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.show()

# ----------------------------------------------------
# K-means clustering to learn K visual_words from data
NUM_VISUAL_WORDS = 64

km = KMeans(n_clusters=NUM_VISUAL_WORDS, max_iter=50, n_init=1, verbose=False)
km.fit(patches)

visual_words = km.cluster_centers_
print("visual_words shape " + str(visual_words.shape))

# Visualize the learned vocabulary of visual words
fig = plt.figure()
num_col = int(np.ceil(float(NUM_VISUAL_WORDS) / 4))
for i in xrange(NUM_VISUAL_WORDS):
    ax = fig.add_subplot(4, num_col, i+1)
    visual_word_ = visual_words[i, :]
    visual_word_ = visual_word_.reshape(PATCH_SIZE, PATCH_SIZE)
    ax.imshow(visual_word_, interpolation='none', cmap=cm.Greys_r)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.show()

# -------------------------------------------------------------
# Learn a KNN classifier, each visual word represents one class
# In this case KNN is used simply as a way to search the nearest neighbor
# visual word in our vocabulary

neig = KNeighborsClassifier(n_neighbors=1)
neig.fit(visual_words, range(0, NUM_VISUAL_WORDS))

# Extract features from train images
train_features = np.zeros((images.shape[0], visual_words.shape[0]))

for i in range(0, images.shape[0]):  # for each image
    # Do sliding window (8x8) in each image to extract patches
    # then normalize, whiten and build the Bag of Words histogram
    for x in range(0, 32 - PATCH_SIZE + 1, STEP_SIZE):
        for y in range(0, 32 - PATCH_SIZE + 1, STEP_SIZE):
            patch = images[i, x: x + PATCH_SIZE, y: y + PATCH_SIZE]
            patch = patch.reshape(1, -1)

            # Preprocessing (I): normalize
            mu = patch.mean(axis=1)  # mean values
            sigma = (
                patch.std(axis=1) +
                max(np.ptp(patch, axis=1) / 20.0, 0.0001))  # std + small value
            patch = (patch-(mu[np.newaxis, :]).T)/(sigma[np.newaxis, :]).T

            # Set NaN values to 0
            w = np.isnan(patch)
            patch[w] = 0

            # Preprocessing (II): ZCA whitening
            patch = patch - M  # substract average value
            patch = np.dot(patch, pca)  # perform pca whitening

            # BoW
            nn = neig.predict(patch)
            train_features[i, nn] = train_features[i, nn] + 1

    # Histogram Normalization
    train_features[i, :] = train_features[i, :] / max(train_features[i, :])

# If you want to save your training features
with gzip.open('./BoW_train_features.pklz', 'wb') as f:
    pickle.dump((train_labels, train_features), f, pickle.HIGHEST_PROTOCOL)

# Extract features from test images
images = test_images.reshape((-1, 32, 32)).astype('float32') / 255
test_features = np.zeros((images.shape[0], visual_words.shape[0]))

for i in range(0, images.shape[0]):  # for each image
    # Do sliding window (8x8) in each image to extract patches
    # then normalize, whiten and build the Bag of Words histogram
    for x in range(0, 32 - PATCH_SIZE + 1, STEP_SIZE):
        for y in range(0, 32 - PATCH_SIZE + 1, STEP_SIZE):
            patch = images[i, x: x + PATCH_SIZE, y: y + PATCH_SIZE]
        patch = patch.reshape(1, -1)

        # PreProcessing (I): Normalize
        mu = patch.mean(axis=1)  # mean values
        sigma = (
            patch.std(axis=1) +
            max(np.ptp(patch, axis=1) / 20.0, 0.0001))  # std + small value
        patch = (patch - (mu[np.newaxis, :]).T) / (sigma[np.newaxis, :]).T

        # Set NaN values to 0
        w = np.isnan(patch)
        patch[w] = 0

        # PreProcessing (II): ZCA whitening
        patch = patch - M  # subtract average value
        patch = np.dot(patch, pca)  # perform pca whitening

        # BoW
        nn = neig.predict(patch)
        test_features[i, nn] = test_features[i, nn] + 1

    # Histogram normalization
    test_features[i, :] = test_features[i, :] / max(test_features[i, :])

# If you want to save your test features
with gzip.open('./BoW_test_features.pklz', 'wb') as f:
    pickle.dump((test_labels, test_features), f, pickle.HIGHEST_PROTOCOL)
