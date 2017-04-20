"""BoW (No preprocessing)."""
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


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

# ----------------------------------------------------
# K-means clustering to learn K visual_words from data
NUM_VISUAL_WORDS = 64

km = KMeans(n_clusters=NUM_VISUAL_WORDS, max_iter=50, n_init=1, verbose=False)
km.fit(train_patches)

visual_words = km.cluster_centers_
print("visual_words shape " + str(visual_words.shape))

# Visualize the learned vocabulary of visual words
fig = plt.figure()
num_col = int(np.ceil(NUM_VISUAL_WORDS / 4.0))  # Round up to higher
for i in xrange(NUM_VISUAL_WORDS):
    ax = fig.add_subplot(4, num_col, i+1)
    visual_word_ = visual_words[i, :]
    visual_word_ = visual_word_.reshape(PATCH_SIZE, PATCH_SIZE)
    ax.imshow(visual_word_, interpolation='none', cmap=cm.Greys_r)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

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

            # # Set NaN values to 0
            # w = np.isnan(train_patches)
            # train_patches[w] = 0

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

            # # Set NaN values to 0
            # w = np.isnan(train_patches)
            # train_patches[w] = 0

            # BoW
            nn = neig.predict(test_patches)
            test_features[i, nn] = test_features[i, nn] + 1

    # Histogram Normalization
    test_features[i, :] = test_features[i, :] / max(test_features[i, :])

# Save your test features
with gzip.open('./BoW-no_preprocess.pklz', 'wb') as f:
    pickle.dump(
        (train_labels, train_features, test_labels, test_features),
        f, pickle.HIGHEST_PROTOCOL)
