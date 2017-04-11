import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt

# Import the train/test data (raw image pixels).
with gzip.open('../raw_pixels_dataset_5980.pklz', 'rb') as f:
    (train_labels, train_images, test_labels, test_images) = pickle.load(f)

print("train_images shape" + str(train_images.shape))
print("test_images shape" + str(test_images.shape))

# Show a few samples of the positive and negative classes.
num_text = sum(train_labels == 0)
fig = plt.figure()
for i in range(1, 6):
    ax = fig.add_subplot(2, 5, i)
    ax.imshow(
        np.reshape(
            train_images[
                np.random.randint(0, num_text), :],
            [32, 32]),
        cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = fig.add_subplot(2, 5, i+5)
    ax.imshow(
        np.reshape(
            train_images[
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
# PreProcessing (I) Contrast Normalization of Patches

mu = patches.mean(axis=1)  # mean values

# standard deviation (plus a small value)
# sigma = patches.std(axis=1) + (np.ptp(patches, axis=1) / 20.0)
sigma = patches.std(axis=1) + 0.0001

# subtract the mean and divide by the standard deviation
patches = (patches - mu.reshape([-1, 1])) / (sigma.reshape([-1, 1]))

# Set NaN values (if exist) to 0
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
plt.show()
