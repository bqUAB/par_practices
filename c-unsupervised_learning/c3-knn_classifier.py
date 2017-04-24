"""K-NN Classifier."""
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

with gzip.open('./BoW-64-8.pklz', 'rb') as f:
    (train_labels, train_img, test_labels, test_img) = pickle.load(f)

print("Train labels shape " + str(train_labels.shape))
print("Train images shape " + str(train_img.shape))
print("Test labels shape " + str(test_labels.shape))
print("Test images shape " + str(test_img.shape))

n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(train_img, train_labels).predict(test_img)

print("y_ shape " + str(y_.shape))

print "BoW Test Accuracy : ", (str(float(
    np.sum(y_ == test_labels)) / test_labels.shape[0]))

# confusion matrix on test data
cm_bow = confusion_matrix(test_labels, y_)
print 'BoW Confusion matrix:'
print (cm_bow)

# Plot BoW confusion matrix
plt.matshow(cm_bow)
plt.title('BoW Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# BoW Precision
bow_precision = float(cm_bow[1, 1]) / float((cm_bow[1, 1] + cm_bow[0, 1]))
print "BoW Precision:", bow_precision

# BoW Recall
bow_recall = float(cm_bow[1, 1]) / float((cm_bow[1, 0] + cm_bow[1, 1]))
print "BoW Recall:", bow_recall
