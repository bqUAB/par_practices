"""Custom Kernel Function with BoW."""
import pickle
import gzip
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix


def chisquare_kernel(x, y):
    """Based on the Chi-Square distribution. Commonly applied to histograms."""
    kernel = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            divisor = (1/2)*(x[i, :] + y[j, :])
            # Check for indeterminations (divisions by 0)
            for k in range(divisor.shape[0]):
                if divisor[k] == 0:
                    # Change 0 for a small number
                    divisor[k] = 0.00001
            kernel[i][j] = 1 - np.sum((pow(x[i, :] - y[j, :], 2))/(divisor))
    return kernel


# Load the train and test datasets
with gzip.open('../c-unsupervised_learning/BoW-128-4.pklz', 'rb') as f:
    (train_labels, train_features, test_labels, test_features) = pickle.load(f)

classifier = svm.SVC(kernel=chisquare_kernel)
classifier = classifier.fit(train_features, train_labels)

print "Train Accuracy : "+str(classifier.score(train_features, train_labels))

# Run classifier
result = classifier.predict(train_features)

# confusion matrix on test data
cm = confusion_matrix(train_labels, result)

# Precision
precision = float(cm[1, 1]) / float((cm[1, 1] + cm[0, 1]))
print "Precision:", precision

# Recall
recall = float(cm[1, 1]) / float((cm[1, 0] + cm[1, 1]))
print "Recall:", recall
