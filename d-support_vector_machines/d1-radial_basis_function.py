"""Radial Basis Function."""
import pickle
import gzip
from sklearn import svm, grid_search
from sklearn.metrics import confusion_matrix

# Load the train and test datasets
with gzip.open('../c-unsupervised_learning/BoW-128-4.pklz', 'rb') as f:
    (train_labels, train_features, test_labels, test_features) = pickle.load(f)

# Grid search for SVM parameter C
# Checking only three possible values for the C parameter
parameters = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
classifier = svm.SVC(kernel='rbf')  # Radial Basis Function kernel
grid = grid_search.GridSearchCV(classifier, parameters)
grid.fit(train_features, train_labels)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Train classifier (takes around 3 minutes with this kernel and data)
print "Using the best score"
classifier = svm.SVC(kernel='rbf', C=10.0, gamma=0.1)
classifier = classifier.fit(train_features, train_labels)

print "Train Accuracy : "+str(classifier.score(train_features, train_labels))

# Run classifier
result = classifier.predict(test_features)
print "Test Accuracy : "+str(classifier.score(test_features, test_labels))

# confusion matrix on test data
cm = confusion_matrix(test_labels, result)

# Precision
precision = float(cm[1, 1]) / float((cm[1, 1] + cm[0, 1]))
print "Precision:", precision

# Recall
recall = float(cm[1, 1]) / float((cm[1, 0] + cm[1, 1]))
print "Recall:",
recall
