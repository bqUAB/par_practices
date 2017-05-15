"""SVM multiclass classification with different kernels."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, grid_search
from sklearn.metrics import confusion_matrix


def plot_classifier_boundary(clf, X, y):
    """Draw the decision function of a classifier in a 2D space."""
    # create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = (np.meshgrid(np.arange(x_min, x_max, h),
              np.arange(y_min, y_max, h)))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()


iris = datasets.load_iris()
train_features = iris.data[:, :2]  # Here we only use the first two features.
train_labels = iris.target

# Grid search for SVM parameter C
# Checking only three possible values for the C parameter
parameters = {'C': [0.1, 1, 10]}
classifier = svm.SVC(kernel='linear')  # Linear kernel
grid = grid_search.GridSearchCV(classifier, parameters)
grid.fit(train_features, train_labels)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Train classifier using the best parameter
classifier = svm.SVC(kernel='linear', C=0.1)
classifier = classifier.fit(train_features, train_labels)

# Run classifier
result = classifier.predict(train_features)
print "Train Accuracy : "+str(classifier.score(train_features, train_labels))

# confusion matrix on test data
cm = confusion_matrix(train_labels, result)

# Precision
precision = float(cm[1, 1]) / float((cm[1, 1] + cm[0, 1]))
print "Precision:", precision

# Recall
recall = float(cm[1, 1]) / float((cm[1, 0] + cm[1, 1]))
print "Recall:", recall

plot_classifier_boundary(classifier, train_features, train_labels)
