import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
print("Shape of X:", np.shape(X))
T = np.linspace(0, 5, 500)[:, np.newaxis]
print("Shape of T:", np.shape(T))
y = np.sin(X).ravel()
print("Shape of y:", np.shape(y))

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))
n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, c='k', label='data')
    plt.plot(T, y_, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))
print("y_ shape " + str(y_.shape))
plt.show()
