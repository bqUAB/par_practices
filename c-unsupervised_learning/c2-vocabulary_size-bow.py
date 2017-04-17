"""Effect of vocabulary size in Bow."""
import numpy as np
import matplotlib.pyplot as plt

# Variables initialization
step_size = 2  # Powers of two
counter = 1  # Number of iterations
X = np.zeros(0)
Y = np.zeros(0)


# Create a varying vocabulary size
while step_size <= 128:
    print("We're on time %d with step size %d" % (counter, step_size))

    # Data creation
    X = np.append(X, step_size)
    Y = np.append(Y, counter)  # <- Values to change

    # increase step size and counter
    step_size = step_size * 2
    counter += 1

print("X: ")
print(X)
print("Y: ")
print(Y)

# Plot the Test Accuracy (y) as a function [y = f(x)] of  the vocabulary
# size (x).
plt.plot(X, Y)
plt.title('Test Accuracy')
plt.xlabel('Vocabulary Size')
plt.show()
