# Import Numpy
import numpy as np


# Creating a vector (1D array)
A = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
print "A =", A

# Create a two dimensional array (2x2 matrix)
B = np.array([[1, 2], [3, 4]])
print("B =")
print(B)

# Create a two dimensional array that is not a matrix
C = np.array([[1, 2, 3], [4, 5], [6, 7, 8]])
print("C =")
print(C)


m = "Size of the first dimension (columns) for A ="
print m, len(A)
m = "Size of the first dimension (columns) for B ="
print m, len(B)
m = "Size of the first dimension (columns) for C ="
print m, len(C)


m = "Shape of A ="
print m, A.shape
m = "Shape of B ="
print m, B.shape
m = "Shape of C ="
print m, C.shape


m = "Dimensions of A ="
print m, A.ndim
m = "Dimensions of B ="
print m, B.ndim
m = "Dimensions of C ="
print m, C.ndim
