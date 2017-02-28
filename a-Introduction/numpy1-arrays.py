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


print(np.linspace(30, 39, 10))
print(np.linspace(40, 50))

x = np.arange(9).reshape((3, 3))
print "x ="
print(x)
print "np.diag(x) returns:", np.diag(x)
print "np.diag(x, k=1) returns:", np.diag(x, k=1)
print "np.diag(x, k=-1) returns:", np.diag(x, k=-1)
print "np.diag([1, 2, 3]) returns:"
print(np.diag([1, 2, 3]))

A = np.array([1, 2, 3])
B = np.array([1.0, 2.0, 3.0])
A.dtype, B.dtype

C = np.array([4, 5, 6], dtype=float)
print(C)

A = np.random.randint(10, size=(4, 4))
B = np.zeros([2, 2]).reshape([4, 1])
# Append column
A = np.append(A, B, axis=1)
print(A)

C = np.zeros([5, 1]).reshape([1, 5])
# Append row
print(np.append(A, C, axis=0))

# Or
A = np.random.randint(10, size=(4, 4))
C = np.zeros([4, 1])
# Append column
print(np.append(A, C, axis=1))
