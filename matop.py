import numpy as np

class MatOp:
    @staticmethod
    def add(A, B):
        """Add two matrices"""
        return np.add(A, B)

    @staticmethod
    def subtract(A, B):
        """Subtract two matrices"""
        return np.subtract(A, B)

    @staticmethod
    def multiply(A, B):
        """Matrix multiplication"""
        return np.matmul(A, B)

    @staticmethod
    def elementwise_multiply(A, B):
        """Element-wise multiplication"""
        return np.multiply(A, B)

    @staticmethod
    def transpose(A):
        """Transpose of a matrix"""
        return np.transpose(A)

    @staticmethod
    def determinant(A):
        """Determinant of a matrix"""
        return np.linalg.det(A)

    @staticmethod
    def inverse(A):
        """Inverse of a matrix"""
        return np.linalg.inv(A)

    @staticmethod
    def kronecker(A, B):
        """Kronecker product of two matrices"""
        return np.kron(A, B)


A = np.array([[1.9, -0.9], [1, 0]])
B = np.array([[1], [0]])
I = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
AA = MatOp.kronecker(A, A)
BB = MatOp.kronecker(B, B)
invI = MatOp.inverse(MatOp.subtract(I, AA))
Vecnull = MatOp.multiply(invI, BB)
NullM = Vecnull.reshape(2, 2)
OneM = MatOp.multiply(A, NullM)
TwoM = MatOp.multiply(A, OneM)
ThreeM = MatOp.multiply(A, TwoM)
Null = NullM[0, 0]
One = OneM[0, 0]
Two = TwoM[0, 0]
Three = ThreeM[0, 0]
print("Variance:")
print(Null)
print("Autocovariances:")
print(One)
print(Two)
print(Three)
print("Autocorrelations:")
print(One / Null)
print(Two / Null)
print(Three / Null)
eigenvalues = np.linalg.eigvals(A)
print("Eigenvalues:", eigenvalues)