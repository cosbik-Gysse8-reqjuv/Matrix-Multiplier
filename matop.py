import numpy as np

class MatrixOps:
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


# Example usage
if __name__ == "__main__":
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[0, 5], [6, 7]])

    print("A:\n", A)
    print("B:\n", B)

    print("\nA + B:\n", MatrixOps.add(A, B))
    print("\nA * B (matrix multiply):\n", MatrixOps.multiply(A, B))
    print("\nA elementwise B:\n", MatrixOps.elementwise_multiply(A, B))
    print("\nTranspose of A:\n", MatrixOps.transpose(A))
    print("\nDeterminant of A:\n", MatrixOps.determinant(A))
    print("\nInverse of A:\n", MatrixOps.inverse(A))
    print("\nKronecker product of A and B:\n", MatrixOps.kronecker(A, B))