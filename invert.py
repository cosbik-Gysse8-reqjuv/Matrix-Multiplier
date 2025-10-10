import numpy as np

def invert_matrix(matrix):
    """
    Returns the inverse of a given square matrix (real or complex).
    
    Parameters:
        matrix (list of lists or np.ndarray): The input matrix.
        
    Returns:
        np.ndarray: The inverse of the matrix.
    """
    # Convert to numpy array (handles real or complex)
    A = np.array(matrix, dtype=complex)
    
    # Check if square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square to invert.")
    
    # Check if invertible (det ≠ 0)
    det = np.linalg.det(A)
    if det == 0:
        raise np.linalg.LinAlgError("Matrix is singular and cannot be inverted.")
    
    # Compute the inverse
    A_inv = np.linalg.inv(A)
    
    return A_inv


# Example usage
if __name__ == "__main__":
    M = [
        [3-2j, 3+2j, 0],
        [8-1j, 8+1j, 0],
        [13, 13, 1]
    ]
    
    inv_M = invert_matrix(M)
    print("Matrix inverse:\n", inv_M)
