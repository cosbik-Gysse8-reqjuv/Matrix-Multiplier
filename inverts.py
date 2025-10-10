import sympy as sp

def invert_matrix_exact(matrix):
    """
    Returns the exact inverse of a given square matrix using fractions.
    Works for real or complex entries.
    """
    # Convert to a SymPy Matrix
    A = sp.Matrix(matrix)
    
    # Check if the matrix is square
    if A.rows != A.cols:
        raise ValueError("Matrix must be square to invert.")
    
    # Compute determinant to check invertibility
    if A.det() == 0:
        raise sp.NonInvertibleMatrixError("Matrix is singular and cannot be inverted.")
    
    # Compute exact inverse
    A_inv = A.inv()
    
    # Simplify fractions
    A_inv_simplified = sp.simplify(A_inv)
    
    return A_inv_simplified


if __name__ == "__main__":
    # Define matrices (they can include fractions or complex numbers)
    A = [
        [3-2*sp.I, 3+2*sp.I, 0],
        [8-1*sp.I, 8+1*sp.I, 0],
        [13, 13, 1]
    ]
    B = [
        [0.64 + 0.48*sp.I, 0, 0],
        [0, 0.64 - 0.48*sp.I, 0],
        [0, 0, 1]
    ]
    
    # Convert lists to SymPy Matrix
    A_mat = sp.Matrix(A)
    B_mat = sp.Matrix(B)
    
    # Invert A
    inv_A = invert_matrix_exact(A)
    
    # Multiply the matrices
    D = A_mat * B_mat
    E = D * inv_A
    E_simplified = sp.simplify(E)
    
    # Display results
    print("Matrix A:")
    sp.pprint(A_mat)
    print("\nMatrix B:")
    sp.pprint(B_mat)
    print("\nInverse of A:")
    sp.pprint(inv_A)
    print("\nA * B:")
    sp.pprint(D)
    print("\n(A * B) * inv(A) simplified:")
    sp.pprint(E_simplified)
