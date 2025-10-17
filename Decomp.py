import numpy as np

def normalized_eig(B):
    """
    Returns V, D, and V_inv such that B = V D V_inv,
    and the first row of V is normalized to 1.
    """
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eig(B)
    
    # Normalize each column of V so that its top element = 1
    for i in range(eigvecs.shape[1]):
        eigvecs[:, i] = eigvecs[:, i] / eigvecs[0, i]
    
    # Diagonal matrix of eigenvalues
    D = np.diag(eigvals)
    
    # Inverse of V
    V_inv = np.linalg.inv(eigvecs)
    
    return eigvecs, D, V_inv

# Example:
B = np.array([[0.84, -0.08],
              [-0.08, 0.96]])

V, D, V_inv = normalized_eig(B)

print("V (normalized eigenvectors):\n", V)
print("\nD (eigenvalues):\n", D)
print("\nCheck VDV^-1 ≈ B:\n", V @ D @ V_inv)
