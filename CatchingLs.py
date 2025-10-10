import sympy as sp

def find_W_Wprime_normalized(B, X):
    """
    Given matrices B and X, find W and W' such that B = W * X * W',
    where W is lower triangular with 1's on the diagonal.
    """
    B = sp.Matrix(B)
    X = sp.Matrix(X)

    # Check dimensions
    n = B.shape[0]
    if B.shape != X.shape or B.rows != B.cols:
        raise ValueError("B and X must be square matrices of the same size.")

    # Define symbolic entries for the strictly lower part of W
    # (upper part = 0, diagonal = 1)
    w_symbols = sp.symbols(f'w0:{int(n*(n-1)/2)}')
    idx = 0
    W = sp.eye(n)
    for i in range(n):
        for j in range(i):
            W[i, j] = w_symbols[idx]
            idx += 1

    # Compute W' from the equation B = W X W'  ⇒  W' = X⁻¹ * W⁻¹ * B
    if X.det() == 0:
        raise ValueError("Matrix X must be invertible.")
    W_prime = sp.simplify(X.inv() * W.inv() * B)

    return sp.simplify(W), sp.simplify(W_prime)


# Example usage
if __name__ == "__main__":
    # Define symbols (you can replace with numeric values)
    θ = sp.atan(0.75)  # angle in radians
    a = 0.8
    # Example X: complex rotation-like matrix
    X = sp.Matrix([
        [a*sp.cos(θ), a*sp.sin(θ), 0],
        [-a*sp.sin(θ),  a*sp.cos(θ), 0],
        [0, 0, 1]
    ])

    # Example B: any 2x2 numeric or symbolic matrix
    B = sp.Matrix([
        [-0.32, 0.48, 0],
        [-2.4, 1.6, 0],
        [-3.48, 0.72, 1]
    ])

    # Compute normalized W and corresponding W'
    W, W_prime = find_W_Wprime_normalized(B, X)

    print("Normalized lower-triangular W:")
    sp.pprint(W)

    print("\nCorresponding W':")
    sp.pprint(W_prime)

    # Optional verification
    print("\nCheck that B = W * X * W':")
    check_expr = sp.simplify(W * X * W_prime - B)
    sp.pprint(check_expr)
