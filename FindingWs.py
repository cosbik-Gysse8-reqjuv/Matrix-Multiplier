import sympy as sp

def find_W_Wprime(B, X):
    """
    Given B and X (possibly symbolic with sin, cos, etc.),
    find symbolic W and W' such that B = W * X * W'.
    """
    B = sp.Matrix(B)
    X = sp.Matrix(X)
    # Check invertibility
    if X.det() == 0:
        raise ValueError("Matrix X must be invertible.")
    if B.det() == 0:
        raise ValueError("Matrix B must be invertible.")
    n = B.shape[0]
    # Build normalized lower triangular W (ones on diagonal, symbolic below)
    w_symbols = sp.symbols(f'w0:{int(n*(n-1)/2)}')
    idx = 0
    W = sp.eye(n)
    for i in range(n):
        for j in range(i):
            W[i, j] = w_symbols[idx]
            idx += 1
    # Compute W' = (W*X)^(-1) * B
    WX = W * X
    if WX.det() == 0:
        raise ValueError("W*X is not invertible for these symbolic variables.")
    W_prime = WX.inv() * B
    return sp.simplify(W), sp.simplify(W_prime)


# Example usage
if __name__ == "__main__":
    θ = sp.atan(0.75)  # angle in radians
    a = 0.8
    # Define matrices with sine and cosine
    X = sp.Matrix([
        [a*sp.cos(θ), a*sp.sin(θ), 0],
        [-a*sp.sin(θ),  a*sp.cos(θ), 0],
        [0, 0, 1]
    ])
    B = sp.Matrix([
        [-0.32, 0.48, 0],
        [-2.4, 1.6, 0],
        [-3.48, 0.72, 1]
    ])
    W, W_prime = find_W_Wprime(B, X)
    print("Numerical W:")
    sp.pprint(W.evalf())
    print("\nNumerical W':")
    sp.pprint(W_prime.evalf())
    # Verify numerically that B ≈ W * X * W'
    print("\nCheck (W * X * W' - B):")
    sp.pprint((W * X * W_prime - B).evalf())
