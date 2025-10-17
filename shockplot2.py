import numpy as np
import matplotlib.pyplot as plt

def compute_irf_structural(B, A, shock_vectors, k=20):
    """
    Compute impulse responses for shocks transformed by A (structural shocks).

    Parameters:
        B : 2x2 numpy array
            Transition matrix
        A : 2x2 numpy array
            Impact matrix
        shock_vectors : list of 1D numpy arrays
            Each is a column vector a_i, e.g., [1,0]' or [0,1]'
        k : int
            Horizon

    Returns:
        irfs : numpy array of shape (n, len(shock_vectors), k+1)
            IRFs[:, j, t] = response at horizon t to shock j
    """
    n = B.shape[0]
    m = len(shock_vectors)
    irfs = np.zeros((n, m, k+1))
    
    # Horizon 0: apply structural shocks u = A * a_i
    for j, a in enumerate(shock_vectors):
        u0 = (A @ a).flatten()
        irfs[:, j, 0] = u0
    
    # Horizons 1..k: propagate via B
    for t in range(1, k+1):
        for j in range(m):
            irfs[:, j, t] = B @ irfs[:, j, t-1]
    
    return irfs

def plot_irf_structural(irfs, k=20, save_dir='./'):
    n, m, _ = irfs.shape
    horizons = np.arange(k+1)
    
    for j in range(m):
        plt.figure(figsize=(6,4))
        for i in range(n):
            plt.plot(horizons, irfs[i, j, :], marker='o', label=f'var {i+1}')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.title(f'Response to structural shock {j+1}')
        plt.xlabel('Horizon')
        plt.ylabel('Response')
        plt.legend()
        plt.grid(True)
        
        filename = f"{save_dir}StructShock{j+1}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        plt.close()


# ---------------- Example ----------------
B = np.array([[0.84, -0.08],
              [-0.08, 0.96]])

A = np.array([[0.895, 4.92],
              [0.0002, 2.236]])

# Define the shocks you want to apply (as vectors)
a1 = np.array([[1], [0]])  # first shock
a2 = np.array([[0], [1]])  # second shock

shock_vectors = [a1, a2]
k = 20

irfs = compute_irf_structural(B, A, shock_vectors, k)
plot_irf_structural(irfs, k, save_dir='./')
