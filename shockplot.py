import numpy as np
import matplotlib.pyplot as plt

def compute_irf(B, A, k=20):
    n = B.shape[0]
    irfs = np.zeros((n, n, k+1))
    irfs[:, :, 0] = A
    for t in range(1, k+1):
        irfs[:, :, t] = B @ irfs[:, :, t-1]
    return irfs

def plot_irf_by_shock(irfs, k=20, save_dir='./'):
    n = irfs.shape[0]
    horizons = np.arange(k+1)
    
    for j in range(n):  # iterate over shocks
        plt.figure(figsize=(6,4))
        for i in range(n):  # plot each variable's response to this shock
            plt.plot(horizons, irfs[i, j, :], marker='o', label=f'var {i+1}')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.title(f'Response to shock {j+1}')
        plt.xlabel('Horizon')
        plt.ylabel('Response')
        plt.legend()
        plt.grid(True)
        
        # Save figure as PNG
        filename = f"{save_dir}IRF_shock{j+1}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
        
        plt.close()  # close figure to free memory

# Example usage
B = np.array([[0.84, -0.08],
              [-0.08, 0.96]])
A = np.array([[0.895, 4.92],
              [0.0002, 2.236]])
k = 20

irfs = compute_irf(B, A, k)
plot_irf_by_shock(irfs, k, save_dir='./')
