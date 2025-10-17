import numpy as np
import matplotlib.pyplot as plt

def compute_irf(B, S, shock_vector, k=20):
    n = B.shape[0]
    irf = np.zeros((n, k+1))
    u0 = (S @ shock_vector).flatten()
    irf[:, 0] = u0
    for t in range(1, k+1):
        irf[:, t] = B @ irf[:, t-1]
    return irf

def plot_irf_S_shocks_subplots(B, S_list, shock_list, k=20, S_labels=None, shock_labels=None,
                               var_labels=None, filename='IRF_S_shocks_subplots.png'):
    """
    Plot impulse responses for multiple S matrices and shocks in separate subplots with custom variable labels.
    
    var_labels: list of strings for each variable (length = number of variables)
    """
    horizons = np.arange(k+1)
    n = B.shape[0]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    linestyles = ['-', '--']  # solid for var1, dashed for var2

    fig, axes = plt.subplots(1, len(shock_list), figsize=(12,5), sharey=True)
    
    if len(shock_list) == 1:
        axes = [axes]
    
    for sh_idx, shock in enumerate(shock_list):
        ax = axes[sh_idx]
        for s_idx, S in enumerate(S_list):
            irf = compute_irf(B, S, shock, k)
            for var_idx in range(n):
                label_parts = []
                if var_labels is not None:
                    label_parts.append(var_labels[var_idx])
                if S_labels is not None:
                    label_parts.append(S_labels[s_idx])
                label = ", ".join(label_parts) if label_parts else f"var {var_idx+1}"
                
                ax.plot(horizons, irf[var_idx, :],
                        color=colors[s_idx % len(colors)],
                        linestyle=linestyles[var_idx % len(linestyles)],
                        marker='o',
                        label=label)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Horizon')
        ax.set_title(shock_labels[sh_idx] if shock_labels else f'Shock {sh_idx+1}')
        ax.grid(True)
        if sh_idx == 0:
            ax.set_ylabel('Response')
    
    axes[0].legend(fontsize=8, loc='upper right')
    plt.suptitle('Impulse Responses for Multiple S Matrices and Shocks')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure as {filename}")
    plt.show()

# ---------------- Example ----------------
B = np.array([[0.84, -0.08],
              [-0.08, 0.96]])

# Specify S matrices for two theta values
theta1 = 0
theta2 = 0.1799 
S1 = np.array([[5*np.cos(theta1), -5*np.sin(theta1)],
               [2.2*np.cos(theta1)+0.4*np.sin(theta1), -2.2*np.sin(theta1)+0.4*np.cos(theta1)]])
S2 = np.array([[5*np.cos(theta2), -5*np.sin(theta2)],
               [2.2*np.cos(theta2)+0.4*np.sin(theta2), -2.2*np.sin(theta2)+0.4*np.cos(theta2)]])
S_list = [S1, S2]
S_labels = [f'theta={np.round(theta1,2)}', f'theta={np.round(theta2,2)}']

# Two shocks with custom labels
shock1 = np.array([[1], [0]])  # demand shock
shock2 = np.array([[0], [1]])  # supply shock
shock_list = [shock1, shock2]
shock_labels = ['Demand Shock', 'Supply Shock']

# Variable labels
var_labels = ['Demand impulse response', 'Supply impulse response']

# Plot and save
plot_irf_S_shocks_subplots(B, S_list, shock_list, k=20,
                           S_labels=S_labels, shock_labels=shock_labels,
                           var_labels=var_labels,
                           filename='IRF_S_shocks_subplots.png')
