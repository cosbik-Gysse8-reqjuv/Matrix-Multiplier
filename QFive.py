import numpy as np
import matplotlib.pyplot as plt

# Example: 3x3 system matrix B
B = np.array([
    [-0.32, 0.48, 0],
    [-2.4, 1.6, 0],
    [-3.48, 0.72, 1]
])

# Horizon
k_max = 20

# Number of variables
n = B.shape[0]

# Create impulse vectors a1, a2, a3
impulses = np.eye(n)  # a1=[1,0,0]', a2=[0,1,0]', a3=[0,0,1]'

# Initialize response arrays
responses = np.zeros((n, n, k_max+1))  # shape: (variable, impulse, time)

# Compute impulse responses
for j in range(n):
    responses[:, j, 0] = impulses[:, j]  # time 0 response is the impulse itself
    for k in range(1, k_max+1):
        responses[:, j, k] = B @ responses[:, j, k-1]

# Plot impulse responses
time = np.arange(k_max+1)
for j in range(n):   # impulse
    plt.figure(figsize=(12, 8))
    for i in range(n):       # variable
        plt.plot(time, responses[i, j, :], marker='o', label=f'x{i+1} response')
    plt.xlabel('Horizon k')
    plt.ylabel('Value')
    plt.title(f'Response to impulse a{j+1}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filename = f'impulse_response_a{j+1}.png'
    plt.savefig(filename)
    print(f'Plot saved as {filename}')
    plt.close()
