
import numpy as np
import matplotlib.pyplot as plt

# Define the matrix
A = np.array([[1.85, 0.9],
              [1.0,  0.0]])

# Compute A^x for x from 1 to 200 and collect top-left element (position [0, 0])
powers = range(1, 201)
top_left_values = [np.linalg.matrix_power(A, x)[0, 0] for x in powers]


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(powers, top_left_values, label='Top-left entry of A^x')
plt.xlabel('k (years after shock)')
plt.ylabel('Impulse response')
plt.title('Impulse Response Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('matrix_power_plot.png')