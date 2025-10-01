
import numpy as np
import matplotlib.pyplot as plt

# Define the matrix
A = np.array([[1.8, -0.9],
              [1.0,  0.0]])

# Compute A^x for x from 1 to 200 and collect top-left element (position [0, 0])
powers = range(1, 201)
top_left_values = [np.linalg.matrix_power(A, x)[0, 0] for x in powers]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(powers, top_left_values, label='Top-left entry of A^x')
plt.xlabel('x (Power)')
plt.ylabel('Top-left value of A^x')
plt.title('Top-left Corner of Matrix [[1.8, -0.9], [1, 0]]^x for x from 1 to 200')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()