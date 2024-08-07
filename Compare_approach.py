import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams['font.family'] = 'Segoe UI'

# Generate a full factorial design in a 3D cube
X_full = np.linspace(-1, 1, 5)
Y_full = np.linspace(-1, 1, 5)
Z_full = np.linspace(-1, 1, 5)
X_full, Y_full, Z_full = np.meshgrid(X_full, Y_full, Z_full)

# Flatten the arrays for easy plotting
X_full_flat = X_full.flatten()
Y_full_flat = Y_full.flatten()
Z_full_flat = Z_full.flatten()

# Create an inverted Gaussian surface for the second plot
X = np.linspace(-3, 3, 50)
Y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(X, Y)
Z = np.exp(-0.5 * (X**2 + Y**2))

# Generate 25 points near the global optimum on the Gaussian surface
x_points = np.linspace(-1, 1, 5)
y_points = np.linspace(-1, 1, 5)
x_points, y_points = np.meshgrid(x_points, y_points)
z_points = np.exp(-0.5 * (x_points**2 + y_points**2))

# Flatten the arrays for easy plotting
X_optimized = x_points.flatten()
Y_optimized = y_points.flatten()
Z_optimized = z_points.flatten()

# Find the index of the global optimal point (maximum) at (0, 0)
global_optimal_idx = np.argmax(Z_optimized)

# Offset for the global optimal point star
Z_optimal_star = Z_optimized[global_optimal_idx] + 0.1

# Create a figure with 3D subplots
fig = plt.figure(figsize=(14, 7))

# Plot 1: Full Factorial Design in a 3D cube
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_full_flat, Y_full_flat, Z_full_flat, color='blue', label='Experimental Points')
ax1.set_title('Full Factorial Design in 3D Cube')
ax1.set_xlabel('Factor 1')
ax1.set_ylabel('Factor 2')
ax1.set_zlabel('Factor 3')
ax1.legend()

# Plot 2: Inverted Gaussian Manifold with points near the global optimum
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, color='lightblue', alpha=0.5)
ax2.scatter(X_optimized, Y_optimized, Z_optimized, color='blue', label='Experimental Points')
ax2.scatter(X_optimized[global_optimal_idx], Y_optimized[global_optimal_idx], Z_optimal_star, color='red', s=200, marker='*', label='Global Optimal')
ax2.set_title('Optimized Solution Space with Inverted Gaussian Manifold')
ax2.set_xlabel('Factor 1')
ax2.set_ylabel('Factor 2')
ax2.set_zlabel('Response')
ax2.legend()

plt.tight_layout()

# Save the figure as a PNG file
plt.savefig('optimized_solution_space.png')

plt.show()
