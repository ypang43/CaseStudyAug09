import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
file_path = 'Data.xlsx'
data = pd.read_excel(file_path)

# Extract ingredients and properties
ingredients = data.iloc[:4, 1:].T
properties = data.iloc[4:, 1:].T

# Clean the data by replacing non-numeric values with NaN and then filling with column mean
properties.replace('-', np.nan, inplace=True)
properties = properties.apply(pd.to_numeric, errors='coerce')
properties.fillna(properties.mean(), inplace=True)

# Standardize the data
scaler = StandardScaler()
ingredients_scaled = scaler.fit_transform(ingredients)
properties_scaled = scaler.fit_transform(properties)

# Perform PCA to reduce dimensions to 3
pca = PCA(n_components=3)
ingredients_pca = pca.fit_transform(ingredients_scaled)

# Create polynomial features including interaction terms
poly = PolynomialFeatures(degree=3, include_bias=False)
ingredients_poly = poly.fit_transform(ingredients_pca)

# Fit Lasso regression for the first property (target function)
lasso = Lasso(alpha=0.1)
lasso.fit(ingredients_poly, properties_scaled[:, 0])

# Create a mesh grid for the parameter space in 3D
x = np.linspace(ingredients_pca[:, 0].min(), ingredients_pca[:, 0].max(), 50)
y = np.linspace(ingredients_pca[:, 1].min(), ingredients_pca[:, 1].max(), 50)
z = np.linspace(ingredients_pca[:, 2].min(), ingredients_pca[:, 2].max(), 50)
xx, yy, zz = np.meshgrid(x, y, z)

# Generate polynomial features for the grid points
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
grid_poly = poly.transform(grid_points)

# Predict the target function over the parameter space
predicted = lasso.predict(grid_poly).reshape(xx.shape)

# 3D plot without projection
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx[:, :, 0], yy[:, :, 0], predicted[:, :, 0], cmap='jet', alpha=0.8)

# Add scatter plot of actual data points for reference
ax.scatter(ingredients_pca[:, 0], ingredients_pca[:, 1], properties_scaled[:, 0], c='r', marker='o', label='Data points')

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('Target Function')
ax.legend()
plt.title('3D Manifold of Solution Space with Interaction Terms')
plt.show()
