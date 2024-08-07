import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import plotly.graph_objects as go

def load_and_prepare_data(file_path):
    data = pd.read_excel(file_path)

    # Extract ingredients and properties
    ingredients = data.iloc[:4, 1:].T
    properties = data.iloc[4:, 1:].T

    # Clean the data by replacing non-numeric values with NaN and then filling with column mean
    properties.replace('-', np.nan, inplace=True)
    properties = properties.apply(pd.to_numeric, errors='coerce')
    properties.fillna(properties.mean(), inplace=True)

    return ingredients, properties

def perform_pca_and_standardize(ingredients, properties):
    # Standardize the data
    scaler = StandardScaler()
    ingredients_scaled = scaler.fit_transform(ingredients)
    properties_scaled = scaler.fit_transform(properties)

    # Perform PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    ingredients_pca = pca.fit_transform(ingredients_scaled)

    return ingredients_pca, properties_scaled, pca

def create_polynomial_features(ingredients_pca):
    # Create polynomial features including interaction terms
    poly = PolynomialFeatures(degree=3, include_bias=False)
    ingredients_poly = poly.fit_transform(ingredients_pca)
    return ingredients_poly, poly

def fit_lasso_regression(ingredients_poly, properties_scaled):
    # Fit Lasso regression for the first property (target function)
    lasso = Lasso(alpha=0.1)
    lasso.fit(ingredients_poly, properties_scaled[:, 0])
    return lasso

def generate_mesh_grid(ingredients_pca):
    # Create a mesh grid for the parameter space in 3D
    x = np.linspace(ingredients_pca[:, 0].min(), ingredients_pca[:, 0].max(), 50)
    y = np.linspace(ingredients_pca[:, 1].min(), ingredients_pca[:, 1].max(), 50)
    z = np.linspace(ingredients_pca[:, 2].min(), ingredients_pca[:, 2].max(), 50)
    xx, yy, zz = np.meshgrid(x, y, z)

    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    return xx, yy, zz, grid_points

def predict_on_grid(grid_points, lasso, poly, xx_shape):
    # Generate polynomial features for the grid points
    grid_poly = poly.transform(grid_points)

    # Predict the target function over the parameter space
    predicted = lasso.predict(grid_poly).reshape(xx_shape)
    return predicted

def plot_regression_manifold(xx, yy, zz, predicted, ingredients_pca, properties_scaled, optimal_point_pca):
    # Create a 3D plot with Plotly
    fig = go.Figure(data=[go.Surface(x=xx[:, :, 0], y=yy[:, :, 0], z=predicted[:, :, 0], colorscale='Jet', opacity=0.8)])

    # Add scatter plot of actual data points
    fig.add_trace(go.Scatter3d(x=ingredients_pca[:, 0], y=ingredients_pca[:, 1], z=properties_scaled[:, 0], mode='markers', marker=dict(size=5, color='red'), name='Data points'))

    # Highlight the optimal point
    fig.add_trace(go.Scatter3d(x=[optimal_point_pca[0]], y=[optimal_point_pca[1]], z=[optimal_point_pca[2]], mode='markers', marker=dict(size=10, color='blue'), name='Optimal Point'))

    fig.update_layout(scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='Target Function'
    ), title='3D Manifold of Solution Space with Interaction Terms')

    return fig

def train_and_plot_regression_plane(file_path, optimal_point):
    ingredients, properties = load_and_prepare_data(file_path)
    ingredients_pca, properties_scaled, pca = perform_pca_and_standardize(ingredients, properties)
    ingredients_poly, poly = create_polynomial_features(ingredients_pca)
    lasso = fit_lasso_regression(ingredients_poly, properties_scaled)
    xx, yy, zz, grid_points = generate_mesh_grid(ingredients_pca)
    predicted = predict_on_grid(grid_points, lasso, poly, xx.shape)
    
    # Transform the optimal point to PCA space
    optimal_point_pca = pca.transform([optimal_point])
    
    fig = plot_regression_manifold(xx, yy, zz, predicted, ingredients_pca, properties_scaled, optimal_point_pca[0])
    return fig
