import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Load data
file_path = 'Data.xlsx'
data = pd.read_excel(file_path, header=None)
data.columns = data.iloc[0]  # Set the first row as column names
data = data[1:]  # Remove the first row from the data
data.reset_index(drop=True, inplace=True)
data.replace('-', np.nan, inplace=True)

# Debug: print the column names and the first few rows
print("Column Names:", data.columns)
print("First few rows of the data:\n", data.head())

# Define necessary columns
columns_needed = [
    'Ingredient A', 'Ingredient B', 'Ingredient C', 'Ingredient D', 
    'Initial: Fiber tear #1', 'Initial: Fiber tear #2', 'Initial: Fast Load', 'Initial: slow load',
    'After 1000 hrs: Fiber tear After Aging #1', 'After 1000 hrs: Fiber tear After Aging #2', 
    'After 1000 hrs: Fast Load', 'After 1000 hrs: slow load'
]

# Ensure the columns are numeric
for column in columns_needed:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Impute missing values
data.fillna(data.mean(), inplace=True)

# Prepare data for PCA
ingredients = data[['Ingredient A', 'Ingredient B', 'Ingredient C', 'Ingredient D']]
properties = data[[
    'Initial: Fiber tear #1', 'Initial: Fiber tear #2', 'Initial: Fast Load', 'Initial: slow load',
    'After 1000 hrs: Fiber tear After Aging #1', 'After 1000 hrs: Fiber tear After Aging #2', 
    'After 1000 hrs: Fast Load', 'After 1000 hrs: slow load'
]]

# Standardize the data
scaler = StandardScaler()
ingredients_scaled = scaler.fit_transform(ingredients)
properties_scaled = scaler.fit_transform(properties)

# Perform PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(ingredients_scaled)

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
pca_df = pd.concat([pca_df, properties.reset_index(drop=True)], axis=1)

# Plot PCA results with ingredients as vectors
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', data=pca_df)
for i in range(len(ingredients.columns)):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
              color='r', alpha=0.5)
    plt.text(pca.components_[0, i]*1.15, pca.components_[1, i]*1.15, 
             ingredients.columns[i], color='g', ha='center', va='center')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA of Ingredients')
plt.grid(True)
plt.show()

# Plot importance of ingredients
explained_variance = pca.explained_variance_ratio_
fig = px.bar(x=ingredients.columns, y=explained_variance, title='Ingredient Importance',
             labels={'x': 'Ingredients', 'y': 'Explained Variance'})
fig.show()

# Save the plots as HTML files for interactive viewing
pca_plot = go.Figure(data=go.Scatter(x=pca_df['PCA1'], y=pca_df['PCA2'], mode='markers',
                                     marker=dict(color=pca_df['Initial: Fiber tear #1'], colorscale='Viridis', size=10),
                                     text=pca_df['Initial: Fiber tear #1']))
pca_plot.update_layout(title='PCA Plot with Fiber Tear #1', xaxis_title='PCA1', yaxis_title='PCA2')
pca_plot.write_html('pca_plot.html')

importance_plot = go.Figure(data=go.Bar(x=ingredients.columns, y=explained_variance))
importance_plot.update_layout(title='Ingredient Importance', xaxis_title='Ingredients', yaxis_title='Explained Variance')
importance_plot.write_html('importance_plot.html')
