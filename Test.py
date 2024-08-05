import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates
from sklearn.cluster import KMeans
import numpy as np

# Load the data from the provided Excel file
data_file_path = 'Data.xlsx'
data = pd.read_excel(data_file_path, sheet_name='Input data')

# Transpose the data to have ingredients as features and formulations as samples
data_t = data.set_index('Column').T

# Separate ingredients and properties
ingredients = data_t[['Ingredient A', 'Ingredient B', 'Ingredient C', 'Ingredient D']]
properties = data_t.drop(['Ingredient A', 'Ingredient B', 'Ingredient C', 'Ingredient D', 'Other'], axis=1)

# Replace non-numeric values with NaN
properties.replace('-', np.nan, inplace=True)
properties = properties.apply(pd.to_numeric)

# Combine ingredients with original properties
combined_df = pd.concat([ingredients.reset_index(drop=True), properties.reset_index(drop=True)], axis=1)

# Correlation Matrix
plt.figure(figsize=(12, 10))
corr_matrix = combined_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Including Ingredients')
plt.show()

# Pair Plot - Ensure only numeric columns are used
numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
sns.pairplot(combined_df[numeric_cols])
plt.suptitle('Pair Plot Including Ingredients', y=1.02)
plt.show()

# Standardize the ingredients data for PCA and clustering
scaler = StandardScaler()
ingredients_scaled = scaler.fit_transform(ingredients.fillna(0))

# Apply PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(ingredients_scaled)

# Create a DataFrame for PCA components
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])

# Combine PCA components with original properties
combined_df_pca = pd.concat([pca_df, properties.reset_index(drop=True)], axis=1)

# Parallel Coordinates Plot - Drop rows with NaN values
plt.figure(figsize=(12, 8))
parallel_coordinates(combined_df.dropna(), 'Initial: Fiber tear #1', colormap='viridis')
plt.title('Parallel Coordinates Plot Including Ingredients')
plt.show()

# Cluster Analysis
# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
combined_df_pca['Cluster'] = kmeans.fit_predict(ingredients_scaled)

# Plot clusters in PCA space
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=combined_df_pca, palette='viridis', s=100, alpha=0.6, edgecolor='k')
plt.title('PCA with KMeans Clusters')
plt.show()
