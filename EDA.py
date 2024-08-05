import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
import seaborn as sns
import numpy as np

# Load the data from the provided Excel file
data_file_path = 'Data.xlsx'
data = pd.read_excel(data_file_path, sheet_name='Input data')

# Transpose the data to have ingredients as features and formulations as samples
data_t = data.set_index('Column').T

# Separate ingredients and properties
ingredients = data_t[['Ingredient A', 'Ingredient B', 'Ingredient C', 'Ingredient D']]
properties = data_t.drop(['Ingredient A', 'Ingredient B', 'Ingredient C', 'Ingredient D', 'Other'], axis=1)

# Debug: Print the first few rows of ingredients and properties
print("Ingredients:")
print(ingredients.head())
print("\nProperties:")
print(properties.head())

def get_first_value(row, column_name):
    value = row[column_name]
    if isinstance(value, pd.Series):
        return value.iloc[0] if not value.empty else np.nan
    return value

def label_property(row):
    initial_fiber_tear_1 = get_first_value(row, 'Initial: Fiber tear #1')
    initial_fiber_tear_2 = get_first_value(row, 'Initial: Fiber tear #2')
    initial_fast_load = get_first_value(row, 'Initial: Fast Load')
    initial_slow_load = get_first_value(row, 'Initial: slow load')
    after_500_fiber_tear_1 = get_first_value(row, 'After 500 hrs: Fiber tear After Aging #1')
    after_500_fiber_tear_2 = get_first_value(row, 'After 500 hrs: Fiber tear After Aging #2')
    after_500_fast_load = get_first_value(row, 'After 500 hrs: Fast Load')
    after_500_slow_load = get_first_value(row, 'After 500 hrs: slow load')

    # Debugging: print extracted values
    print(f"Initial Fiber Tear #1: {initial_fiber_tear_1}, Initial Fiber Tear #2: {initial_fiber_tear_2}, "
          f"Initial Fast Load: {initial_fast_load}, Initial Slow Load: {initial_slow_load}, "
          f"After 500 hrs Fiber Tear #1: {after_500_fiber_tear_1}, After 500 hrs Fiber Tear #2: {after_500_fiber_tear_2}, "
          f"After 500 hrs Fast Load: {after_500_fast_load}, After 500 hrs Slow Load: {after_500_slow_load}")

    if pd.notna(initial_fiber_tear_1) and pd.notna(initial_fiber_tear_2) and \
       pd.notna(initial_fast_load) and pd.notna(after_500_fast_load) and \
       pd.notna(initial_slow_load) and pd.notna(after_500_slow_load):
        if (initial_fiber_tear_1 < 20 and initial_fiber_tear_2 < 20 and
            initial_fast_load > 4.25 and after_500_fast_load > 4 and
            initial_slow_load > 4.25 and after_500_slow_load > 4):
            return 'High Performance'
        elif initial_fiber_tear_1 >= 20 or initial_fiber_tear_2 >= 20:
            return 'High Tear'
        elif min(initial_fast_load, after_500_fast_load) <= 4.25:
            return 'Low Fast Load'
        elif min(initial_slow_load, after_500_slow_load) <= 4.25:
            return 'Low Slow Load'
        else:
            return 'Other'
    return 'Missing Data'

# Apply the labeling function
properties['Label'] = properties.apply(label_property, axis=1)

# Debugging: Ensure 'Label' column is added correctly
print("Label value counts:")
print(properties['Label'].value_counts())

# Debugging: Print rows that are labeled as 'High Tear' to understand why
print("Rows labeled as 'High Tear':")
print(properties[properties['Label'] == 'High Tear'])

# Fill NaN values with 0 for PCA
ingredients_filled = ingredients.fillna(0)

# Standardize the data
scaler = StandardScaler()
ingredients_scaled = scaler.fit_transform(ingredients_filled)

# Apply PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(ingredients_scaled)

# Create a DataFrame for PCA components
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
pca_df['Label'] = properties['Label']

# Debugging: Ensure there are data points in PCA DataFrame
print("PCA DataFrame head:")
print(pca_df.head())

# Check for rows where the Label is NaN in PCA DataFrame
missing_label_df = pca_df[pca_df['Label'] == 'Missing Data']
print("Rows with 'Missing Data' label:")
print(missing_label_df)

# Fill NaN labels with 'Missing Data' in PCA DataFrame for plotting purposes
pca_df['Label'] = pca_df['Label'].fillna('Missing Data')

# Plot PCA with annotations and circles
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Label', data=pca_df, s=100, alpha=0.6, edgecolor='k')

# Plot vectors for ingredients
for i, (var, component) in enumerate(zip(['Ingredient A', 'Ingredient B', 'Ingredient C', 'Ingredient D'], pca.components_.T)):
    plt.arrow(0, 0, component[0]*5, component[1]*5, color='r', alpha=0.75, head_width=0.15)
    plt.text(component[0]*5.2, component[1]*5.2, var, color='r', ha='center', va='center')

# Annotations for quadrants
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.text(3, 3, 'Positive Correlation', fontsize=12, color='green', ha='center', bbox=dict(facecolor='white', edgecolor='green'))
plt.text(-3, 3, 'Negative Correlation', fontsize=12, color='red', ha='center', bbox=dict(facecolor='white', edgecolor='red'))
plt.text(-3, -3, 'Negative Correlation', fontsize=12, color='red', ha='center', bbox=dict(facecolor='white', edgecolor='red'))
plt.text(3, -3, 'Positive Correlation', fontsize=12, color='green', ha='center', bbox=dict(facecolor='white', edgecolor='green'))

# Plot settings
plt.title('PCA of Ingredient Compositions')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
plt.grid(False)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()

# Decision Tree for the key property
property_name = 'Initial: Fiber tear #1'
combined_data = pd.concat([ingredients, properties], axis=1)

X = combined_data[['Ingredient A', 'Ingredient B', 'Ingredient C', 'Ingredient D']]
y = combined_data[property_name]

# Handle NaN values in y
y = y.fillna(y.mean())

# Train Decision Tree
tree_model = DecisionTreeRegressor(max_depth=4)
tree_model.fit(X, y)

# Plot Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=['Ingredient A', 'Ingredient B', 'Ingredient C', 'Ingredient D'], 
          filled=True, rounded=True, fontsize=12)
plt.title(f'Decision Tree for {property_name}')
plt.show()
