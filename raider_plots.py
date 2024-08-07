import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.impute import SimpleImputer

# Set font properties for title
font_properties = FontProperties(fname="C:\\Windows\\Fonts\\segoeui.ttf", size=18, weight='bold')

# Function to load data and preprocess
def load_and_preprocess_data(file_name):
    data = pd.read_excel(file_name, header=None)
    data = data.T
    data.columns = data.iloc[0] + '_' + data.columns.astype(str)
    data = data[1:]
    data.reset_index(drop=True, inplace=True)
    data.replace('-', np.nan, inplace=True)

    # Check and impute necessary columns
    columns_needed = [
        'Ingredient A_1', 'Ingredient B_2', 'Ingredient C_3', 'Ingredient D_4', 
        'Initial: Fiber tear #1_6', 'Initial: Fiber tear #2_7', 'Initial: Fast Load_8', 'Initial: slow load_9',
        'After 1000 hrs: Fiber tear After Aging #1_14', 'After 1000 hrs: Fiber tear After Aging #2_15', 
        'After 1000 hrs: Fast Load_16', 'After 1000 hrs: slow load_17'
    ]

    if all(column in data.columns for column in columns_needed):
        imputer = SimpleImputer(strategy='mean')
        data[columns_needed] = imputer.fit_transform(data[columns_needed])

    # Remove unnecessary suffixes
    data.columns = [col.split('_')[0] for col in data.columns]

    # Rename columns for clarity
    rename_columns = {
        'After 1000 hrs: Fiber tear After Aging #1': 'Fiber tear #1_aging',
        'After 1000 hrs: Fiber tear After Aging #2': 'Fiber tear #2_aging',
        'After 1000 hrs: Fast Load': 'Fast Load_aging',
        'After 1000 hrs: slow load': 'slow load_aging'
    }
    data.rename(columns=rename_columns, inplace=True)

    return data

# Function to create a radar plot
def create_radar_plot(ax, corr_df, ingredients, properties, title, fill_negative=False, legend=False):
    labels = properties
    num_vars = len(labels)

    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Define enough colors for all ingredients and pairs
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange']

    if fill_negative:
        # Fill the negative area with light gray
        ax.fill_between(angles, 0, -1, color='lightgrey', alpha=0.5)

    for idx, ingredient in enumerate(ingredients):
        if idx >= len(colors):  # Ensure we do not run out of colors
            color = np.random.rand(3,)
        else:
            color = colors[idx]

        if fill_negative:
            values = corr_df.loc[ingredient].clip(upper=0).abs().tolist()
        else:
            values = corr_df.loc[ingredient].clip(lower=0).tolist()

        values += values[:1]  # Complete the loop

        ax.plot(angles, values, color=color, linewidth=2, label=ingredient)
        ax.fill(angles, values, color=color, alpha=0.25)

    # Labels for each point
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=14, wrap=True)
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    # Set range for radar plot
    ax.set_ylim(-1 if fill_negative else 0, 1)

    # Add y-axis labels for context
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, 0.5, 1] if not fill_negative else [-1, -0.5, 0], fontsize=14)

    ax.set_title(title, fontproperties=font_properties, y=1.1)

    if legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.2), fontsize=12)

# Load and preprocess data
data = load_and_preprocess_data('DataPlotting.xlsx')

# Separate ingredients and properties
ingredients = ['Ingredient A', 'Ingredient B', 'Ingredient C', 'Ingredient D']
properties = [
    'Initial: Fiber tear #1', 'Initial: Fiber tear #2', 'Initial: Fast Load', 'Initial: slow load',
    'Fiber tear #1_aging', 'Fiber tear #2_aging', 'Fast Load_aging', 'slow load_aging'
]

# Calculate correlation for all ingredients
combined_df = pd.concat([data[ingredients], data[properties]], axis=1)
numeric_combined_df = combined_df.apply(pd.to_numeric, errors='coerce')
numeric_combined_df = numeric_combined_df.fillna(numeric_combined_df.mean())
corr_df = numeric_combined_df.corr().loc[ingredients, properties]

# Create a 2x1 grid of plots for single ingredients
fig1, axs1 = plt.subplots(2, 1, figsize=(14, 14), subplot_kw=dict(polar=True))
fig1.subplots_adjust(hspace=0.3)

# Plot positive correlations for single ingredients
create_radar_plot(axs1[0], corr_df, ingredients, properties, 'Positive Radar Plot for Single Ingredients', legend=True)

# Plot negative correlations for single ingredients
create_radar_plot(axs1[1], corr_df, ingredients, properties, 'Negative Radar Plot for Single Ingredients', fill_negative=True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('radar_plots_single_2x1.png', dpi=500)
plt.show()

# Compute differences between each pair of ingredients
ingredient_pairs = pd.DataFrame()
ingredient_pairs['A-B'] = numeric_combined_df['Ingredient A'] - numeric_combined_df['Ingredient B']
ingredient_pairs['A-C'] = numeric_combined_df['Ingredient A'] - numeric_combined_df['Ingredient C']
ingredient_pairs['A-D'] = numeric_combined_df['Ingredient A'] - numeric_combined_df['Ingredient D']
ingredient_pairs['B-C'] = numeric_combined_df['Ingredient B'] - numeric_combined_df['Ingredient C']
ingredient_pairs['B-D'] = numeric_combined_df['Ingredient B'] - numeric_combined_df['Ingredient D']
ingredient_pairs['C-D'] = numeric_combined_df['Ingredient C'] - numeric_combined_df['Ingredient D']

# Combine differences with properties
diff_combined_df = pd.concat([ingredient_pairs, data[properties]], axis=1)

# Compute correlation matrix for the differences
diff_corr_matrix = diff_combined_df.corr().loc[ingredient_pairs.columns, properties]

# Create a 2x1 grid of plots for ingredient pairs
fig2, axs2 = plt.subplots(2, 1, figsize=(14, 14), subplot_kw=dict(polar=True))
fig2.subplots_adjust(hspace=0.3)

# Plot positive correlations for ingredient pairs
create_radar_plot(axs2[0], diff_corr_matrix, ingredient_pairs.columns, properties, 'Positive Radar Plot for Ingredient Pairs', legend=True)

# Plot negative correlations for ingredient pairs
create_radar_plot(axs2[1], diff_corr_matrix, ingredient_pairs.columns, properties, 'Negative Radar Plot for Ingredient Pairs', fill_negative=True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('radar_plots_pairs_2x1.png', dpi=500)
plt.show()
