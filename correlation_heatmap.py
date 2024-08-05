# correlation_heatmap.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import streamlit as st
import numpy as np

def plot_correlation_heatmap(data_file):
    # Load and preprocess the data
    data = pd.read_excel(data_file, header=None)
    data = data.T
    data.columns = data.iloc[0] + '_' + data.columns.astype(str)
    data = data[1:]
    data.reset_index(drop=True, inplace=True)
    data.replace('-', np.nan, inplace=True)
    
    # Check and impute necessary columns
    columns_needed = [
        'Ingredient A_1', 'Ingredient B_2', 'Ingredient C_3', 'Ingredient D_4', 
        'Initial: Fiber tear #1_6', 'Initial: Fiber tear #2_7', 'Initial: Fast Load_8', 'Initial: slow load_9',
        'After 500 hrs: Fiber tear After Aging #1_10', 'After 500 hrs: Fiber tear After Aging #2_11', 
        'After 500 hrs: Fast Load_12', 'After 500 hrs: slow load_13'
    ]
    
    if all(column in data.columns for column in columns_needed):
        imputer = SimpleImputer(strategy='mean')
        data[columns_needed] = imputer.fit_transform(data[columns_needed])
    
        # Prepare data for the heatmap
        ingredients = data[['Ingredient A_1', 'Ingredient B_2', 'Ingredient C_3', 'Ingredient D_4']]
        properties = data.drop(['Ingredient A_1', 'Ingredient B_2', 'Ingredient C_3', 'Ingredient D_4'], axis=1)
        combined_df = pd.concat([ingredients.reset_index(drop=True), properties.reset_index(drop=True)], axis=1)
    
        # Ensure only numeric data for correlation matrix
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        combined_df[numeric_cols] = combined_df[numeric_cols].fillna(combined_df[numeric_cols].mean())
    
        # Plot correlation heatmap
        plt.figure(figsize=(6, 5))
        corr_matrix = combined_df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix Including Ingredients')
        st.pyplot(plt)
    else:
        st.error("The necessary columns are not present in the dataset.")
