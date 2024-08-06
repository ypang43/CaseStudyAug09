import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def display_heatmap(ingredients, properties):
    st.subheader("Correlation Matrix")
    combined_df = pd.concat([ingredients, properties], axis=1)

    # Ensure only numeric data for correlation matrix
    numeric_combined_df = combined_df.apply(pd.to_numeric, errors='coerce')
    numeric_combined_df = numeric_combined_df.fillna(numeric_combined_df.mean())

    plt.figure(figsize=(12, 4))
    corr_matrix = numeric_combined_df.corr().loc[ingredients.columns, properties.columns]
    sns.heatmap(corr_matrix, annot=True, cmap='seismic', cbar_kws={'label': '(+) Positive (-) Negative Correlation'})
    
    st.pyplot(plt)
