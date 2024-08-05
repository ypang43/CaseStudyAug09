# heatmap.py
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def display_heatmap(df):
    st.subheader("Correlation Matrix")
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Including Ingredients')
    st.pyplot(plt)
