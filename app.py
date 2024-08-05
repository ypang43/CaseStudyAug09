import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
import openai
from dotenv import load_dotenv
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set the OpenAI API key
#openai.api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Define function to get LLM response
def get_llm_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

# Set Streamlit page configuration
st.set_page_config(page_title="Adhesive Technology Predictor Tool", layout="wide")

# Load CSS from file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("styles.css")

# Use st.image to display the logo
st.sidebar.image("HenkelLogo.png", use_column_width=True)

st.markdown(
    """
    <div class="title">
        Adhesive Technology Predictor Tool
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h2 class='virtual-assistant reduce-space'>Hi! I Am Your Virtual Henkel Assistant</h2>", unsafe_allow_html=True)

# LLM Response section
user_input = st.text_input("Ask a question about adhesive technology:", key='user_input')
if user_input:
    response = get_llm_response(user_input)
    st.markdown(f'<div class="gpt-response">{response}</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h2 class='sidebar-title reduce-space'>Adjust Formulation Composition</h2>", unsafe_allow_html=True)

# Define slider and input function
def slider_and_input(label, default_value):
    col1, col2 = st.sidebar.columns([3, 1])
    slider_value = col1.slider(label, 0.0, 3.0, default_value, step=0.01)
    input_value = col2.number_input(label, 0.0, 3.0, slider_value, step=0.01)
    return max(slider_value, input_value)

# Component sliders
component_a = slider_and_input("Ingredient A", 1.5)
component_b = slider_and_input("Ingredient B", 1.5)
component_c = slider_and_input("Ingredient C", 1.5)
component_d = slider_and_input("Ingredient D", 1.5)

total_components = component_a + component_b + component_c + component_d
remaining_percentage = 100 - total_components

if total_components > 12:
    st.sidebar.error("Total composition should not exceed 12. Adjust the components.")
else:
    st.sidebar.write(f"Remaining percentage for other ingredients: {remaining_percentage:.2f}%")

# Load data
data = pd.read_excel('Data.xlsx', header=None)

# Option to show/hide data table
if st.checkbox("Show Raw Data", key='show_raw_data'):
    st.write("Raw Data:")
    st.dataframe(data)

# Transform data
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
    combined_df[numeric_cols].fillna(combined_df[numeric_cols].mean(), inplace=True)

    # Plot correlation heatmap
    st.subheader("Correlation Matrix")
    plt.figure(figsize=(10, 8))
    corr_matrix = combined_df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Including Ingredients')
    st.pyplot(plt)

    # Prepare data for modeling
    X = data[['Ingredient A_1', 'Ingredient B_2', 'Ingredient C_3', 'Ingredient D_4']]
    y_initial_fiber_tear_1 = data['Initial: Fiber tear #1_6']
    y_initial_fiber_tear_2 = data['Initial: Fiber tear #2_7']
    y_initial_fast_load = data['Initial: Fast Load_8']
    y_initial_slow_load = data['Initial: slow load_9']
    y_after_500_fiber_tear_1 = data['After 500 hrs: Fiber tear After Aging #1_10']
    y_after_500_fiber_tear_2 = data['After 500 hrs: Fiber tear After Aging #2_11']
    y_after_500_fast_load = data['After 500 hrs: Fast Load_12']
    y_after_500_slow_load = data['After 500 hrs: slow load_13']

    # Model selection
    model_choice = st.selectbox("Choose a regression model", ["Ridge Regressor", "Lasso Regressor", "ElasticNet Regressor", "KNN Regressor"], key='model_choice')

    if model_choice == "Ridge":
        model_initial_fiber_tear_1 = Ridge().fit(X, y_initial_fiber_tear_1)
        model_initial_fiber_tear_2 = Ridge().fit(X, y_initial_fiber_tear_2)
        model_initial_fast_load = Ridge().fit(X, y_initial_fast_load)
        model_initial_slow_load = Ridge().fit(X, y_initial_slow_load)
        model_after_500_fiber_tear_1 = Ridge().fit(X, y_after_500_fiber_tear_1)
        model_after_500_fiber_tear_2 = Ridge().fit(X, y_after_500_fiber_tear_2)
        model_after_500_fast_load = Ridge().fit(X, y_after_500_fast_load)
        model_after_500_slow_load = Ridge().fit(X, y_after_500_slow_load)
    elif model_choice == "Lasso":
        model_initial_fiber_tear_1 = Lasso().fit(X, y_initial_fiber_tear_1)
        model_initial_fiber_tear_2 = Lasso().fit(X, y_initial_fiber_tear_2)
        model_initial_fast_load = Lasso().fit(X, y_initial_fast_load)
        model_initial_slow_load = Lasso().fit(X, y_initial_slow_load)
        model_after_500_fiber_tear_1 = Lasso().fit(X, y_after_500_fiber_tear_1)
        model_after_500_fiber_tear_2 = Lasso().fit(X, y_after_500_fiber_tear_2)
        model_after_500_fast_load = Lasso().fit(X, y_after_500_fast_load)
        model_after_500_slow_load = Lasso().fit(X, y_after_500_slow_load)
    elif model_choice == "ElasticNet":
        model_initial_fiber_tear_1 = ElasticNet().fit(X, y_initial_fiber_tear_1)
        model_initial_fiber_tear_2 = ElasticNet().fit(X, y_initial_fiber_tear_2)
        model_initial_fast_load = ElasticNet().fit(X, y_initial_fast_load)
        model_initial_slow_load = ElasticNet().fit(X, y_initial_slow_load)
        model_after_500_fiber_tear_1 = ElasticNet().fit(X, y_after_500_fiber_tear_1)
        model_after_500_fiber_tear_2 = ElasticNet().fit(X, y_after_500_fiber_tear_2)
        model_after_500_fast_load = ElasticNet().fit(X, y_after_500_fast_load)
        model_after_500_slow_load = ElasticNet().fit(X, y_after_500_slow_load)
    else:
        model_initial_fiber_tear_1 = KNeighborsRegressor().fit(X, y_initial_fiber_tear_1)
        model_initial_fiber_tear_2 = KNeighborsRegressor().fit(X, y_initial_fiber_tear_2)
        model_initial_fast_load = KNeighborsRegressor().fit(X, y_initial_fast_load)
        model_initial_slow_load = KNeighborsRegressor().fit(X, y_initial_slow_load)
        model_after_500_fiber_tear_1 = KNeighborsRegressor().fit(X, y_after_500_fiber_tear_1)
        model_after_500_fiber_tear_2 = KNeighborsRegressor().fit(X, y_after_500_fiber_tear_2)
        model_after_500_fast_load = KNeighborsRegressor().fit(X, y_after_500_fast_load)
        model_after_500_slow_load = KNeighborsRegressor().fit(X, y_after_500_slow_load)

    # Define prediction function
    def predict_properties(a, b, c, d):
        input_data = pd.DataFrame([[a, b, c, d]], columns=['Ingredient A_1', 'Ingredient B_2', 'Ingredient C_3', 'Ingredient D_4'])
        initial_fiber_tear_1 = model_initial_fiber_tear_1.predict(input_data)[0]
        initial_fiber_tear_2 = model_initial_fiber_tear_2.predict(input_data)[0]
        initial_fast_load = model_initial_fast_load.predict(input_data)[0]
        initial_slow_load = model_initial_slow_load.predict(input_data)[0]
        after_500_fiber_tear_1 = model_after_500_fiber_tear_1.predict(input_data)[0]
        after_500_fiber_tear_2 = model_after_500_fiber_tear_2.predict(input_data)[0]
        after_500_fast_load = model_after_500_fast_load.predict(input_data)[0]
        after_500_slow_load = model_after_500_slow_load.predict(input_data)[0]
        return (initial_fiber_tear_1, initial_fiber_tear_2, initial_fast_load, initial_slow_load, 
                after_500_fiber_tear_1, after_500_fiber_tear_2, after_500_fast_load, after_500_slow_load)

    # Predict properties
    initial_fiber_tear_1, initial_fiber_tear_2, initial_fast_load, initial_slow_load, \
    after_500_fiber_tear_1, after_500_fiber_tear_2, after_500_fast_load, after_500_slow_load = predict_properties(component_a, component_b, component_c, component_d)

    # Display predicted properties with gauge charts
    st.subheader("Predicted Properties", anchor='predicted_properties')

    def create_gauge_chart(value, title, threshold, max_value, critical_value):
        color = "darkgreen" if (critical_value <= value <= max_value) else "red"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={'font': {'color': color}},
            title={'text': title},
            gauge={
                'axis': {'range': [0, max_value]},
                'bar': {'color': color},
                'steps': [],
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': critical_value
                }
            }
        ))
        fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), height=150)  # Adjust height here
        return fig

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(create_gauge_chart(initial_fiber_tear_1, "Initial Fiber Tear #1", 80, 100, 80), use_container_width=True, config={'displayModeBar': False})
    with col2:
        st.plotly_chart(create_gauge_chart(after_500_fiber_tear_1, "Fiber Tear After 500 hrs #1", 80, 100, 80), use_container_width=True, config={'displayModeBar': False})
    with col3:
        st.plotly_chart(create_gauge_chart(initial_fiber_tear_1, "Fiber Tear After 1000 hrs #1", 80, 100, 80), use_container_width=True, config={'displayModeBar': False})

    col4, col5, col6 = st.columns(3)
    with col4:
        st.plotly_chart(create_gauge_chart(initial_fiber_tear_2, "Initial Fiber Tear #2", 80, 100, 80), use_container_width=True, config={'displayModeBar': False})
    with col5:
        st.plotly_chart(create_gauge_chart(after_500_fiber_tear_2, "Fiber Tear After 500 hrs #2", 80, 100, 80), use_container_width=True, config={'displayModeBar': False})
    with col6:
        st.plotly_chart(create_gauge_chart(initial_fiber_tear_2, "Fiber Tear After 1000 hrs #2", 80, 100, 80), use_container_width=True, config={'displayModeBar': False})

    col7, col8, col9 = st.columns(3)
    with col7:
        st.plotly_chart(create_gauge_chart(initial_fast_load, "Initial Fast Load", 4.25, 5, 4.25), use_container_width=True, config={'displayModeBar': False})
    with col8:
        st.plotly_chart(create_gauge_chart(after_500_fast_load, "Fast Load After 500 hrs", 4, 5, 4), use_container_width=True, config={'displayModeBar': False})
    with col9:
        st.plotly_chart(create_gauge_chart(initial_fast_load, "Fast Load After 1000 hrs", 4, 5, 4), use_container_width=True, config={'displayModeBar': False})

    # Define optimization function
    def objective_function(x):
        a, b, c, d = x
        initial_fiber_tear_1, initial_fiber_tear_2, initial_fast_load, initial_slow_load, \
        after_500_fiber_tear_1, after_500_fiber_tear_2, after_500_fast_load, after_500_slow_load = predict_properties(a, b, c, d)
        return -(initial_fast_load + after_500_fast_load) + (initial_fiber_tear_1 + initial_fiber_tear_2 + after_500_fiber_tear_1 + after_500_fiber_tear_2)

    # Optimize composition
    if st.button("Optimize Composition", key='optimize_composition'):
        bounds = [(0, 3), (0, 3), (0, 3), (0, 3)]
        initial_guess = [1.5, 1.5, 1.5, 1.5]
        result = minimize(objective_function, initial_guess, bounds=bounds)
        optimal_a, optimal_b, optimal_c, optimal_d = result.x
        st.markdown(
            f"<div class='gpt-response'>Optimal Composition: Ingredient A: {optimal_a:.2f}, Ingredient B: {optimal_b:.2f}, Ingredient C: {optimal_c:.2f}, Ingredient D: {optimal_d:.2f}</div>",
            unsafe_allow_html=True
        )
        
        # Suggesting a range
        st.markdown(
            f"<div class='gpt-response'>Optimal Range: Ingredient A: {max(0, optimal_a-0.1):.2f} - {min(3, optimal_a+0.1):.2f}, "
            f"Ingredient B: {max(0, optimal_b-0.1):.2f} - {min(3, optimal_b+0.1):.2f}, "
            f"Ingredient C: {max(0, optimal_c-0.1):.2f} - {min(3, optimal_c+0.1):.2f}, "
            f"Ingredient D: {max(0, optimal_d-0.1):.2f} - {min(3, optimal_d+0.1):.2f}</div>",
            unsafe_allow_html=True
        )

        # Highlight the location of the solution in the latent space
        latent_space = pd.DataFrame(np.random.rand(100, 3), columns=['X', 'Y', 'Z'])  # Dummy latent space data
        optimal_location_fig = px.scatter_3d(latent_space, x='X', y='Y', z='Z', title="Latent Space with Optimal Solution")
        optimal_location_fig.add_trace(go.Scatter3d(x=[optimal_a], y=[optimal_b], z=[optimal_c], mode='markers', marker=dict(size=5, color='red')))
        st.plotly_chart(optimal_location_fig, use_container_width=True)

    # Define task addition function
    def add_task(task_name, duration):
        task_start = pd.to_datetime('today')
        task_finish = task_start + pd.Timedelta(days=duration)
        st.session_state.tasks.append({'Task': task_name, 'Start': task_start, 'Finish': task_finish})

    # Initialize session state
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []

    # Sidebar for adding tasks
    st.sidebar.markdown("<h2 class='sidebar-title reduce-space'>Add Tasks</h2>", unsafe_allow_html=True)
    if st.sidebar.button('Add Experiment'):
        add_task('New Experiment', 2)

    if st.sidebar.button('Add Aging Study'):
        add_task('New Aging Study', 42)

    # Display schedule data
    schedule_data = pd.DataFrame(st.session_state.tasks)

    if not schedule_data.empty:
        project_start = pd.to_datetime('today')
        project_end = project_start + pd.Timedelta(weeks=8)
        total_duration = (schedule_data['Finish'] - schedule_data['Start']).sum()
        if total_duration > pd.Timedelta(weeks=8):
            st.sidebar.error("Project duration exceeds the maximum allowed 8 months")
        elif total_duration < pd.Timedelta(weeks=6):
            st.sidebar.warning("Project duration is less than the minimum required 6 months")

        fig_gantt = px.timeline(schedule_data, x_start='Start', x_end='Finish', y='Task')

        st.subheader("Experiment Scheduling", anchor='experiment_scheduling')
        st.plotly_chart(fig_gantt, use_container_width=True)
    else:
        st.write("No tasks added yet. Use the buttons on the sidebar to add tasks.")

else:
    st.write("Please upload an Excel file to proceed.")

# Footer
st.markdown(
    """
    <div class="footer">
        © Yuan Pang All rights reserved<br>
        Dashboard for Henkel interview demonstration ONLY<br>
        Version 0.1.4 <br>
        2024 August 05 <br>
    </div>
    """,
    unsafe_allow_html=True
)
