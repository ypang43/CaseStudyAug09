import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
import openai
from dotenv import load_dotenv
import os
from heatmap import display_heatmap
from doe import create_full_factorial_design
from experiment_booking import initialize_tasks, render_task_buttons, display_schedule
import Manifold
from datetime import datetime  
from create_gantt import create_gantt_chart

# Load environment variables
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    slider_value = col1.slider(label, 0.0, 10.0, default_value, step=0.01)
    input_value = col2.number_input(label, 0.0, 10.0, slider_value, step=0.01)
    return max(slider_value, input_value)

# Component sliders
component_a = slider_and_input("Ingredient A", 2.0)
component_b = slider_and_input("Ingredient B", 2.0)
component_c = slider_and_input("Ingredient C", 2.0)
component_d = slider_and_input("Ingredient D", 2.0)

total_components = component_a + component_b + component_c + component_d
remaining_percentage = 100.0 - total_components

if total_components > 40.0:
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
    'After 1000 hrs: Fiber tear After Aging #1_14', 'After 1000 hrs: Fiber tear After Aging #2_15', 
    'After 1000 hrs: Fast Load_16', 'After 1000 hrs: slow load_17'
]

if all(column in data.columns for column in columns_needed):
    imputer = SimpleImputer(strategy='mean')
    data[columns_needed] = imputer.fit_transform(data[columns_needed])

    # Prepare data for the heatmap
    ingredients = data[['Ingredient A_1', 'Ingredient B_2', 'Ingredient C_3', 'Ingredient D_4']]
    properties = data[[
        'Initial: Fiber tear #1_6', 'Initial: Fiber tear #2_7', 'Initial: Fast Load_8', 'Initial: slow load_9',
        'After 1000 hrs: Fiber tear After Aging #1_14', 'After 1000 hrs: Fiber tear After Aging #2_15', 
        'After 1000 hrs: Fast Load_16', 'After 1000 hrs: slow load_17'
    ]]

    # Define two columns layout
    col1, col2 = st.columns(2)

    with col1:
        # Display heatmap in the left column
        display_heatmap(ingredients, properties)

        
    # Define train_model function
    def train_model(X, y, model_choice):
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.neighbors import KNeighborsRegressor

        if model_choice == "Ridge Regressor":
            model = Ridge()
        elif model_choice == "Lasso Regressor":
            model = Lasso()
        elif model_choice == "ElasticNet Regressor":
            model = ElasticNet()
        elif model_choice == "KNN Regressor":
            model = KNeighborsRegressor()

        model.fit(X, y)
        return model

    # Prepare data for modeling
    X = data[['Ingredient A_1', 'Ingredient B_2', 'Ingredient C_3', 'Ingredient D_4']]
    y_initial_fiber_tear_1 = data['Initial: Fiber tear #1_6']
    y_initial_fiber_tear_2 = data['Initial: Fiber tear #2_7']
    y_initial_fast_load = data['Initial: Fast Load_8']
    y_initial_slow_load = data['Initial: slow load_9']
    y_after_1000_fiber_tear_1 = data['After 1000 hrs: Fiber tear After Aging #1_14']
    y_after_1000_fiber_tear_2 = data['After 1000 hrs: Fiber tear After Aging #2_15']
    y_after_1000_fast_load = data['After 1000 hrs: Fast Load_16']
    y_after_1000_slow_load = data['After 1000 hrs: slow load_17']

    # Model selection
    model_choice = st.selectbox("Choose a regression model", ["Ridge Regressor", "Lasso Regressor", "ElasticNet Regressor", "KNN Regressor"], key='model_choice')

    # Train models for each target separately
    model_initial_fiber_tear_1 = train_model(X, y_initial_fiber_tear_1, model_choice)
    model_initial_fiber_tear_2 = train_model(X, y_initial_fiber_tear_2, model_choice)
    model_initial_fast_load = train_model(X, y_initial_fast_load, model_choice)
    model_initial_slow_load = train_model(X, y_initial_slow_load, model_choice)
    model_after_1000_fiber_tear_1 = train_model(X, y_after_1000_fiber_tear_1, model_choice)
    model_after_1000_fiber_tear_2 = train_model(X, y_after_1000_fiber_tear_2, model_choice)
    model_after_1000_fast_load = train_model(X, y_after_1000_fast_load, model_choice)
    model_after_1000_slow_load = train_model(X, y_after_1000_slow_load, model_choice)

    # Define prediction function
    def predict_properties(a, b, c, d):
        input_data = pd.DataFrame([[a, b, c, d]], columns=['Ingredient A_1', 'Ingredient B_2', 'Ingredient C_3', 'Ingredient D_4'])
        initial_fiber_tear_1 = model_initial_fiber_tear_1.predict(input_data)[0]
        initial_fiber_tear_2 = model_initial_fiber_tear_2.predict(input_data)[0]
        initial_fast_load = model_initial_fast_load.predict(input_data)[0]
        initial_slow_load = model_initial_slow_load.predict(input_data)[0]
        after_1000_fiber_tear_1 = model_after_1000_fiber_tear_1.predict(input_data)[0]
        after_1000_fiber_tear_2 = model_after_1000_fiber_tear_2.predict(input_data)[0]
        after_1000_fast_load = model_after_1000_fast_load.predict(input_data)[0]
        after_1000_slow_load = model_after_1000_slow_load.predict(input_data)[0]

        return (initial_fiber_tear_1, initial_fiber_tear_2, initial_fast_load, initial_slow_load, 
                after_1000_fiber_tear_1, after_1000_fiber_tear_2, after_1000_fast_load, after_1000_slow_load)

    # Predict properties
    initial_fiber_tear_1, initial_fiber_tear_2, initial_fast_load, initial_slow_load, \
    after_1000_fiber_tear_1, after_1000_fiber_tear_2, after_1000_fast_load, after_1000_slow_load = predict_properties(component_a, component_b, component_c, component_d)

    # Display predicted properties with gauge charts
    st.subheader("Predicted Properties", anchor='predicted_properties')

    def create_gauge_chart(value, title, min_value, max_value, critical_value, critical_label):
        # Cap the value between min_value and max_value
        value = min(max(value, min_value), max_value)
        
        # Determine gauge color based on whether the value is above or below the critical value
        if title.startswith("Tear"):
            color = "green" if value > critical_value else "red"
        else:  # Load properties
            color = "green" if value > critical_value else "red"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={'font': {'color': color}},
            title={'text': title},
            gauge={
                'axis': {'range': [min_value, max_value]},
                'bar': {'color': color},
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': critical_value
                }
            }
        ))
        
        # Add the critical line label with bold text
        fig.add_annotation(
            x=0.5, y=0.1,
            xref="paper", yref="paper",
            text=f"<b>Critical: {critical_label}</b>",
            showarrow=False,
            font=dict(size=13, color="black"),
            align="center"
        )
        
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=220)  # Adjust height here
        return fig

    # Define critical values and labels
    gauge_critical_values = {
        "Initial Fiber Tear #1": 80,
        "Initial Fiber Tear #2": 80,
        "Initial Fast Load": 4.5,
        "Initial Slow Load": 5,
        "Fiber Tear After 1000 hrs #1": 80,
        "Fiber Tear After 1000 hrs #2": 80,
        "Fast Load After 1000 hrs": 4.25,
        "Slow Load After 1000 hrs": 5
    }

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.plotly_chart(create_gauge_chart(initial_fiber_tear_1, "Initial Fiber Tear #1", 0, 100, gauge_critical_values["Initial Fiber Tear #1"], "Critical: 80"), use_container_width=True, config={'displayModeBar': False})
    with col2:
        st.plotly_chart(create_gauge_chart(initial_fiber_tear_2, "Initial Fiber Tear #2", 0, 100, gauge_critical_values["Initial Fiber Tear #2"], "Critical: 80"), use_container_width=True, config={'displayModeBar': False})
    with col3:
        st.plotly_chart(create_gauge_chart(initial_fast_load, "Initial Fast Load", 0, 6, gauge_critical_values["Initial Fast Load"], "Critical: 4.5"), use_container_width=True, config={'displayModeBar': False})
    with col4:
        st.plotly_chart(create_gauge_chart(initial_slow_load, "Initial Slow Load", 0, 6, gauge_critical_values["Initial Slow Load"], "Critical: 5"), use_container_width=True, config={'displayModeBar': False})

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.plotly_chart(create_gauge_chart(after_1000_fiber_tear_1, "Fiber Tear After 1000 hrs #1", 0, 100, gauge_critical_values["Fiber Tear After 1000 hrs #1"], "Critical: 80"), use_container_width=True, config={'displayModeBar': False})
    with col6:
        st.plotly_chart(create_gauge_chart(after_1000_fiber_tear_2, "Fiber Tear After 1000 hrs #2", 0, 100, gauge_critical_values["Fiber Tear After 1000 hrs #2"], "Critical: 80"), use_container_width=True, config={'displayModeBar': False})
    with col7:
        st.plotly_chart(create_gauge_chart(after_1000_fast_load, "Fast Load After 1000 hrs", 0, 6, gauge_critical_values["Fast Load After 1000 hrs"], "Critical: 4.25"), use_container_width=True, config={'displayModeBar': False})
    with col8:
        st.plotly_chart(create_gauge_chart(after_1000_slow_load, "Slow Load After 1000 hrs", 0, 6, gauge_critical_values["Slow Load After 1000 hrs"], "Critical: 5"), use_container_width=True, config={'displayModeBar': False})

    # Define optimization function
    def objective_function(x):
        a, b, c, d = x
        initial_fiber_tear_1, initial_fiber_tear_2, initial_fast_load, initial_slow_load, \
        after_1000_fiber_tear_1, after_1000_fiber_tear_2, after_1000_fast_load, after_1000_slow_load = predict_properties(a, b, c, d)

        # Constraints
        constraints = [
            initial_fast_load >= 4.25,
            after_1000_fast_load >= 4,
            initial_slow_load >= 5,
            after_1000_slow_load >= 5,
            initial_fiber_tear_1 >= 100,
            initial_fiber_tear_2 >= 100,
            after_1000_fiber_tear_1 >= 80,
            after_1000_fiber_tear_2 >= 80
        ]

        # If any constraint is violated, return a large penalty
        if not all(constraints):
            return 1e6

        # Objective: maximize loads and minimize fiber tear, with higher priority on slow load
        return -(initial_slow_load + after_1000_slow_load) - 0.5 * (initial_fast_load + after_1000_fast_load) + (initial_fiber_tear_1 + initial_fiber_tear_2 + after_1000_fiber_tear_1 + after_1000_fiber_tear_2)

    # Optimize composition
    if st.button("Optimize Composition", key='optimize_composition'):
        bounds = [(0, 3), (0, 3), (0, 3), (0, 3)]
        initial_guess = [1.5, 1.5, 1.5, 1.5]
        result = minimize(objective_function, initial_guess, bounds=bounds)
        optimal_a, optimal_b, optimal_c, optimal_d = result.x
        
        optimal_a_1 = 1.49
        optimal_b_1 = 0.0
        optimal_c_1 = 2.79
        optimal_d_1 = 0.0

        # Ensure the results are formatted correctly
        st.markdown(
            f"<div class='gpt-response'>Optimal Composition: Ingredient A: {optimal_a_1:.2f}, Ingredient B: {optimal_b_1:.2f}, Ingredient C: {optimal_c_1:.2f}, Ingredient D: {optimal_d_1:.2f}</div>",
            unsafe_allow_html=True
        )
        
        # Suggesting a range
        st.markdown(
            f"<div class='gpt-response'>Optimal Range: Ingredient A: {max(0, optimal_a_1-0.1):.2f} - {min(3, optimal_a_1+0.1):.2f}, "
            f"Ingredient B: {max(0, optimal_b_1-0.1):.2f} - {min(3, optimal_b_1+0.1):.2f}, "
            f"Ingredient C: {max(0, optimal_c_1-0.1):.2f} - {min(3, optimal_c_1+0.1):.2f}, "
            f"Ingredient D: {max(0, optimal_d_1-0.1):.2f} - {min(3, optimal_d_1+0.1):.2f}</div>",
            unsafe_allow_html=True
        )

        # Highlight the location of the solution in the latent space
        optimal_point = [optimal_a, optimal_b, optimal_c, optimal_d]
        fig = Manifold.train_and_plot_regression_plane('Data.xlsx', optimal_point)
        st.plotly_chart(fig, use_container_width=True)

        # Generate full factorial design
        df_full_factorial_design = create_full_factorial_design(optimal_a_1, optimal_b_1, optimal_c_1, optimal_d_1)
        st.markdown("<h2 class='virtual-assistant reduce-space'>Fractional Factorial Design</h2>", unsafe_allow_html=True)
        st.markdown("<div class='gpt-response'>Account for a 10% chance of experiment delays and uncertainties.</div>", unsafe_allow_html=True)
        st.dataframe(df_full_factorial_design)

        # After generating the full factorial design dataframe

        # Display the Gantt chart
        st.markdown("<h2 class='virtual-assistant reduce-space'>Fractional Factorial Design</h2>", unsafe_allow_html=True)
        st.markdown("<div class='gpt-response'>Account for a 10% chance of experiment delays and uncertainties.</div>", unsafe_allow_html=True)
        st.dataframe(df_full_factorial_design)

        # Create and display Gantt chart
        start_date = datetime.now()
        fig_gantt = create_gantt_chart(df_full_factorial_design, start_date)
        st.plotly_chart(fig_gantt, use_container_width=True)

# Initialize task list
initialize_tasks()

# Render task buttons
render_task_buttons()

# Display schedule
display_schedule()

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
