import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def initialize_tasks():
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []

def add_task(task_name, duration_days, start_date):
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []

    task_start_datetime = datetime.combine(start_date, datetime.min.time())
    task_end_datetime = task_start_datetime + timedelta(days=duration_days)
    
    task = {'Task': task_name, 'Start': task_start_datetime, 'Finish': task_end_datetime}
    st.session_state.tasks.append(task)
    
    st.success(f"Added {task_name} from {task_start_datetime.date()} to {task_end_datetime.date()}")

def render_task_buttons():
    st.sidebar.markdown("<h2 class='sidebar-title reduce-space'>Add Tasks</h2>", unsafe_allow_html=True)
    
    start_date = st.sidebar.date_input("Select Start Date", value=datetime.today())
    
    if st.sidebar.button('Add Formulation Development (2 days)'):
        add_task('Formulation Development', 2, start_date)
    
    if st.sidebar.button('Add Curing (3 days)'):
        add_task('Curing', 3, start_date)
    
    if st.sidebar.button('Add Aging (6 weeks)'):
        add_task('Aging', 42, start_date)

def display_schedule():
    if 'tasks' in st.session_state and st.session_state.tasks:
        schedule_data = pd.DataFrame(st.session_state.tasks)
        
        # Define colors for each task type
        colors = {
            'Formulation Development': 'red',
            'Curing': 'orange',
            'Aging': 'blue'
        }
        
        fig_gantt = px.timeline(schedule_data, x_start='Start', x_end='Finish', y='Task', title="Experiment Scheduling", color='Task')
        fig_gantt.update_traces(marker=dict(color=[colors.get(task) for task in schedule_data['Task']]))
        fig_gantt.update_yaxes(categoryorder="total ascending")
        
        st.plotly_chart(fig_gantt, use_container_width=True)
    else:
        st.write("No tasks added yet. Use the buttons to add tasks.")
