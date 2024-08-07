import pandas as pd
from datetime import datetime, timedelta
import plotly.figure_factory as ff

def create_gantt_chart(df_full_factorial_design, start_date):
    tasks = []
    current_date = start_date
    days_for_formulation = 2
    days_for_curing = 3
    weeks_for_aging = 6
    max_experiments_per_monday = 5

    experiment_count = len(df_full_factorial_design)
    task_id = 0

    for week in range(0, experiment_count, max_experiments_per_monday):
        monday_date = current_date + timedelta(weeks=week // max_experiments_per_monday)

        for i in range(min(max_experiments_per_monday, experiment_count - week)):
            task_id += 1
            task = {
                'Task': f'Experiment {task_id}',
                'Start': monday_date.strftime("%Y-%m-%d"),
                'Finish': (monday_date + timedelta(days=days_for_formulation)).strftime("%Y-%m-%d"),
                'Resource': 'Formulation Development'
            }
            tasks.append(task)

            task = {
                'Task': f'Experiment {task_id}',
                'Start': (monday_date + timedelta(days=days_for_formulation)).strftime("%Y-%m-%d"),
                'Finish': (monday_date + timedelta(days=days_for_formulation + days_for_curing)).strftime("%Y-%m-%d"),
                'Resource': 'Curing'
            }
            tasks.append(task)

            task = {
                'Task': f'Experiment {task_id}',
                'Start': (monday_date + timedelta(days=days_for_formulation + days_for_curing)).strftime("%Y-%m-%d"),
                'Finish': (monday_date + timedelta(days=days_for_formulation + days_for_curing + weeks_for_aging * 7)).strftime("%Y-%m-%d"),
                'Resource': 'Aging'
            }
            tasks.append(task)

    colors = {
        'Formulation Development': 'rgb(255, 0, 0)',
        'Curing': 'rgb(255, 255, 0)',
        'Aging': 'rgb(0, 0, 255)'
    }

    fig = ff.create_gantt(tasks, index_col='Resource', show_colorbar=True, group_tasks=True, colors=colors)
    fig.update_layout(
        title='<span style="font-size:24px; color:#333;">Experimental Schedule Gantt Chart</span>',
        height=800,
        margin=dict(t=100, b=50),  # Adjusted to avoid overlap
        xaxis_title="Time",
        yaxis_title="Experiments",
        font=dict(size=12)
    )
    return fig
