import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import sys
import numpy as np

'''
Plot learning curve for a couple different lambda values in a given environment.
Saves the plot as PNG and SVG files.
Input:
    - Command line argument 1: environment name (str)
'''

# Recover env name from command line argument
env_name = sys.argv[1]

# Load dataframe
return_per_timestep_for_each_lambda = pd.read_csv(f'runs/analyzed_data/average_return_per_timestep_for_each_lambda_{env_name}.csv')

# Retrieve data
_ = return_per_timestep_for_each_lambda.iloc[-1]
return_per_timestep_for_each_lambda = return_per_timestep_for_each_lambda.iloc[:-1]

# Lambdas to plot
lambdas_to_plot = ['0.0', '0.5', '0.9', '0.95', '0.99', '1.0']

# Plot learning curves
fig = px.line(title=f'Learning Curves for different λ in {env_name}', labels={'index': 'Timesteps', 'value': 'Episodic Return', 'variable': 'λ value'})
for labas in lambdas_to_plot:
    fig.add_scatter(x=return_per_timestep_for_each_lambda.index, y=return_per_timestep_for_each_lambda[labas], mode='lines', name=f'λ={labas}')
fig.update_layout(width=2000, height=800)
fig.update_xaxes(dtick=50_000)
# Font size
fig.update_layout(
    title_font_size=30,
    xaxis_title_font_size=25,
    yaxis_title_font_size=25,
    legend_font_size=20,
    xaxis=dict(tickfont=dict(size=20)),
    yaxis=dict(tickfont=dict(size=25))
)   
fig.write_image(f'runs/analyzed_data/Learning_curves_different_lambdas_{env_name}.png')
fig.write_image(f'runs/analyzed_data/Learning_curves_different_lambdas_{env_name}.svg')