import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import sys
import numpy as np

'''
Plot Convergence Value for a given environment.
Saves the plot as PNG and SVG files.
Input:
    - Command line argument 1: environment name (str)
'''

# Recover env name from command line argument
env_name = sys.argv[1]

##############################################################################################################################################
# Calculate convergence info
# Load convergence info and plot convergence value vs lambda
convergence_info = pd.read_csv(f'runs/analyzed_data/convergence_info_{env_name}.csv')
fig2 = px.line(convergence_info, x='lambda', y='convergence_value_mean', title=f'Convergence Value per λ in {env_name}', labels={'lambda': 'λ value', 'convergence_value_mean': 'Convergence Value (Mean +- CI 95%)'})
# Add confidence intervals as shaded area
lower_bound = convergence_info['convergence_value_mean'] - convergence_info['convergence_value_ci95']
upper_bound = convergence_info['convergence_value_mean'] + convergence_info['convergence_value_ci95']
fig2.add_traces([
    px.scatter(x=convergence_info['lambda'], y=lower_bound).update_traces(mode='lines', line=dict(color='lightgrey'), showlegend=False).data[0],
    px.scatter(x=convergence_info['lambda'], y=upper_bound).update_traces(mode='lines', line=dict(color='lightgrey'), fill='tonexty', fillcolor='rgba(211,211,211,0.5)', showlegend=False).data[0]
])
fig2.update_traces(line=dict(color='blue'))
fig2.update_layout(width=2000, height=800)
fig2.update_xaxes(dtick=0.1)
# Font size
fig2.update_layout(
    title_font_size=30,
    xaxis_title_font_size=25,
    yaxis_title_font_size=25,
    legend_font_size=20,
    xaxis=dict(tickfont=dict(size=20)),
    yaxis=dict(tickfont=dict(size=25))
)

fig2.write_image(f'runs/analyzed_data/Convergence_value_vs_lambda_{env_name}.png')
fig2.write_image(f'runs/analyzed_data/Convergence_value_vs_lambda_{env_name}.svg')