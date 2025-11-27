import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import numpy as np

'''
Plot Area Under the Curve (AUC) vs lambda for a given environment.
Saves the plot as PNG and SVG files.
Input:
    - Command line argument 1: environment name (str)
    - (Optional) Command line argument 2: percentage of timesteps to sum up to (float between 0 and 1)
'''

# Recover env name from command line argument
env_name = sys.argv[1]

# Load dataframe
return_per_timestep_for_each_lambda = pd.read_csv(f'runs/analyzed_data/average_return_per_timestep_for_each_lambda_{env_name}.csv')
convergence_info = pd.read_csv(f'runs/analyzed_data/convergence_info_{env_name}.csv')

# Retrieve data
confidence_intervals = return_per_timestep_for_each_lambda.iloc[-1]
return_per_timestep_for_each_lambda = return_per_timestep_for_each_lambda.iloc[:-1]

# Calculate AUC for each lambda
# Check for a arg, if there is one, sum only up to that percentage of timesteps
if len(sys.argv) > 2:
    percentage_to_sum = sys.argv[2]
    sum_up_to_idx = int(return_per_timestep_for_each_lambda.shape[0] * float(percentage_to_sum))
    y = return_per_timestep_for_each_lambda.iloc[:sum_up_to_idx].mean(axis=0)
else:
    y = return_per_timestep_for_each_lambda.mean(axis=0)

# Plot AUC vs lambda
x = np.array(return_per_timestep_for_each_lambda.columns, dtype=np.float32)
fig = px.line(x=x, y=y)

# Add confidence intervals as shaded area
lower_bound = y - confidence_intervals
upper_bound = y + confidence_intervals
fig.add_traces([
    px.scatter(x=x, y=lower_bound).update_traces(mode='lines', line=dict(color='lightgrey'), showlegend=False).data[0],
    px.scatter(x=x, y=upper_bound).update_traces(mode='lines', line=dict(color='lightgrey'), fill='tonexty', fillcolor='rgba(211,211,211,0.5)', showlegend=False).data[0]
])

fig.update_layout(title=f'Area under the curve (AUC) in {env_name} (Average ± CI 95%)', xaxis_title='λ value', yaxis_title='Area Under the Curve (AUC)')
fig.update_traces(line=dict(color='blue'))
fig.update_layout(width=2000, height=800)
fig.update_xaxes(dtick=0.1)

x_conv = convergence_info['lambda']
y_conv = convergence_info['convergence_value_mean']
ci_conv = convergence_info['convergence_value_ci95']

# Calculate Bounds
lower_conv = y_conv - ci_conv
upper_conv = y_conv + ci_conv

# Convergence Confidence Interval (Shading - Red/Grey)
fig.add_traces([
    # Lower Bound (used as a reference for filling)
    px.scatter(x=x_conv, y=lower_conv).update_traces(mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip').data[0],
    # Upper Bound (Fills down to the previous trace/lower bound)
    px.scatter(x=x_conv, y=upper_bound).update_traces(mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', showlegend=False, hoverinfo='skip').data[0]
])

# Main Convergence Line
fig.add_trace(go.Scatter(
    x=x_conv, 
    y=y_conv, 
    mode='lines', 
    name='Convergence Value (Metric 2)',
    line=dict(color='red', width=3) # Use a distinct color
))
fig.update_traces(line=dict(color='blue'))
fig.update_layout(width=2000, height=800)
fig.update_xaxes(dtick=0.1)

# Font size
fig.update_layout(
    title_font_size=30,
    xaxis_title_font_size=25,
    yaxis_title_font_size=25,
    legend_font_size=20,
    xaxis=dict(tickfont=dict(size=20)),
    yaxis=dict(tickfont=dict(size=25))
)

# Save plot
fig.write_image(f'runs/analyzed_data/unified_AUC{env_name}.svg')
fig.write_image(f'runs/analyzed_data/unified_AUC{env_name}.png')