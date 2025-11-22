import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import sys
import numpy as np

'''
Plot Convergence Speed vs lambda, and Convergence Value for a given environment.
Saves the plot as PNG and SVG files.
Input:
    - Command line argument 1: environment name (str)
    - (Optional) Command line argument 2: percentage of timesteps to average from for the convergence value (float between 0 and 1)
'''


# Configs
window_size = 2000
threshold = 3

# Recover env name from command line argument
env_name = sys.argv[1]

# Load dataframe
return_per_timestep_for_each_lambda = pd.read_csv(f'runs/analysed_data/average_return_per_timestep_for_each_lambda_{env_name}.csv')

# Retrieve data
confidence_intervals = return_per_timestep_for_each_lambda.iloc[-1]
return_per_timestep_for_each_lambda = return_per_timestep_for_each_lambda.iloc[:-1] # DF with shape (timesteps, lambdas)

# Get running average
running_average_df = return_per_timestep_for_each_lambda.rolling(window=window_size, min_periods=1).mean()

# Get the timestep where running average crosses the threshold
convergence_speeds = []
for lambda_value in running_average_df.columns:
    running_average_series = running_average_df[lambda_value]

    if len(sys.argv) > 2:
        percentage_to_average = sys.argv[2]
        average_from_idx = int(return_per_timestep_for_each_lambda.shape[0] * float(percentage_to_average))
        final_average = np.mean(running_average_series[-average_from_idx:])
    else:
        final_average = np.mean(running_average_series[-10_000:])
    
    difference_to_final = np.abs(running_average_series - final_average)

    convergence_timestep = np.where(difference_to_final <= threshold)[0]

    if len(convergence_timestep) == 0:
        convergence_speeds.append([float(lambda_value), np.nan])
    else:
        convergence_speeds.append([float(lambda_value), convergence_timestep[0]])

# Convert to DataFrame
convergence_speeds = pd.DataFrame(convergence_speeds, columns=['lambda', 'convergence_timestep'])

# Save plot convergence speed vs lambda
fig = px.line(convergence_speeds, x='lambda', y='convergence_timestep', title=f'Convergence Speed per λ in {env_name}', labels={'lambda': 'λ value', 'convergence_timestep': 'Steps to Convergence'})
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

fig.write_image(f'runs/analysed_data/Convergence_speed_vs_lambda_{env_name}.png')
fig.write_image(f'runs/analysed_data/Convergence_speed_vs_lambda_{env_name}.svg')

##############################################################################################################################################
# Load convergence info and plot convergence value vs lambda
convergence_info = pd.read_csv(f'runs/analysed_data/convergence_info_{env_name}.csv')
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

fig2.write_image(f'runs/analysed_data/Convergence_value_vs_lambda_{env_name}.png')
fig2.write_image(f'runs/analysed_data/Convergence_value_vs_lambda_{env_name}.svg')