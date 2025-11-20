import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import sys
import numpy as np

# Configs
window_size = 2000
theshold = 450

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
    convergence_timestep = np.where(running_average_series >= theshold)[0]
    if len(convergence_timestep) > 0:
        convergence_speeds.append([float(lambda_value), convergence_timestep[0]])
    else:
        convergence_speeds.append([float(lambda_value), np.nan])

# Convert to DataFrame
convergence_speeds = pd.DataFrame(convergence_speeds, columns=['lambda', 'convergence_timestep'])

# Save plot convergence speed vs lambda
fig = px.line(convergence_speeds, x='lambda', y='convergence_timestep', title=f'Convergence Speed per λ in {env_name}', labels={'lambda': 'λ value', 'convergence_timestep': 'Steps to Convergence'})
fig.update_xaxes(dtick=0.1)
fig.update_traces(width=2000, height=800)
fig.write_image(f'runs/analysed_data/Convergence_speed_vs_lambda_{env_name}.png')