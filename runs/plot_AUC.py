import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import sys
import numpy as np

# Recover env name from command line argument
env_name = sys.argv[1]

# Load dataframe
return_per_timestep_for_each_lambda = pd.read_csv(f'runs/analysed_data/average_return_per_timestep_for_each_lambda_{env_name}.csv')

# Calculate AUC for each lambda
y = return_per_timestep_for_each_lambda.mean(axis=0)

# Plot AUC vs lambda
x = np.array(return_per_timestep_for_each_lambda.columns, dtype=np.float32)
fig = px.line(x=x, y=y)
fig.update_layout(title=f'Area under curve (AUC) in {env_name}', xaxis_title='λ value', yaxis_title='Area Under Curve (AUC)')

# Save plot
fig.write_image(f'runs/analysed_data/AUC_vs_lambda_{env_name}.png')