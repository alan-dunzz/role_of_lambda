import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import sys
import numpy as np

# Recover env name from command line argument
env_name = sys.argv[1]

# Load dataframe
return_per_timestep_for_each_lambda = pd.read_csv(f'runs/analysed_data/average_return_per_timestep_for_each_lambda_{env_name}.csv')

# Retrieve data. Last row is CI, so we ignore it for AUC calculation
confidence_intervals = return_per_timestep_for_each_lambda.iloc[-1]
return_per_timestep_for_each_lambda = return_per_timestep_for_each_lambda.iloc[:-1]


# Calculate AUC for each lambda
y = return_per_timestep_for_each_lambda.mean(axis=0)

# Plot AUC vs lambda with confidence intervals
x = np.array(return_per_timestep_for_each_lambda.columns, dtype=np.float32)
fig = px.line(x=x, y=y, error_y=confidence_intervals, markers=True)

fig.update_layout(title=f'Area under curve (AUC) in {env_name} (Average +- CI 95%)', xaxis_title='λ value', yaxis_title='Area Under Curve (AUC)')

# Save plot
fig.write_image(f'runs/analysed_data/AUC_vs_lambda_{env_name}.png')