import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import sys
import numpy as np

# Recover env name from command line argument
env_name = sys.argv[1]

# Load dataframe
return_per_timestep_for_each_lambda = pd.read_csv(f'runs/analysed_data/average_return_per_timestep_for_each_lambda_{env_name}.csv')

# Retrieve data
confidence_intervals = return_per_timestep_for_each_lambda.iloc[-1]
return_per_timestep_for_each_lambda = return_per_timestep_for_each_lambda.iloc[:-1]

# Calculate AUC for each lambda
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

fig.update_layout(title=f'Area under the curve (AUC) in {env_name} (Average +- CI 95%)', xaxis_title='λ value', yaxis_title='Area Under the Curve (AUC)')
fig.update_traces(line=dict(color='blue'), width=2000, height=800)
fig.update_xaxes(dtick=0.1)

# Save plot
fig.write_image(f'runs/analysed_data/AUC_vs_lambda_{env_name}.png')