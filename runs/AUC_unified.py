import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator

# 1. Configuration for Fonts (mimicking Computer Modern)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",  # Use Computer Modern for math (lambda)
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext":True,
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18
})

# Recover env name
env_name = 'Acrobot-v1'

#Load data
base_path = fr"C:\Users\aland\Desktop\University of Alberta\CMPUT 655 Reinforcement Learning\Project\code"
convergence_info = pd.read_csv(fr"{base_path}\convergence_data_{env_name}.csv")
early_learning_info = pd.read_csv(fr"{base_path}\early_learning_data_{env_name}.csv")

#Initialize Figure
fig, ax = plt.subplots(figsize=(10, 7), dpi=100)

# Early learning data
x_early = early_learning_info['lambda']
y_early_mean = early_learning_info['early_learning_value_mean']
y_early_lower = early_learning_info['early_learning_p5']
y_early_upper = early_learning_info['early_learning_p95']

# Shaded Area (Tolerance Interval)
ax.fill_between(x_early, y_early_lower, y_early_upper, 
                color='#e8b37d', alpha=0.4, label='_nolegend_')

# Mean Line
ax.plot(x_early, y_early_mean, 
        color='#e87d13', linewidth=4, label='Early learning')

# Convergence data
x_conv = convergence_info['lambda']
y_conv_mean = convergence_info['convergence_value_mean']
y_conv_lower = convergence_info['convergence_p5']
y_conv_upper = convergence_info['convergence_p95']

# Shaded Area (Tolerance Interval)
ax.fill_between(x_conv, y_conv_lower, y_conv_upper, 
                color='lightblue', alpha=0.4, label='_nolegend_')

# Mean Line
ax.plot(x_conv, y_conv_mean, 
        color='#2456a6', linewidth=4, label='Convergence')

# Title and Labels
ax.set_title(f"{env_name}", pad=15)
ax.set_xlabel(r"$\lambda$") # Using LaTeX for lambda
ax.set_ylabel("AUC",rotation=0,ha='right')

# Axis Ticks formatting
ax.tick_params(axis='both', which='major', width=2, length=6)

# Set grid
ax.grid(True, color='lightgray', linestyle='-', linewidth=1, alpha=0.8)
ax.set_axisbelow(True) # Ensure grid is behind the plot lines

# Enforce specific tick intervals (dtick)
ax.xaxis.set_major_locator(MultipleLocator(0.1)) 
ax.yaxis.set_major_locator(MultipleLocator(50))  
ax.set_xlim(0, 1)
# Legend Customization
legend = ax.legend(
    fontsize=16, 
    loc='upper left', 
    bbox_to_anchor=(0.65, 0.18),
    frameon=True,          
    edgecolor='lightgray', 
    facecolor='white',     
    framealpha=0.8         
)

legend.get_frame().set_linewidth(1.5)

for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(1)
    ax.spines[spine].set_color('black')

# Hide the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save files 
output_png = fr"{base_path}\unified_AUC_vs_lambda_{env_name}.png"
output_svg = fr"{base_path}\unified_AUC_vs_lambda_{env_name}.svg"

plt.tight_layout() # Adjust layout to prevent clipping
plt.savefig(output_png, dpi=100)
plt.savefig(output_svg, format='svg')

print("Plot generated successfully")
