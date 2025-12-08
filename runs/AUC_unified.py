import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator

# 1. Configuration for Fonts
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": True,
    "axes.formatter.use_mathtext": True,
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "svg.fonttype":'none',
})

# Environments to plot (Left -> Right)
envs = ['Cartpole-v1', 'Acrobot-v1']

# Base path (Update this if running on a different machine)
base_path = fr"C:\Users\aland\Desktop\University of Alberta\CMPUT 655 Reinforcement Learning\Project\code"

# Initialize Figure with 2 subplots (1 row, 2 columns)
# Increased width (18) to accommodate both plots side-by-side comfortably
fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=100)

# Lists to store handles/labels for the shared legend later
lines = []
labels = []

for i, env_name in enumerate(envs):
    ax = axes[i] # Select the current axis
    
    # Load data
    try:
        convergence_info = pd.read_csv(fr"{base_path}\convergence_data_{env_name}.csv")
        early_learning_info = pd.read_csv(fr"{base_path}\early_learning_data_{env_name}.csv")
    except FileNotFoundError:
        print(f"Files not found for {env_name}")

    # --- Early Learning Plotting ---
    x_early = early_learning_info['lambda']
    y_early_mean = early_learning_info['early_learning_value_mean']
    
    ax.fill_between(x_early, 
                    early_learning_info['early_learning_p5'], 
                    early_learning_info['early_learning_p95'], 
                    color='#e8b37d', alpha=0.4, label='_nolegend_')
    
    l1, = ax.plot(x_early, y_early_mean, 
            color='#e87d13', linewidth=4, label='Early learning')

    # --- Convergence Plotting ---
    x_conv = convergence_info['lambda']
    y_conv_mean = convergence_info['convergence_value_mean']
    
    ax.fill_between(x_conv, 
                    convergence_info['convergence_p5'], 
                    convergence_info['convergence_p95'], 
                    color='lightblue', alpha=0.4, label='_nolegend_')
    
    l2, = ax.plot(x_conv, y_conv_mean, 
            color='#2456a6', linewidth=4, label='Final performance')

    # Store handles for legend (only need to do this once)
    if i == 0:
        lines = [l1, l2]
        labels = [l1.get_label(), l2.get_label()]

    # --- Styling ---
    ax.set_title(f"{env_name}", pad=15)
    ax.set_xlabel("λ")
    ax.set_ylabel("AUC", rotation=0, ha='right')

    # Axis Ticks formatting
    ax.tick_params(axis='both', which='major', width=2, length=6)
    
    # Grid
    ax.grid(True, color='lightgray', linestyle='-', linewidth=1, alpha=0.8)
    ax.set_axisbelow(True)

    # Spines
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1)
        ax.spines[spine].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Limits and Locators
    ax.xaxis.set_major_locator(MultipleLocator(0.1)) 
    ax.yaxis.set_major_locator(MultipleLocator(100))  
    ax.set_xlim(0, 1)

# --- Shared Legend Configuration ---
# We place the legend on the Figure object, not the individual Axes
legend = fig.legend(
    lines, labels,
    loc='lower center',      # Position
    bbox_to_anchor=(0.5, 0), # Anchor point (center bottom)
    ncol=2,                  # Horizontal layout
    fontsize=18,
    frameon=True,
    edgecolor='lightgray',
    facecolor='white',
    framealpha=1
)
legend.get_frame().set_linewidth(1.5)

# Adjust layout to prevent legend overlap and clipping
# Increase 'bottom' so there is room for the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.2) 

# Save files 
output_png = fr"{base_path}\combined_AUC_vs_lambda.png"
output_svg = fr"{base_path}\combined_AUC_vs_lambda.svg"

plt.savefig(output_png, dpi=100)
plt.savefig(output_svg, format='svg')

#plt.show()
print("Combined plot generated successfully")