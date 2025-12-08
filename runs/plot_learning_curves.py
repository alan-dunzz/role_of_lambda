import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------------------
# Formatting for plot
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
# ---------------------------

current_directory = os.getcwd()
csv_paths = {
    "CartPole-v1": fr"{current_directory}/runs/analyzed_data/average_return_for_plotting_CartPole-v1.csv",
    "Acrobot-v1": fr"{current_directory}/runs/analyzed_data/average_return_for_plotting_Acrobot-v1.csv"
}

lambdas = [0, 0.5, 0.9, 0.95, 0.99, 1]
cmap = plt.cm.Blues
colors = [cmap(i) for i in np.linspace(0.3, 0.9, len(lambdas))]
window = 750  # rolling average

# Create figure with 2 subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

for ax, (env_name, csv_path) in zip(axes, csv_paths.items()):
    # Load CSV
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["timestep", 0, 0.5, 0.9, 0.95, 0.99, 1]

    # Plot each λ
    for lam, color in zip(lambdas, colors):
        smoothed = df[lam].rolling(window=window, min_periods=1).mean()
        ax.plot(df["timestep"], smoothed, label=rf"$\lambda = {lam}$", color=color, linewidth=2)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Labels and title
    ax.set_xlabel("Timesteps")
    ax.set_title(f"Learning Curves for {env_name}", pad=15)

    # Format x-axis as 25k, 50k, ...
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

    # Legend from max → min λ
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    # Grid
    ax.grid(True, alpha=0.3)

# Y-label only on the left subplot
axes[0].set_ylabel("Episodic Return")

# Adjust spacing between subplots
plt.tight_layout()

# Save figure
plt.savefig(fr"{current_directory}/runs/analyzed_data/learning_curves_two_envs.png", dpi=300, bbox_inches="tight")
plt.show()
