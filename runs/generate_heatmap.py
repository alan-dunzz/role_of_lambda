import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

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

base_path = fr"C:\Users\aland\Desktop\University of Alberta\CMPUT 655 Reinforcement Learning\Project\code\AUC_anneal_lr_"

def sci_notation_formatter(x):
    if x == 0: return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / (10**exponent)
    
    return r"$%.0f \times 10^{%d}$" % (coeff, exponent)

def heatmap(env,cmap_v):
    for scheme in ['True','False']:
        path = base_path + scheme + fr'_{env}.csv'
    
        df = pd.read_csv(path,index_col=0)
        df.columns = [float(col.split("_")[1]) for col in df.columns]
        df.index = [float(idx.split("_")[1]) for idx in df.index]

        df = df.sort_index(ascending=False)
        df.index = df.index.map(lambda x: sci_notation_formatter(x))

        min_v = 10  * math.floor(np.min(df) / 10)
        max_v = 10 * math.ceil(np.max(df) / 10)

        plt.figure(figsize=(8.3, 7))

        plt.imshow(df, aspect='equal', vmin=min_v, vmax=max_v,cmap=cmap_v)
        plt.colorbar(label="AUC",fraction=0.046, pad=0.04, ticks = np.linspace(min_v,max_v,5, dtype=int))

        plt.xticks(range(len(df.columns)), df.columns, rotation=0)
        plt.yticks(range(len(df.index)), df.index)

        plt.xlabel(r"$\lambda$",labelpad=10)
        plt.ylabel(r"$\alpha$",rotation=0,labelpad=10)

        plt.title(fr"{env['env_name']}",pad = 15)

        plt.tight_layout()

        plt.savefig(fr"C:\Users\aland\Desktop\University of Alberta\CMPUT 655 Reinforcement Learning\Project\code\Heatmaps\AUC_anneal_{scheme}_{env}_{cmap_v}.png") 
        plt.savefig(fr"C:\Users\aland\Desktop\University of Alberta\CMPUT 655 Reinforcement Learning\Project\code\Heatmaps\AUC_anneal_{scheme}_{env}.svg") 
        plt.close()

for env in ['Acrobot-v1','CartPole-v1']:
    heatmap(env,'GnBu')

