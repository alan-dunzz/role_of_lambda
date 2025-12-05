import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

Acrobot = {'env_name' : 'Acrobot-v1','min' :-500, 'max' : -100}
Cartpole = {'env_name' : 'CartPole-v1','min' :0, 'max' : 500}

base_path = fr"C:\Users\aland\Desktop\University of Alberta\CMPUT 655 Reinforcement Learning\Project\code\AUC_anneal_lr_"

def heatmap(env,cmap_v):
    for scheme in ['True','False']:
        path = base_path + scheme + fr'_{env['env_name']}.csv'
    
        df = pd.read_csv(path,index_col=0)
        df.columns = [float(col.split("_")[1]) for col in df.columns]
        df.index = [float(idx.split("_")[1]) for idx in df.index]

        df = df.sort_index(ascending=False)
        df.index = df.index.map(lambda x: f'{x:.1e}')
        plt.figure(figsize=(8.3, 7))

        plt.imshow(df, aspect='equal', vmin=env['min'], vmax=env['max'],cmap=cmap_v)
        plt.colorbar(label="AUC",fraction=0.046, pad=0.04, ticks = np.linspace(env['min'],env['max'],5))

        plt.xticks(range(len(df.columns)), df.columns, rotation=0)
        plt.yticks(range(len(df.index)), df.index)

        plt.xlabel(r"$\lambda$",labelpad=10)
        plt.ylabel(r"$\alpha$",rotation=0,labelpad=10)

        plt.title(fr"{env['env_name']}",pad = 15)

        plt.tight_layout()

        plt.savefig(fr"C:\Users\aland\Desktop\University of Alberta\CMPUT 655 Reinforcement Learning\Project\code\Heatmaps\AUC_anneal_{scheme}_{env['env_name']}_{cmap_v}.png") 
        plt.savefig(fr"C:\Users\aland\Desktop\University of Alberta\CMPUT 655 Reinforcement Learning\Project\code\Heatmaps\AUC_anneal_{scheme}_{env['env_name']}.svg") 
        plt.close()

# for cmap in ['viridis','grey','inferno']:
#     for env in [Acrobot,Cartpole]:
#         heatmap(env,cmap)
for env in [Acrobot,Cartpole]:
    heatmap(env,'GnBu')

