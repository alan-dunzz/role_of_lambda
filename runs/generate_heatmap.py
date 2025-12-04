import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,                     # use LaTeX to render text
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], # Computer Modern (LaTeX default)
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
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
        plt.figure(figsize=(8, 7))

        plt.imshow(df, aspect='equal', vmin=env['min'], vmax=env['max'],cmap=cmap_v)
        plt.colorbar(label="AUC")

        plt.xticks(range(len(df.columns)), df.columns, rotation=0)
        plt.yticks(range(len(df.index)), df.index)

        plt.xlabel(r"$\lambda$",labelpad=10)
        plt.ylabel(r"$\alpha$",rotation=0,labelpad=10)

        plt.title(fr"{env['env_name']}")

        plt.tight_layout()

        plt.savefig(fr"C:\Users\aland\Desktop\University of Alberta\CMPUT 655 Reinforcement Learning\Project\code\Heatmaps\AUC_anneal_{scheme}_{env['env_name']}_{cmap_v}.png") 
        #plt.savefig(fr"C:\Users\aland\Desktop\University of Alberta\CMPUT 655 Reinforcement Learning\Project\code\Heatmaps\AUC_anneal_{lr_scheme}_{env_name}.svg") 
        plt.close()

# for cmap in ['viridis','grey','inferno']:
#     for env in [Acrobot,Cartpole]:
#         heatmap(env,cmap)
for env in [Acrobot,Cartpole]:
    heatmap(env,'GnBu')

