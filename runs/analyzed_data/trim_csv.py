import pandas as pd

envs = ['CartPole-v1','Acrobot-v1']
lambdas = ['0.0','0.5','0.9','0.95','0.99','1.0']
steps = 200_000

for env in envs:
    df = pd.read_csv(f'average_return_per_timestep_for_each_lambda_{env}.csv')
    df = df[lambdas]
    print("Current dataframe: \n",df.head())
    df = df[:steps]
    df.to_csv(f'average_return_for_plotting_{env}.csv')