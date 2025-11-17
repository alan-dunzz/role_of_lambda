import sys
import numpy as np
from ppo import ppo_run


lambdas = np.concatenate([np.linspace(0,0.9,10) , np.linspace(0.9,1,21)[1:]])
seeds = np.arange(0,30,1, dtype=int)
env = "CartPole-v1"

# Create all combinations of lambda and seed
combinations = [(gae_lambda, seed) for gae_lambda in lambdas for seed in seeds]
i = int(sys.argv[1])
gae_lambda, seed = combinations[i]

seed = int(seed)

ppo_run(
    env_id=env,
    gae_lambda=gae_lambda,seed=seed, 
    total_timesteps=500_000
)
