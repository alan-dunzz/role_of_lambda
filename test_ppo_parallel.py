import sys
import numpy as np
from ppo import ppo_run


lambdas = [0.95]  # example with more than one lambda
seeds = np.arange(0,5,1)
env = "CartPole-v1"

# Create all combinations of lambda and seed
combinations = [(gae_lambda, seed) for gae_lambda in lambdas for seed in seeds]
i = int(sys.argv[1])
gae_lambda, seed = combinations[i]

ppo_run(env_id=env,gae_lambda=gae_lambda,seed=seed)
