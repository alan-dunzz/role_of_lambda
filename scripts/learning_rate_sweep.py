import sys
import numpy as np
from ppo import ppo_run
import os
import itertools

# Define the range of lambda values, learning rates, seeds and environment
learning_rates = np.array([1e-3,3e-3,6e-3,1e-4,3e-4,6e-4,1e-5,3e-5,6e-5])
lambdas = [0.0,0.36,0.68,0.84,0.92,0.96,0.98,0.99,1.0]
anneal_lrs = [True,False]
seeds = np.arange(0,30,1, dtype=int)
env = "CartPole-v1"

# Create all combinations of lambda, learning rates and seed
combinations = list(itertools.product(lambdas, learning_rates,seeds,anneal_lrs))
i = int(sys.argv[1])
laba, learning_rate, seed, anneal_lr = combinations[i]

seed = int(seed)

step_and_episodic_returns = ppo_run(
    env_id=env,
    gae_lambda=laba, seed=seed,
    total_timesteps=500_000,
    learning_rate=learning_rate,
    anneal_lr=anneal_lr
)

# Save the array with episodic returns to a file in folder runs/{env}/lambda_{value}/seed_{value}.csv
# Make sure the folder exists and create it if it does not
save_folder = f"runs/{env}/lr_sweep/lambda_{laba}/learning_rate_{learning_rate}/anneal_lr_{anneal_lr}/"
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, f"seed_{seed}.csv")

# Save to CSV with header
with open(save_path, "w") as f:
    f.write("global_step,episodic_return\n")
    for step, ret in step_and_episodic_returns:
        f.write(f"{step},{ret}\n")
