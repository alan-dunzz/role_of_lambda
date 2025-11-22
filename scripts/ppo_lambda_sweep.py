import sys
import numpy as np
from ppo import ppo_run
import os

# Define the range of lambda values, seeds and environment
lambdas = np.concatenate([np.linspace(0,0.9,91) , np.linspace(0.9,1,21)[1:]])
seeds = np.arange(0,100,1, dtype=int)
env = "CartPole-v1"

# Create all combinations of lambda and seed
combinations = [(gae_lambda, seed) for gae_lambda in lambdas for seed in seeds]
i = int(sys.argv[1])
gae_lambda, seed = combinations[i]

seed = int(seed)

step_and_episodic_returns = ppo_run(
    env_id=env,
    gae_lambda=gae_lambda,seed=seed, 
    total_timesteps=500_000
)

# Save the array with episodic returns to a file in folder runs/{env}/lambda_{value}/seed_{value}.csv
# Make sure the folder exists and create it if it does not
save_folder = f"runs/{env}/lambda_{gae_lambda}"
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, f"seed_{seed}.csv")

# Save to CSV with header
with open(save_path, "w") as f:
    f.write("global_step,episodic_return\n")
    for step, ret in step_and_episodic_returns:
        f.write(f"{step},{ret}\n")
