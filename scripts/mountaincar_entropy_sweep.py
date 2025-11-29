import sys
import numpy as np
from ppo import ppo_run
import os
import itertools

# Define the range of lambda values, seeds and environment
entropy_coefficients = [0.1, 0.2, 0.3, 0.4, 0.5]
use_entropy_array = [True, False]
seeds = np.arange(0,30,1, dtype=int)
env = "MountainCar-v0"

# Create all combinations of lambda and seed
combinations = list(itertools.product(entropy_coefficients, seeds, use_entropy_array))
i = int(sys.argv[1])
entropy_coefficient, seed, use_entropy = combinations[i]

seed = int(seed)

step_and_episodic_returns = ppo_run(
    env_id=env,
    gae_lambda=0.95, seed=seed,
    total_timesteps=1_000_000,
    num_steps=2048,
    ent_coef=entropy_coefficient,
    anneal_entropy=use_entropy
)

# Save the array with episodic returns to a file in folder runs/{env}/lambda_{value}/seed_{value}.csv
# Make sure the folder exists and create it if it does not
save_folder = f"runs/{env}/entropy_sweep/ent_coef_{entropy_coefficient}/use_entropy_{use_entropy}/"
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, f"seed_{seed}.csv")

# Save to CSV with header
with open(save_path, "w") as f:
    f.write("global_step,episodic_return\n")
    for step, ret in step_and_episodic_returns:
        f.write(f"{step},{ret}\n")
