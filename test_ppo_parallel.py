import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor

def test_ppo_lambda(env, gae_lambda, seed):
    terminal_size = shutil.get_terminal_size().columns
    print(fr"PPO env: {env} seed: {seed} lambda: {gae_lambda}".center(terminal_size,'-'))
    subprocess.run(
        fr"python cleanrl/ppo.py --seed {seed} --gae_lambda {gae_lambda} --env-id {env} --total-timesteps 100000",
        shell=True,
        check=True
    )

lambdas = [0.95, 0.9]  # example with more than one lambda
seeds = [0, 1, 2]
env = "CartPole-v1"

# Create all combinations of lambda and seed
combinations = [(gae_lambda, seed) for gae_lambda in lambdas for seed in seeds]

# Run in parallel using threads
with ThreadPoolExecutor(max_workers=len(combinations)) as executor:
    futures = [executor.submit(test_ppo_lambda, env, gae_lambda, seed) for gae_lambda, seed in combinations]

# Optionally wait for all to finish
for future in futures:
    future.result()
