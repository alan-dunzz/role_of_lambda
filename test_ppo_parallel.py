import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor

def test_ppo_lambda(env, gae_lambda, seed):
    terminal_size = shutil.get_terminal_size().columns
    print(fr"PPO env: {env} seed: {seed} lambda: {gae_lambda}".center(terminal_size,'-'))
    subprocess.run(
        fr"python ppo.py --track --seed {seed} --gae_lambda {gae_lambda} --env-id {env} --total-timesteps 100000",
        shell=True,
        check=True
    )

env = "CartPole-v1"

test_ppo_lambda(env,0.95,0)
