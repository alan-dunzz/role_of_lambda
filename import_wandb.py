import wandb
import pandas as pd
import os
from tqdm import tqdm

LOG_FILE = "completed_files.txt"

# --- Load completed filenames ---
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        completed_files = set(line.strip() for line in f.readlines())
else:
    completed_files = set()

api = wandb.Api()

current_dir = os.getcwd()

runs = api.runs("alan_dunzz-university-of-alberta/role_of_lambda")
i = 0
total = len(runs)

for run in tqdm(runs, total=total):
    try:
        seed = run.config.get("seed")
        gae_lambda = run.config.get("gae_lambda")
        if isinstance(gae_lambda, float):
            lambda_clean = f"{gae_lambda:.3f}".rstrip("0").rstrip(".")
        else:
            lambda_clean = str(gae_lambda)
        print(fr"gae_lambda = {lambda_clean}, seed = {seed}")
        
        out_file = f"run_lambda_{lambda_clean}_seed_{seed}.csv"
        i+=1
        if out_file in completed_files:
            print(f"[SKIP] {out_file} already exists. ({i}/{total})")
            pass
        else:
            df = pd.DataFrame(run.scan_history(keys=['charts/episodic_return','global_step']))
            df.rename(columns={"charts/episodic_return": "episodic_return"}, inplace=True)
            out_dir = os.path.join(current_dir, "runs", f"lambda_{lambda_clean}")
            
            os.makedirs(out_dir, exist_ok=True)
            
            full_path = os.path.join(out_dir, out_file)
            df.to_csv(full_path, index=False)
            with open(LOG_FILE, "a") as f:
                f.write(out_file + "\n")
            completed_files.add(out_file)
            print(f"File saved successfully ({i}/{total})")
    except:
        print(f"Error. Couldn't save file {out_file} ({i}/{total})")
        break
print(f"Job completed: ({i}/{total})")
