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
total = len(runs)
i = 0
"""
# Recover specific runs
run_ids = ['xckpo1v3', 'eji4nqbn', 'yo98yc9n', 'ww24f3vf', '8mvyp4nv', 'x2jza2di', 'm0guqn3d', 'vd7cerhm', 'iv0qt0bf', 'o13ce5sx', 'mk0ax9zz',
           'j9htc3wr','zhhxja3m','2olnnbry','35mcr3nt','r1ncx0cb','r8g5z7nj','t2y9fpo7','t8vs1911','y13qxkjv','zoetdrby','00in18gq','349pczwl',
           '4bvvz9ob','3xfipjbv','0xajzt1j','1xn9k4d7','2iipxilh','37c1i4r5','3lr4pyqu','wlcihes3','4ptztvsr','5kk8dkzs','5xgsj8j8','dkh4z8g8',
           'e51dclkq','a2q5wtfe','5xgsj8j8','27m46zgq','297jurs7','2hxgz47p','4r0mr9gm','rh5y5frg','rm5hijb5','uogvnoxo','wl6dvnct','x1hu7hlz',
           'xe9d7u30','xv8u5vfs','z5383pnz','4ffa17cw','554xiw9o','6hh5segt','00poim9j','3ig7hp95','1v4hl07f','36o1zxdk','6ashr68d','vxiujj17','zgsspocr']

for run_id in run_ids:
    run = api.run(f"alan_dunzz-university-of-alberta/role_of_lambda/{run_id}")
"""
# Iterate over all runs
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
