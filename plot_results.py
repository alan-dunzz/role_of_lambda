import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os

run_path = r"\runs\lambda_0.3\run_lambda_0.3_seed_11.csv"
event_file = [f for f in os.listdir(run_path) if f.startswith("events")][0]
ea = event_accumulator.EventAccumulator(os.path.join(run_path, event_file))
ea.Reload()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df = pd.read_csv(run_path)
df.rename(columns={"charts/episodic_return": "episodic_return"}, inplace=True)

# Plot
lambda_value = float(run_path.split('__')[3])
plt.figure(figsize=(8,5))
plt.plot(df["global_step"], df["episodic_return"], label=f"λ={lambda_value}")

plt.xlabel("Training Steps")
plt.ylabel("Episodic Return")
plt.title("PPO Training Performance")
plt.legend()
plt.grid(True)
plt.show()

