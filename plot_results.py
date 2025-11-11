import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os

run_path = "runs/CartPole-v1__ppo__0__0.95__1762816838"
event_file = [f for f in os.listdir(run_path) if f.startswith("events")][0]
ea = event_accumulator.EventAccumulator(os.path.join(run_path, event_file))
ea.Reload()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = []
for tag in ea.Tags()["scalars"]:
    for s in ea.Scalars(tag):
        data.append({"step": s.step, "tag": tag, "value": s.value})

df = pd.DataFrame(data)
df.to_csv("runs/CartPole-v0__ppo__1__1762745559.csv", index=False)
#print(df.head())
#print(df.columns)
#print(df["tag"].unique())

# Filter for the episodic return tag
df_return = df[df["tag"] == "charts/episodic_return"]

# Plot
lambda_value = float(run_path.split('__')[3])
plt.figure(figsize=(8,5))
plt.plot(df_return["step"], df_return["value"], label=f"λ={lambda_value}")

plt.xlabel("Training Steps")
plt.ylabel("Episodic Return")
plt.title("PPO Training Performance")
plt.legend()
plt.grid(True)
plt.show()

