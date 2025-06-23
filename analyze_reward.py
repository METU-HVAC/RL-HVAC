import pandas as pd
import matplotlib.pyplot as plt
import os

# 1) Specify which CSV you want to plot
csv_file = "/home/mehmetbh/workspace/RL-Paper/reward_logs/validation_rewards_ep6.csv"  # or "validation_rewards_ep1.csv", etc.
csv_path = os.path.join("reward_logs", csv_file)

# 2) Read the CSV into a DataFrame
df = pd.read_csv(csv_path)

# 3) Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df["timestep"], df["ac_fan_energy_term"],     label="AC Fan Energy Term", linewidth=1)
plt.plot(df["timestep"], df["window_fan_energy_term"], label="Window Fan Energy Term", linewidth=1)
plt.plot(df["timestep"], df["comfort_term"],           label="Comfort Term",          linewidth=1)
plt.plot(df["timestep"], df["co2_term"],               label="CO₂ Term",              linewidth=1)

plt.xlabel("Timestep")
plt.ylabel("Sub-Reward Value")
plt.title(f"Sub-Rewards Over Time ({csv_file})")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()

# 4) Save the figure to PNG
png_filename = csv_file.replace(".csv", ".png")  # e.g. "validation_rewards_ep3.png"
png_path = os.path.join("reward_logs", png_filename)
plt.savefig(png_path)