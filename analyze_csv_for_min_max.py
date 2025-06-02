
import pandas as pd
import numpy as np

# Path to the CSV file
csv_path = "/home/mehmetbh/workspace/RL-Paper/results/multispeed_setpoint/hot_A403medium_baseline_2025-05-26_22:54/multispeed_setpoint.csv"

# Load the CSV file
df = pd.read_csv(csv_path)

# Desired observation keys
keys_of_interest = [
    "months", "day_of_months", "hours", "outdoor_temperatures", "outdoor_humidities",
    "htg_setpoints", "clg_setpoints", "air_temperatures", "air_humidities",
    "people_occupants", "air_co2s", "window_fan_energies", "total_electricity_HVACs"
]

# Filter the dataframe for those columns
df_filtered = df[keys_of_interest]

# Compute min and max values
min_values = df_filtered.min()
max_values = df_filtered.max()

# Format as numpy.float32 and prepare output strings
min_formatted = [f"np.float32({v})" for v in min_values.tolist()]
max_formatted = [f"np.float32({v})" for v in max_values.tolist()]

min_output = "obs_mins = [\n    " + ",\n    ".join(min_formatted) + "\n]"
max_output = "obs_maxs = [\n    " + ",\n    ".join(max_formatted) + "\n]"

print(min_output, max_output)