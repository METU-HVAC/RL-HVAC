import pandas as pd
import matplotlib.pyplot as plt
import os
# streamlit run experiments/ablation/plot_logs_online.py
# streamlit run app.py --server.port 8501 --server.address 0.0.0.0

csv_files = ["results/observations/on_off_20250312.csv", "results/observations/setpoint_20250312.csv"]  

# Variables to plot


variables_to_plot = ["outdoor_temperatures","outdoor_humidities","htg_setpoints",
                     "clg_setpoints","air_temperatures","air_humidities","people_occupants",
                     "air_co2s","window_fan_energies","total_electricity_HVACs","temp_violations",
                     "co2_violations","window_fan_speeds","ac_fan_speeds"
]
colors = ['b', 'g', 'r']  # Different colors for each dataset

# Plotting a slice (e.g., steps 1000 to 2000)
start_idx = 1000
end_idx = 2000

plt.figure(figsize=(12, 8))

for i, variable in enumerate(variables_to_plot, 1):
    plt.subplot(2, 2, i)

    for file_idx, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        subset_df = df.iloc[start_idx:end_idx]
        time_labels = subset_df['time_labels']
        
        # Plot with different colors
        plt.plot(time_labels, subset_df[variable], label=csv_file.replace('.csv', ''), color=colors[file_idx])

    plt.title(f'{variable} over Time', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(variable, fontsize=12)
    plt.grid(True)

    # Show x-axis labels every 100 steps for clarity
    plt.xticks(ticks=range(0, len(time_labels), 100), labels=time_labels[::100], rotation=45)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    plt.legend()

plt.tight_layout()

# Save the figure
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'comparison_plot.jpg'), format='jpg')
plt.close()
