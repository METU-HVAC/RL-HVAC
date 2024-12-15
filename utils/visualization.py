import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_data(csv_path):
    # Load the data from the CSV file
    df = pd.read_csv(csv_path)
    
    #df = df.iloc[600:900]
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # Plot outdoor and indoor temperatures
    axs[0].plot(df['Time'], df['Outdoor_Temperature'], label='Outdoor Temperature', color='blue')
    axs[0].plot(df['Time'], df['Air_Temperature'], label='Indoor Air Temperature', color='red')
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].set_title('Outdoor and Indoor Air Temperature vs Time')
    axs[0].legend()

    # Plot power consumption
    axs[1].plot(df['Time'], df['Power_Consumption'], label='Power Consumption', color='green')
    axs[1].set_ylabel('Power Consumption (W)')
    axs[1].set_title('Power Consumption vs Time')
    axs[1].legend()

    # Plot temperature violation
    axs[2].plot(df['Time'], df['Temperature_Violation'], label='Temperature Violation', color='orange')
    axs[2].set_ylabel('Temperature Violation')
    axs[2].set_xlabel('Time')
    axs[2].set_title('Temperature Violation vs Time')
    axs[2].legend()

    # Adjust the layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    plt.close()  # Close the plot
    ## Close the plot Then plot three figures, co2 levels, occupancy and power consumption
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # Plot Co2 Levels
    axs[0].plot(df['Time'], df['CO2_Level'], label='CO2 Levels', color='blue')
    axs[0].set_ylabel('CO2 Level')
    axs[0].set_title('CO2 Levels vs Time')
    axs[0].legend()

    # Plot power consumption
    axs[1].plot(df['Time'], df['Power_Consumption'], label='Power Consumption', color='green')
    axs[1].set_ylabel('Power Consumption (W)')
    axs[1].set_title('Power Consumption vs Time')
    axs[1].legend()

    # Plot people occupants
    axs[2].plot(df['Time'], df['People_Occupants'], label='People Occupants', color='orange')
    axs[2].set_ylabel('People Occupant')
    axs[2].set_xlabel('Time')
    axs[2].set_title('People Occupant vs Time')
    axs[2].legend()
    # Adjust the layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    plt.close()  # Close the plot
    
    

def plot_and_save(
    outdoor_temps, htg_setpoints, clg_setpoints, fan_speeds, 
    air_temps, air_humidities, time_labels,total_temperature_violation,power_consumptions,
    people_occupants,co2_levels,episode_type, 
    episode_num, plots_dir
):
    """
    Plot and save temperature, humidity, and fan speed data for a given episode.

    Args:
        outdoor_temps (list[float]): Outdoor temperature data.
        htg_setpoints (list[float]): Heating setpoint temperature data.
        clg_setpoints (list[float]): Cooling setpoint temperature data.
        fan_speeds (list[float]): Fan speed data (%).
        air_temps (list[float]): Air temperature data.
        air_humidities (list[float]): Air humidity data (not currently plotted).
        time_labels (list[str]): Time labels for the x-axis.
        episode_type (str): Type of the episode (e.g., "Training", "Validation").
        episode_num (int): Episode number.
        plots_dir (str): Directory path to save the plots.

    Returns:
        None

    Side Effects:
        - Saves temperature and fan speed plots as PNG files in `plots_dir`.
        - Prints the save paths for the plots.
    """


    os.makedirs(plots_dir, exist_ok=True)

    # Temperature plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_labels, outdoor_temps, label='Outdoor Temp (°C)', color='blue')
    plt.plot(time_labels, htg_setpoints, label='Heating Setpoint (°C)', color='red')
    plt.plot(time_labels, clg_setpoints, label='Cooling Setpoint (°C)', color='green')
    plt.plot(time_labels, air_temps, label='Air Temp (°C)', color='orange')

    plt.xlabel('Time (HH:MM)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'{episode_type} Episode {episode_num}')
    plt.xticks(range(0, len(time_labels), max(1, len(time_labels) // 10)), rotation=45)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.grid(True)

    plot_path = os.path.join(plots_dir, f"{episode_type}_episode_{episode_num}.png")
    plt.savefig(plot_path)
    plt.close()
    #print(f"Saved {episode_type} plot for Episode {episode_num} at {plot_path}")

    # Fan speed plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_labels, fan_speeds, label='Fan Speed (%)', color='purple')

    plt.xlabel('Time (HH:MM)')
    plt.ylabel('Fan Speed (%)')
    plt.title(f'{episode_type} Episode {episode_num} - Fan Speed')
    plt.xticks(range(0, len(time_labels), max(1, len(time_labels) // 10)), rotation=45)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.grid(True)

    fan_speed_plot_path = os.path.join(
        plots_dir, f"{episode_type}_episode_{episode_num}_fan_speed.png"
    )
    plt.savefig(fan_speed_plot_path)
    plt.close()
    #print(f"Saved {episode_type} fan speed plot for Episode {episode_num} at {fan_speed_plot_path}")

if __name__ == '__main__':
    csv_file_path = '/Users/hekimoglu/workspace/RL-HVAC/results/plots/dqn/experiment_2024-12-15_14-26-15/dqn_data.csv'  # Replace with the actual path to your CSV
    plot_csv_data(csv_file_path)
