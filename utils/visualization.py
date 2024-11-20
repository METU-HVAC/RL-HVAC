import os
import matplotlib.pyplot as plt

def plot_and_save(
    outdoor_temps, htg_setpoints, clg_setpoints, fan_speeds, 
    air_temps, air_humidities, time_labels, episode_type, 
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
    print(f"Saved {episode_type} plot for Episode {episode_num} at {plot_path}")

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
    print(f"Saved {episode_type} fan speed plot for Episode {episode_num} at {fan_speed_plot_path}")
