import streamlit as st
import pandas as pd
import plotly.express as px
import os
#streamlit run experiments/ablation/plot_logs_online.py
# Define dataset information
# datasets_info = {
#     "On-Off": {"csv": "results/observations/on_off_20250313.csv", "color": "blue"},
#     "Setpoint": {"csv": "results/observations/setpoint_20250313.csv", "color": "green"},
#     "DQN-LowTempViol": {"csv": "results/observations/dqn_co2_100_temp_100_energy_1_lr_3e-03_11_20250319_2320.csv", "color": "orange"},
#     "DQN-LowPower": {"csv": "results/observations/dqn_co2_100_temp_100_energy_1_lr_3e-03_5_20250319_2308.csv", "color": "red"},
# }
# datasets_info = {
#     "On-Off": {"csv": "results/observations/on_off_20250313.csv", "color": "blue"},
#     "Setpoint": {"csv": "results/observations/setpoint_20250313.csv", "color": "green"},
#     "DQN-LowTempViol": {"csv": "results/observations/dqn_co2_50_temp_100_energy_1_lr_3e-03_12_20250320_0115.csv", "color": "orange"},
#     "DQN-LowPower": {"csv": "results/observations/dqn_co2_50_temp_100_energy_1_lr_3e-03_5_20250320_0100.csv", "color": "red"},
# }
datasets_info = {
    "On-Off": {"csv": "results/observations/on_off_20250313.csv", "color": "blue"},
    "Setpoint": {"csv": "results/observations/setpoint_20250313.csv", "color": "green"},
    "DQN-LowTempViol": {"csv": "results/observations/dqn_co2_10_temp_100_energy_1_lr_3e-03_12_20250320_0546.csv", "color": "orange"},
    "DQN-LowPower": {"csv": "results/observations/dqn_co2_50_temp_50_energy_1_lr_3e-03_5_20250320_0256.csv", "color": "red"},
    "DQN-LowPower2": {"csv": "results/observations/dqn_co2_50_temp_100_energy_1_lr_3e-03_13_20250320_0242.csv", "color": "purple"},
}
dataset_labels = list(datasets_info.keys())

# List of all available variables
variables_to_plot = ["outdoor_temperatures", "outdoor_humidities", "htg_setpoints",
                     "clg_setpoints", "air_temperatures", "air_humidities", "people_occupants",
                     "air_co2s", "window_fan_energies", "total_electricity_HVACs", "temp_violations",
                     "co2_violations", "window_fan_speeds", "ac_fan_speeds"]

# Sidebar configuration for index range
st.sidebar.title("Plot Configuration")
start_idx = st.sidebar.number_input("Start Index", min_value=0, value=1000)
end_idx = st.sidebar.number_input("End Index", min_value=0, value=2000)

# Dataset selection checkboxes
st.sidebar.subheader("Select Dataset(s)")
selected_datasets = []
for dataset in dataset_labels:
    if st.sidebar.checkbox(dataset, value=True, key=f"dataset_{dataset}"):
        selected_datasets.append(dataset)

# Variable selection checkboxes within an expander (to reduce clutter)
with st.sidebar.expander("Select Variable(s) to Plot"):
    selected_variables = []
    for var in variables_to_plot:
        if st.sidebar.checkbox(var, key=f"var_{var}"):
            selected_variables.append(var)

# Main Title
st.title("Interactive HVAC Observation Plots (3xN Layout)")

# Function to generate a plot for a given variable and selected datasets
def generate_plot(variable, selected_datasets):
    fig = None
    for dataset in selected_datasets:
        csv_file = datasets_info[dataset]["csv"]
        color = datasets_info[dataset]["color"]
        df = pd.read_csv(csv_file)
        subset_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        subset_df['timestep'] = range(len(subset_df))
        
        temp_fig = px.line(
            subset_df, x='timestep', y=variable,
            labels={'timestep': 'Timestep', variable: variable},
            line_shape='linear',
            color_discrete_sequence=[color],
            title=f'{variable} over Time'
        )
        temp_fig.update_traces(name=dataset, showlegend=True)
        temp_fig.update_layout(
            xaxis_title='Timestep',
            yaxis_title=variable,
            font=dict(size=14),
            showlegend=True
        )
        if fig is None:
            fig = temp_fig
        else:
            for trace in temp_fig.data:
                fig.add_trace(trace)
    return fig

# Generate plots only if at least one dataset and one variable are selected
last_fig = None
if not selected_datasets:
    st.write("Please select at least one dataset to plot from the sidebar.")
elif not selected_variables:
    st.write("Please select at least one variable to plot from the sidebar.")
else:
    cols = st.columns(3)
    for idx, variable in enumerate(selected_variables):
        col = cols[idx % 3]
        with col:
            fig = generate_plot(variable, selected_datasets)
            st.plotly_chart(fig, use_container_width=True)
            last_fig = fig  # store the last plotted figure

# Save plot section remains in the sidebar
st.sidebar.markdown("### Save the Plot")
if st.sidebar.button("Save Current Plot"):
    if last_fig is not None:
        output_dir = 'figures'
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, 'interactive_plot.jpg')
        last_fig.write_image(filename)
        st.sidebar.success(f"Plot saved to {filename}")
    else:
        st.sidebar.error("No plot available to save. Please select options first.")
