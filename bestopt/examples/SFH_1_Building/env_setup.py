"""
Main simulation runner with new cluster-building-system architecture
"""

from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment
from bestopt.scripts.runtime_plotter import create_hvac_dashboard
import os
from pathlib import Path

PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

cm = ConfigurationManager(config_path)
env = BESTOptEnvironment(cm.config)

# Create the enhanced HVAC dashboard
plotter = create_hvac_dashboard(max_points=96 * 1)

# Run Simulation
for timestep in range(env.total_step):
    # Step the environment
    observations, done, info = env.step()

    # Get building and system IDs
    cluster_id = 'residential_cluster_1'
    building_id = 'SFH_1'
    hvac_system_id = env.building_system_map[building_id].get('thermal')

    # Extract zone temperature
    zone_temperature = env.cluster_states[cluster_id].thermal.systems['SFH_1_building'].components['zone0'].temperature

    # Get supervisory setpoints from action
    cooling_setpoint = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].cooling_setpoint_c
    heating_setpoint = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].heating_setpoint_c

    # Get HVAC system module
    hvac_system = env.system_modules[hvac_system_id]

    # Get supply air parameters (actual values from HVAC system)
    supply_air_temp_real = hvac_system.SAT_actual_C
    supply_air_flow_real = hvac_system.SA_flow_actual_m3s

    # Get HVAC performance metrics
    HVAC_power = hvac_system.FCU_power_total_W
    HVAC_thermal_load = hvac_system.Q_zone_actual_W

    # Get supervisory supply air setpoints
    supply_air_temp_setpt = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].supply_temp_setpoint_c
    supply_air_flow_setpt = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].supply_airflow_setpoint_m3s

    # Update the dashboard with all HVAC data
    plotter.add_data_point(
        timestep=timestep,
        zone_temperature=zone_temperature,
        hvac_thermal_load=HVAC_thermal_load,
        hvac_power=HVAC_power,
        supervisory_cooling_setpoint=cooling_setpoint,
        supervisory_heating_setpoint=heating_setpoint,
        supply_air_temp_real=supply_air_temp_real,
        supply_air_temp_setpt=supply_air_temp_setpt,
        supply_air_flow_real=supply_air_flow_real,
        supply_air_flow_setpt=supply_air_flow_setpt
    )

    if done:
        break

plotter.stop()
plotter.save_as_gif('sfh1_dashboard.gif', fps=20)
print(f"Simulation completed: {env.current_step} steps")