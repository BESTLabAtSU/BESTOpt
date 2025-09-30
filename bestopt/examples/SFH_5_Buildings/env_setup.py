"""
Streamlined HVAC Dashboard Simulation
Two patterns: Single Building or Multi-Building Comparison
"""

from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment
from bestopt.scripts.runtime_plotter import (
    create_hvac_dashboard,
    create_multi_dashboard
)
import os

# Setup paths and initialize environment
PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))
config_path = os.path.join(PROJECT_ROOT_PATH, "examples", "SFH_5_Buildings", "config_setup.json")

cm = ConfigurationManager(str(config_path))
env = BESTOptEnvironment(cm.config)

# Get list of all buildings
building_names = list(env.states.keys())
print(f"Simulating {len(building_names)} buildings: {building_names}\n")

def run_single_building():
    """Monitor one building with full HVAC details"""
    # Select which building to monitor
    target_building = building_names[0]

    # Create dashboard
    plotter = create_hvac_dashboard(
        max_points=96 * 2,
        window_title=f"HVAC Monitor - {target_building}"
    )

    # Run simulation
    for timestep in range(env.total_step):
        env.step()

        # Extract data
        zone_temperature = env.states[target_building].thermal.thermal_zones['zone0'].temperature
        supervisory_cooling_setpoint = env.actions[target_building].thermal.supervisory_cooling_setpoint
        supervisory_heating_setpoint = env.actions[target_building].thermal.supervisory_heating_setpoint
        supply_air_temp_real = env.building_modules[target_building]['hvac_systems']['fcu'].SAT_actual_C
        supply_air_flow_real = env.building_modules[target_building]['hvac_systems']['fcu'].SA_flow_actual_m3s
        HVAC_power = env.building_modules[target_building]['hvac_systems']['fcu'].FCU_power_total_W
        HVAC_thermal_load = env.building_modules[target_building]['hvac_systems']['fcu'].Q_zone_actual_W
        supply_air_temp_setpt = env.actions[target_building].thermal.supervisory_supply_air_temperature
        supply_air_flow_setpt = env.actions[target_building].thermal.supervisory_supply_air_flow_rate

        # Update dashboard
        plotter.add_data_point(
            timestep=timestep,
            zone_temperature=zone_temperature,
            hvac_thermal_load=HVAC_thermal_load,
            hvac_power=HVAC_power,
            supervisory_cooling_setpoint=supervisory_cooling_setpoint,
            supervisory_heating_setpoint=supervisory_heating_setpoint,
            supply_air_temp_real=supply_air_temp_real,
            supply_air_temp_setpt=supply_air_temp_setpt,
            supply_air_flow_real=supply_air_flow_real,
            supply_air_flow_setpt=supply_air_flow_setpt
        )

    plotter.stop()
    print(f"\nSimulation complete - {target_building}")

def run_multi_building():
    """Monitor multiple buildings with comparison dashboard"""
    # Create multi-building dashboard
    plotter = create_multi_dashboard(
        building_names=building_names,
        max_points=96 * 2,
        save_gif=True,
        gif_filename="hvac_simulation.gif"
    )

    # Run simulation
    for timestep in range(env.total_step):
        env.step()

        # Collect data from all buildings
        building_data_dict = {}
        for building_name in building_names:
            zone_temperature = env.states[building_name].thermal.thermal_zones['zone0'].temperature
            HVAC_power = env.building_modules[building_name]['hvac_systems']['fcu'].FCU_power_total_W
            HVAC_thermal_load = env.building_modules[building_name]['hvac_systems']['fcu'].Q_zone_actual_W

            building_data_dict[building_name] = {
                'temperature': zone_temperature,
                'thermal_load': HVAC_thermal_load,
                'power': HVAC_power
            }

        # Update dashboard
        plotter.add_data_point(timestep, building_data_dict)

    plotter.stop()
    print("\nSimulation complete - All buildings")

if __name__ == "__main__":
    PATTERN = 2  # 1 or 2
    if PATTERN == 1:
        run_single_building()
    elif PATTERN == 2:
        run_multi_building()
    else:
        print("Invalid pattern. Choose 1 or 2.")