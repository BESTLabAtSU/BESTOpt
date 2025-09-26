from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment
from bestopt.scripts.runtime_plotter import create_hvac_dashboard
import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent 
config_path = base_dir / "config_setup.json"
cm = ConfigurationManager(str(config_path))

# Configuration
# cm = ConfigurationManager(r"bestopt/examples/SFH_IdealHVAC/config_setup.json")
env = BESTOptEnvironment(cm.config)

# Create the enhanced HVAC dashboard
plotter = create_hvac_dashboard(max_points=200)

# Run Simulation
for timestep in range(env.total_step):
    env.step()

    # Extract zone temperature
    zone_temperature = env.states['SFH_1'].thermal.thermal_zones['zone0'].temperature

    # Get supervisory setpoints
    supervisory_cooling_setpoint = env.actions['SFH_1'].thermal.supervisory_cooling_setpoint
    supervisory_heating_setpoint = env.actions['SFH_1'].thermal.supervisory_heating_setpoint

    # Get supply air parameters (actual values)
    supply_air_temp_real = env.building_modules['SFH_1']['hvac_systems']['fcu'].SAT_actual_C
    supply_air_flow_real = env.building_modules['SFH_1']['hvac_systems']['fcu'].SA_flow_actual_m3s

    # Get HVAC performance metrics
    HVAC_power = env.building_modules['SFH_1']['hvac_systems']['fcu'].FCU_power_total_W
    HVAC_thermal_load = env.building_modules['SFH_1']['hvac_systems']['fcu'].Q_zone_actual_W

    # Get supervisory supply air setpoints
    supply_air_temp_setpt = env.actions['SFH_1'].thermal.supervisory_supply_air_temperature
    supply_air_flow_setpt = env.actions['SFH_1'].thermal.supervisory_supply_air_flow_rate

    # Update the dashboard with all HVAC data
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
