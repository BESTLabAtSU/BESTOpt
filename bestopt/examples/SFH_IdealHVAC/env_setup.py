from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment
from bestopt.scripts.runtime_plotter import quick_realtime_plot

# Configuration
cm = ConfigurationManager("config_setup.json")
env = BESTOptEnvironment(cm.config)
plotter = quick_realtime_plot(max_points=200)
# Run Simulation
for timestep in range(env.total_step):
    env.step()

    # Extract data
    hvac_power = env.actions['SFH_1'].thermal.hvac_power
    temperature = env.states['SFH_1'].thermal.thermal_zones['zone0'].temperature

    # Update real-time plot
    plotter.add_data_point(timestep, temperature, hvac_power)

    print(hvac_power)
    print(temperature)
    print(timestep)

plotter.stop()
