import logging
from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.data_structure import BatteryConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

cm = ConfigurationManager(json_file_path="config.json")
print("Available modules:", cm.get_available_modules())

# add a module
cm.add_entry(
    "modules",
    "battery",
    dataclass_obj=BatteryConfig(),
    parameters={"battery_capacity": 25.0},
    class_path="bestopt.modules.battery.Battery",  # keep consistent naming
    overwrite=True
)

# add disturbances
cm.add_entry(
    "disturbances",
    "weather",
    parameters={"file": "USA_NY_Syracuse.tmy3"},
    class_path="bestopt.disturbances.weather.TMYWeather",
    overwrite=True
)
cm.add_entry(
    "disturbances",
    "price",
    parameters={"tariff_name": "utility_X_tou"},
    class_path="bestopt.disturbances.price.TOU",
    overwrite=True
)

# add controllers
cm.add_entry(
    "controllers",
    "mpc",
    parameters={"horizon": 24, "dt_minutes": 15},
    class_path="bestopt.controllers.mpc.MPCController",
    overwrite=True
)
cm.add_entry(
    "controllers",
    "rule",
    parameters={"cooling_setpoint": 24.0},
    class_path="bestopt.controllers.rule.RuleController",
    overwrite=True
)

# add environment
cm.add_entry(
    "environment",
    parameters={"dt_minutes": 15, "sim_steps": 576},
    class_path="bestopt.env.Environment",
    overwrite=True
)

# select runtime pieces
cm.select_modules(["battery"])
cm.select_disturbances(["weather", "price"])
cm.set_active_controller("mpc")
cm.select_environment()

# optional adjustments at runtime
cm.adjust_module_parameter("battery", "battery_capacity", 30.0)
cm.adjust_disturbance_parameter("weather", "file", "USA_NY_Syracuse.epw")
cm.adjust_controller_parameter("horizon", 48)
cm.adjust_environment_parameter("dt_minutes", 5)

# summary, validate, export, save
cm.print_summary()
issues = cm.validate_configuration()
if issues:
    print("\nValidation issues:")
    for w in issues:
        print(" -", w)

final_cfg = cm.get_final_configuration()
print("\nFinal configuration ready for environment.")
# cm.save_configuration("config_runtime_selected.json")
