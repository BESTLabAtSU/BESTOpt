from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.data_structure import BatteryConfig

config_manager = ConfigurationManager(json_file_path="config.json")
config_manager.get_available_modules()

config_manager.add_module(
    "battery",
    dataclass_obj=BatteryConfig(),
    parameters={"battery_capacity": 25.0},
    class_path="bestopt.modules.Battery",
    overwrite=True
)

config_manager.select_modules(['battery'])
config_manager.print_summary()
config_manager.get_final_configuration()





