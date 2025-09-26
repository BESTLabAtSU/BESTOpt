"""
Configuration file for a single family house
"""

import logging
from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.data_structure import BatteryConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

cm = ConfigurationManager()
logging.info("Started a fresh, empty configuration.")

cm.add_building_component(
    "SFH_1", "thermal_zones", "zone0",
    parameters={
        "model_args": {
            "para": {"Int_h": 8, "Ext_h": 14, "epochs": 20},
            "modeltype": "PI-modnn",
            "startday": 1,
            "trainday": 180,
            "testday": 1,
            # @TODO need a helper function to load data from relative path
            "datapath": "/home/zjiang19/Documents/GitHub/BEST_OPT/bestopt/data/SFH/BLDG/clean/SFH_1.csv",
            "temp_unit": "C"
        },
        "model_path":"/home/zjiang19/Documents/GitHub/BEST_OPT/bestopt/examples/Saved/Eplus/Trained_mdlEnco48_Deco96/PI-modnn_180daysTest_on07-01.pth",
        "scaler_path":"/home/zjiang19/Documents/GitHub/BEST_OPT/bestopt/examples/Scaler/Eplus/ModNN_scaler.pkl",
        "historical_data_path":"/home/zjiang19/Documents/GitHub/BEST_OPT/bestopt/data/SFH/BLDG/clean/SFH_1.csv",
        "encoder_length": 48,
        "retrain": "Off",
    },
    class_path="bestopt.env.modules.building.dynamic.ThermalDynamicsModule"
)

cm.add_building_component(
    "SFH_1", "hvac_systems", "fcu",
    parameters={
    "fan": {"rated_flow_m3s": 1, "rated_power_W": 1*1000}, #fan_power_per_flow ≈ 1,000 – 1,500 W per m³/s
    "fan_ctrl": {"ctrl_type": "linear"},
    "coil": {"epsilon": 0.8},
    "pump": {"rated_flow_m3s": 0.005, "rated_power_W": 0.005*100_000}, # pump_power_per_flow = 100,000 W per m³/s
    "chiller": {"rated_capacity_W": 5_000, "rated_cop": 5.5},
    "tower": {
        "rated_capacity_W": 6_000,
        "rated_fan_power_W": 2000,
        "pump_power_per_flow": 1800,
        "min_approach_C": 3.0,
        "max_approach_C": 7.0
    }
},
    class_path="bestopt.env.modules.hvac.system.FCU.FCUModule"
)


cm.add_controller(
    "SFH_1_THERMAL_Supervisory",
    parameters={
        "domain": "thermal",
        "type": "rule-based",
        "mode": "cooling",
    },
    class_path="bestopt.env.controllers.thermal.SupervisoryController"
)

# cm.add_controller(
#     "SFH_1_Electric_Supervisory",
#     parameters={
#         "domain": "electrical",
#         "type": "rule-based",
#     },
#     class_path="bestopt.env.controllers.electrical.SupervisoryController"
# )

cm.add_disturbance(
    "weather",
    parameters={
        "file_path": "/home/zjiang19/Documents/GitHub/BEST_OPT/bestopt/data/SFH/DIST/weather/weather.csv",
        "simulation_start_time": "2023-08-01 00:00:00", # TODO The same parameters should be handled by one variable
    },
    class_path="bestopt.env.disturbances.weather.WeatherModule"
)

cm.add_disturbance(
    "occupancy",
    parameters={
        "file_path": "/home/zjiang19/Documents/GitHub/BEST_OPT/bestopt/data/SFH/DIST/weather/weather.csv",
        "simulation_start_time": "2023-08-01 00:00:00", # TODO The same parameters should be handled by one variable
    },
    class_path="bestopt.env.disturbances.occupancy.OccupancyModule"
)


cm.add_environment(
    parameters={
        "resolution": 900,   # 15 min in seconds
        "duration": 86400,   # 24 hours in seconds
        "enable_history": True,
        "logging_level": "INFO",
        "simulation_start_time": "2023-08-01 00:00:00",
        "historical_data_path":"/home/zjiang19/Documents/GitHub/BEST_OPT/bestopt/data/SFH/BLDG/clean/SFH_1.csv"
    },
    class_path="bestopt.environment.BestOptEnvironment"
)
cm.select_buildings(["SFH_1"])
cm.select_controller_for_building_domain("SFH_1", "thermal", "SFH_1_THERMAL_Supervisory")
cm.select_disturbances(["weather", "occupancy"])
cm.select_environment()
warnings = cm.validate_configuration()
if warnings:
    print(f"\n⚠ Warnings found: {warnings}")
else:
    print("\n✓ Configuration validation passed")

cm.print_summary()
cm.save_final_configuration("config_setup.json")
print("✓ Saved simulation configuration as simulation_config.json")
