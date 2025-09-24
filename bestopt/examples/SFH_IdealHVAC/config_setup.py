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
    "SFH_1", "hvac_systems", "fan",
    parameters={
        "fan_type": "staged",  # Can be: ideal, constant, staged, variable
        "stage_air_flow_rate": {"stage1":0.2, "stage2":0.4, "stage3":0.6, "stage4":0.8},
    },
    class_path="bestopt.env.modules.hvac.fan.FanModule"
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

cm.add_local_controller(
    "SFH_1.hvac_systems.fan",  # Component path
    # @TODO it should ahve same pare as parent's module
    parameters={
        "fan_type": "staged",  # Can be: ideal, constant, staged, variable
        "stage_air_flow_rate": {"stage1": 0.2, "stage2": 0.4, "stage3": 0.6, "stage4": 0.8},
    },
    class_path="bestopt.env.controllers.hvac.fan.FanLocalController"
)


cm.add_disturbance(
    "weather",
    parameters={
        "file_path": "/home/zjiang19/Documents/GitHub/BEST_OPT/bestopt/data/SFH/DIST/weather/weather.csv",
    },
    class_path="bestopt.env.disturbances.weather.WeatherModule"
)

cm.add_environment(
    parameters={
        "resolution": 900,   # 15 min in seconds
        "duration": 86400,   # 24 hours in seconds
        "enable_history": True,
        "logging_level": "INFO",
        "simulation_start_time": "2023-08-01 00:00:00"
    },
    class_path="bestopt.environment.BestOptEnvironment"
)
cm.select_buildings(["SFH_1"])
cm.select_controller_for_building_domain("SFH_1", "thermal", "SFH_1_THERMAL_Supervisory")
cm.select_disturbances(["weather"])
cm.select_environment()
warnings = cm.validate_configuration()
if warnings:
    print(f"\n⚠ Warnings found: {warnings}")
else:
    print("\n✓ Configuration validation passed")

cm.print_summary()
cm.save_final_configuration("config_setup.json")
print("✓ Saved simulation configuration as simulation_config.json")
