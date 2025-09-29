"""
Configuration file for a single family house
"""

import logging
from bestopt.env.core.config_manager import ConfigurationManager
import os
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))
# @TODO some dataframes are used for multiple modules, need to clean up later
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
            "datapath": os.path.join(PROJECT_ROOT_PATH, "data", "SFH", "BLDG", "clean", "SFH_1.csv"),
            "temp_unit": "C"
        },
        "model_path": os.path.join(PROJECT_ROOT_PATH, "examples", "Trained_mdl", "PImodnn.pth"),
        "scaler_path": os.path.join(PROJECT_ROOT_PATH, "examples", "Scaler", "Eplus", "ModNN_scaler.pkl"),
        "historical_data_path": os.path.join(PROJECT_ROOT_PATH, "data", "SFH", "BLDG", "clean", "SFH_1.csv"),
        "encoder_length": 48,
        "retrain": "Off",
    },
    class_path="bestopt.env.modules.building.dynamic.ThermalDynamicsModule"
)

cm.add_building_component(
    "SFH_1", "hvac_systems", "fcu",
    parameters={
        "fan": {"rated_flow_m3s": 1, "rated_power_W": 1*1000},
        "fan_ctrl": {"ctrl_type": "linear"},
        "coil": {"epsilon": 0.8},
        "pump": {"rated_flow_m3s": 0.005, "rated_power_W": 0.005*100_000},
        "chiller": {"rated_capacity_W": 3500, "rated_cop": 4.5},
        "tower": {
            "rated_capacity_W": 3500,
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
        "precooling": {
                "degree": 0,
                "hours": 0}
    },
    class_path="bestopt.env.controllers.thermal.SupervisoryController"
)

cm.add_disturbance(
    "weather",
    parameters={
        "file_path": os.path.join(PROJECT_ROOT_PATH, "data", "SFH", "DIST", "weather", "weather.csv"),
        "simulation_start_time": "2023-08-01 00:00:00",
    },
    class_path="bestopt.env.disturbances.weather.WeatherModule"
)

cm.add_disturbance(
    "occupancy",
    parameters={
        "file_path": os.path.join(PROJECT_ROOT_PATH, "data", "SFH", "DIST", "occupancy", "occupancy.csv"),
        "simulation_start_time": "2023-08-01 00:00:00",
    },
    class_path="bestopt.env.disturbances.occupancy.OccupancyModule"
)

cm.add_disturbance(
    "price",
    parameters={},
    class_path="bestopt.env.disturbances.price.PriceModule"
)

cm.add_environment(
    parameters={
        "resolution": 900,
        "duration": 86400*3,
        "enable_history": True,
        "logging_level": "INFO",
        "simulation_start_time": "2023-08-01 00:00:00",
        "historical_data_path": os.path.join(PROJECT_ROOT_PATH, "data", "SFH", "BLDG", "clean", "SFH_1.csv")
    },
    class_path="bestopt.environment.BestOptEnvironment"
)

cm.select_buildings(["SFH_1"])
cm.select_controller_for_building_domain("SFH_1", "thermal", "SFH_1_THERMAL_Supervisory")
cm.select_disturbances(["weather", "occupancy", "price"])
cm.select_environment()

warnings = cm.validate_configuration()
if warnings:
    print(f"\n⚠ Warnings found: {warnings}")
else:
    print("\n✓ Configuration validation passed")

cm.print_summary()
cm.save_final_configuration("config_setup.json")
print("✓ Saved simulation configuration as config_setup.json")
