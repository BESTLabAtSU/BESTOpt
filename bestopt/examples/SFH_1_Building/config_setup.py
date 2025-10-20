"""
Configuration setup for Single Family House
"""

import logging
import os
from bestopt.env.core.config_manager import ConfigurationManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

# Initialize configuration manager
cm = ConfigurationManager()
logging.info("Started a fresh, empty configuration.")


cm.add_cluster("residential_cluster_1", parameters={"location": "Syracuse, NY"})
cm.add_building(
    cluster_id="residential_cluster_1",
    building_id="SFH_1",
    parameters={
        "building_type": "single_family_home",
    },
    thermal_zones=["zone0"]  # Single zone for now
)

# Add thermal zone to building
cm.add_thermal_zone_module(
    building_id="SFH_1",
    zone_id="zone0",
    parameters={
        "model_args": {
            "para": {"Int_h": 8, "Ext_h": 14, "epochs": 20},
            "modeltype": "PI-modnn",
            "startday": 1,
            "trainday": 180,
            "testday": 1,
            "datapath": os.path.join(PROJECT_ROOT_PATH, "data", "SFH", "BLDG", "clean", "SFH_1.csv"),
            "temp_unit": "C",
            "device": "cuda:0",
            "save_name": "SFH_1"
        },
        "model_path": os.path.join(PROJECT_ROOT_PATH, "examples", "Saved", "SFH_1",
                                   "Trained_mdlEnco48_Deco96", "PI-modnn_180daysTest_on07-01.pth"),
        "scaler_path": os.path.join(PROJECT_ROOT_PATH, "examples", "Scaler", "SFH_1", "ModNN_scaler.pkl"),
        "historical_data_path": os.path.join(PROJECT_ROOT_PATH, "data", "SFH", "BLDG", "clean", "SFH_1.csv"),
        "encoder_length": 48,
        "retrain": "Off",
        "simulation_start_time": "2023-08-01 00:00:00",
    },
    class_path="bestopt.env.modules.building.thermal.ThermalDynamicsModule"
)

# cm.add_electrical_zone_module(
#     building_id="SFH_1",
#     zone_id="zone0",
#     parameters={
#         "lighting": {"daytime": 600,
#                      "nighttime": 1800},
#         "appliance": {"cooking": 2000,
#                       "tv": 200,
#                       "pc": 400,
#                       "dishwashing": 1000
#                       },
#     },
#     class_path="bestopt.env.modules.building.electrical.ElectricalDynamicModule"
# )

# HVAC System (Thermal Domain)
cm.add_system(
    cluster_id="residential_cluster_1",
    system_id="hvac_system_1",
    system_type="hvac_systems",
    parameters={
        "system_name": "FCU System",
        "system_config": {
            "fan": {"rated_flow_m3s": 1, "rated_power_W": 1000},
            "fan_ctrl": {"ctrl_type": "linear"},
            "coil": {"epsilon": 0.8},
            "pump": {"rated_flow_m3s": 0.005, "rated_power_W": 500},
            "chiller": {"rated_capacity_W": 3500, "rated_cop": 4.5},
            "tower": {
                "rated_capacity_W": 3500,
                "rated_fan_power_W": 2000,
                "pump_power_per_flow": 1800,
                "min_approach_C": 3.0,
                "max_approach_C": 7.0
            }
        }
    },
    class_path="bestopt.env.modules.hvac.system.FCU.FCUModule"
)

cm.add_building_component(
    "SFH_1", "der_systems", "pv_bat_ev",
    parameters={
        "system_config": {"pv":  {"rated_capacity_kW": 2},
                          "bat": {"rated_capacity_kWh": 5,
                                  "initial_soc": 0.3, },
                          "ev":  {"rated_capacity_kWh": 5,
                                  "initial_soc": 0.3, },
                          },
    },
    class_path="bestopt.env.modules.ders.system.der.DERModule"
)

# Water System (placeholder for future)
# cm.add_system(
#     cluster_id="residential_cluster_1",
#     system_id="water_system_1",
#     system_type="water_systems",
#     parameters={"system_name": "Water Heater System"},
#     class_path="bestopt.env.modules.water.system.WaterModule"
# )

# Assign Systems to Buildings
cm.assign_system_to_buildings("hvac_system_1", ["SFH_1"])
# cm.assign_system_to_buildings("der_system_1", ["SFH_1"])

# HVAC System Controller (Thermal)
cm.add_system_controller(
    controller_id="hvac_controller_1",
    system_id="hvac_system_1",
    parameters={
        "domain": "thermal",
        "type": "rule-based",
        "mode": "cooling",
        "precooling": {"degree": 0, "hours": 0},
        "base_cooling": 24.0,
        "base_heating": 18.0,
        "deadband": 0.5,
        "cooling_power_max": 4000.0,
        "heating_power_max": 4000.0
    },
    class_path="bestopt.env.controllers.thermal.SupervisoryController"
)

cm.add_controller(
    "SFH_1_ELECTRIC_Supervisory",
    parameters={
        "domain": "electrical",
        "type": "rule-based",
        "mode": "self_consumption",
        # supervisory controller need to know the system info
        # @ TODO need to reduce the redundancy later
        "system_config": {"pv":  {"rated_capacity_kW": 2},
                          "bat": {"rated_capacity_kWh": 5,
                                  "initial_soc": 0.3, },
                          "ev":  {"rated_capacity_kWh": 5,
                                  "initial_soc": 0.3, },
                          }
    },
    class_path="bestopt.env.controllers.electrical.SupervisoryController"
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
        "resolution": 900,  # 15 minutes
        "duration": 86400 * 3,  # 3 days
        "enable_history": True,
        "logging_level": "INFO",
        "simulation_start_time": "2023-08-01 00:00:00",
    },
    class_path="bestopt.environment.BESTOptEnvironment"
)


# Select cluster and its components
cm.select_cluster("residential_cluster_1")
cm.select_buildings(["SFH_1"])
cm.select_controller_for_building_domain("SFH_1", "thermal", "SFH_1_THERMAL_Supervisory")
cm.select_controller_for_building_domain("SFH_1", "electrical", "SFH_1_ELECTRIC_Supervisory")
cm.select_disturbances(["weather", "occupancy", "price"])
cm.select_environment()

warnings = cm.validate_configuration()
if warnings:
    print(f"\n⚠ Warnings found: {warnings}")
else:
    print("\n✓ Configuration validation passed")

# Print summary
cm.print_summary()

# Save configuration
cm.save_final_configuration("config_setup.json")
print("✓ Saved simulation configuration as config_setup.json")