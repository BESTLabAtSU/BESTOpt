import logging
from bestopt.env.core.config_manager import ConfigurationManager
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

cm = ConfigurationManager()
logging.info("Started a fresh, empty configuration.")

NUM_BUILDINGS = 5

BUILDING_SETPOINTS = {
    "SFH_1": {
        "base_cooling": 22,   # °C
        "base_heating": 20.0  # °C
    },
    "SFH_2": {
        "base_cooling": 22,   # °C
        "base_heating": 20.0  # °C
    },

    "SFH_3": {
        "base_cooling": 22,   # °C
        "base_heating": 20.0  # °C
    },
    "SFH_4": {
        "base_cooling": 24,   # °C
        "base_heating": 20.0  # °C
    },
    "SFH_5": {
        "base_cooling": 24,   # °C
        "base_heating": 20.0  # °C
    },
}

# Common FCU parameters (shared across all buildings)
FCU_PARAMS = {
    "fan": {"rated_flow_m3s": 1, "rated_power_W": 1 * 1000},
    "fan_ctrl": {"ctrl_type": "linear"},
    "coil": {"epsilon": 0.8},
    "pump": {"rated_flow_m3s": 0.005, "rated_power_W": 0.005 * 100_000},
    "chiller": {"rated_capacity_W": 3500, "rated_cop": 4.5},
    "tower": {
        "rated_capacity_W": 3500,
        "rated_fan_power_W": 2000,
        "pump_power_per_flow": 1800,
        "min_approach_C": 3.0,
        "max_approach_C": 7.0
    }
}

# Common thermal model parameters
THERMAL_MODEL_PARAMS = {
    "para": {"Int_h": 8, "Ext_h": 14, "epochs": 20},
    "modeltype": "PI-modnn",
    "startday": 1,
    "trainday": 180,
    "testday": 1,
    "temp_unit": "C",
    "device": "cuda:0",
}


building_list = []

for i in range(1, NUM_BUILDINGS + 1):
    building_name = f"SFH_{i}"
    building_list.append(building_name)

    print(f"\n{'=' * 60}")
    print(f"Configuring {building_name}")
    print(f"{'=' * 60}")

    # Get setpoints for this building (with defaults if not specified)
    base_cooling = BUILDING_SETPOINTS.get(building_name, {}).get("base_cooling", 24.0)
    base_heating = BUILDING_SETPOINTS.get(building_name, {}).get("base_heating", 20.0)

    print(f"  Base Cooling Setpoint: {base_cooling}°C")
    print(f"  Base Heating Setpoint: {base_heating}°C")


    thermal_params = THERMAL_MODEL_PARAMS.copy()
    thermal_params.update({
        "datapath": os.path.join(PROJECT_ROOT_PATH, "data", "SFH", "BLDG", "clean", f"{building_name}.csv"),
        "save_name": building_name
    })

    cm.add_building_component(
        building_name, "thermal_zones", "zone0",
        parameters={
            "model_args": thermal_params,
            "model_path": os.path.join(
                PROJECT_ROOT_PATH, "examples", "Saved", building_name,
                "Trained_mdlEnco48_Deco96", "PI-modnn_180daysTest_on07-01.pth"
            ),
            "scaler_path": os.path.join(
                PROJECT_ROOT_PATH, "examples", "Scaler", building_name, "ModNN_scaler.pkl"
            ),
            "historical_data_path": os.path.join(
                PROJECT_ROOT_PATH, "data", "SFH", "BLDG", "clean", f"{building_name}.csv"
            ),
            "encoder_length": 48,
            "retrain": "Off",
            "simulation_start_time": "2023-08-01 00:00:00",
        },
        class_path="bestopt.env.modules.building.dynamic.ThermalDynamicsModule"
    )

    cm.add_building_component(
        building_name, "hvac_systems", "fcu",
        parameters=FCU_PARAMS,
        class_path="bestopt.env.modules.hvac.system.FCU.FCUModule"
    )

    controller_name = f"{building_name}_THERMAL_Supervisory"

    # Create controller parameters with building-specific setpoints
    controller_params = {
        "domain": "thermal",
        "type": "rule-based",
        "mode": "cooling",
        "base_cooling": base_cooling,
        "base_heating": base_heating,
        "precooling": {
            "degree": 0,
            "hours": 0
        }
    }

    cm.add_controller(
        controller_name,
        parameters=controller_params,
        class_path="bestopt.env.controllers.thermal.SupervisoryController"
    )

    print(f"  ✓ Added thermal zone, FCU, and controller for {building_name}")

print(f"\n{'=' * 60}")
print(f"✓ Successfully configured {NUM_BUILDINGS} buildings with FCU systems")
print(f"{'=' * 60}\n")

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
        "duration": 86400 * 3,
        "enable_history": True,
        "logging_level": "INFO",
        "simulation_start_time": "2023-08-01 00:00:00",
    },
    class_path="bestopt.environment.BestOptEnvironment"
)

cm.select_buildings(building_list)

# Select controllers for each building
for building_name in building_list:
    controller_name = f"{building_name}_THERMAL_Supervisory"
    cm.select_controller_for_building_domain(building_name, "thermal", controller_name)

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

print(f"\n{'=' * 60}")
print("BUILDING SETPOINT SUMMARY")
print(f"{'=' * 60}")
print(f"{'Building':<15} {'Cooling (°C)':<15} {'Heating (°C)':<15}")
print(f"{'-' * 60}")
for building_name in building_list:
    cool_sp = BUILDING_SETPOINTS.get(building_name, {}).get("base_cooling", 24.0)
    heat_sp = BUILDING_SETPOINTS.get(building_name, {}).get("base_heating", 20.0)
    print(f"{building_name:<15} {cool_sp:<15.1f} {heat_sp:<15.1f}")
print(f"{'=' * 60}\n")