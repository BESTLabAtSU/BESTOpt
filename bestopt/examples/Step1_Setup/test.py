"""
BESTOPT Configuration Manager Tutorial

This script demonstrates how to build configurations *from scratch* using ConfigurationManager.
It covers:
  1) Creating a manager with no pre-existing JSON file
  2) Adding buildings and their components (battery, PV, HVAC...)
  3) Adding controllers and disturbances
  4) Adding environment and grid
  5) Selecting which assets participate in a simulation
  6) Adjusting parameters for a specific run
  7) Validating, summarizing, and saving the configuration
"""

import logging
from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.data_structure import BatteryConfig, PVConfig, HVACConfig

# ------------------------------------------------------------------------------
# 0) Logging enabled so users can see what's happening
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ------------------------------------------------------------------------------
# 1) Create a ConfigurationManager (start from scratch)
# ------------------------------------------------------------------------------
cm = ConfigurationManager()
logging.info("Started a fresh, empty configuration.")

# ------------------------------------------------------------------------------
# 2) Add buildings and components
#    Pattern: add_building_component(building_id, component_type, component_id, ...)
#    Common groups: 'batteries', 'pv_systems', 'hvac_systems', 'thermal_zones', etc.
# ------------------------------------------------------------------------------

# --- Office Building 1 ---------------------------------------------------------
# Example A: Battery seeded from a dataclass + a few overrides
cm.add_building_component(
    building_id="office_building_1",
    component_type="batteries",
    component_id="main_battery",
    dataclass_obj=BatteryConfig(battery_capacity=100.0, battery_c_rate=0.5),
    parameters={"battery_soc_min": 0.2},  # override a few fields
    class_path="bestopt.modules.battery.Battery"
)

# Example B: PV defined purely via parameters (no dataclass seed)
cm.add_building_component(
    "office_building_1", "pv_systems", "rooftop_pv",
    parameters={"pv_capacity": 50.0, "efficiency": 0.22, "tilt_angle": 30},
    class_path="bestopt.modules.pv.PVSystem"
)

# Example C: Central HVAC via parameters
cm.add_building_component(
    "office_building_1", "hvac_systems", "central_hvac",
    parameters={"cooling_capacity": 80.0, "heating_capacity": 60.0, "cop_cooling": 3.5},
    class_path="bestopt.modules.hvac.CentralHVAC"
)

# Example D: A thermal zone (geometry/thermal mass info)
cm.add_building_component(
    "office_building_1", "thermal_zones", "main_zone",
    parameters={"floor_area": 1000.0, "volume": 3000.0, "thermal_mass": 50000.0},
    class_path="bestopt.modules.thermal.ThermalZone"
)

# --- Office Building 2 (multiple batteries & PV arrays) ------------------------
cm.add_building_component(
    "office_building_2", "batteries", "battery_bank_1",
    parameters={"battery_capacity": 150.0, "battery_c_rate": 0.33},
    class_path="bestopt.modules.battery.Battery"
)
cm.add_building_component(
    "office_building_2", "batteries", "battery_bank_2",
    parameters={"battery_capacity": 150.0, "battery_c_rate": 0.33},
    class_path="bestopt.modules.battery.Battery"
)
cm.add_building_component(
    "office_building_2", "pv_systems", "south_array",
    parameters={"pv_capacity": 75.0, "efficiency": 0.21, "azimuth": 180},
    class_path="bestopt.modules.pv.PVSystem"
)
cm.add_building_component(
    "office_building_2", "pv_systems", "east_array",
    parameters={"pv_capacity": 40.0, "efficiency": 0.20, "azimuth": 90},
    class_path="bestopt.modules.pv.PVSystem"
)
cm.add_building_component(
    "office_building_2", "hvac_systems", "vrf_system",
    parameters={"total_capacity": 120.0, "number_of_zones": 6, "cop_cooling": 4.0},
    class_path="bestopt.modules.hvac.VRFSystem"
)

# ------------------------------------------------------------------------------
# 3) Add controllers (you can define multiple and pick per-building later)
#    Pattern: add_controller(controller_name, parameters, class_path)
# ------------------------------------------------------------------------------
cm.add_controller(
    "mpc_controller",
    parameters={
        "prediction_horizon": 24,
        "control_horizon": 6,
        "dt_minutes": 15,
        "objective_weights": {"cost": 1.0, "comfort": 0.8}
    },
    class_path="bestopt.controllers.mpc.MPCController"
)

cm.add_controller(
    "rule_based_controller",
    parameters={
        "battery_soc_high_threshold": 0.8,
        "battery_soc_low_threshold": 0.2,
        "peak_hours": [16, 17, 18, 19, 20]
    },
    class_path="bestopt.controllers.rule_based.RuleBasedController"
)

# ------------------------------------------------------------------------------
# 4) Add exogenous signals / disturbances (e.g., weather, price)
#    Pattern: add_disturbance(name, parameters, class_path)
# ------------------------------------------------------------------------------
cm.add_disturbance(
    "weather",
    parameters={
        "data_source": "epw_file",
        "file_path": "/data/weather/syracuse_ny.epw",
        "interpolation_method": "linear"
    },
    class_path="bestopt.disturbances.weather.WeatherDisturbance"
)

cm.add_disturbance(
    "electricity_prices",
    parameters={
        "tariff_type": "time_of_use",
        "peak_price": 0.25,
        "off_peak_price": 0.12,
        "peak_hours": [16, 17, 18, 19, 20],
        "demand_charge": 15.0
    },
    class_path="bestopt.disturbances.pricing.ElectricityPricing"
)

# ------------------------------------------------------------------------------
# 5) Add environment and grid (global to the simulation)
#    Pattern: add_environment(...), add_grid(...)
# ------------------------------------------------------------------------------
cm.add_environment(
    parameters={
        "resolution": 900,   # 15 min in seconds
        "duration": 86400,   # 24 hours in seconds
        "enable_history": True,
        "logging_level": "INFO"
    },
    class_path="bestopt.environment.BestOptEnvironment"
)

cm.add_grid(
    parameters={
        "base_voltage": 4160.0,
        "max_import_capacity": 2000.0,
        "max_export_capacity": 1500.0,
        "enable_islanding": False
    },
    class_path="bestopt.modules.grid.DistributionGrid"
)

print("✓ Added environment and grid")

# ------------------------------------------------------------------------------
# 6) Configure a specific simulation run by selecting assets
#    - pick which buildings participate
#    - assign controllers per building
#    - choose disturbances, environment, and grid
# ------------------------------------------------------------------------------
print("\nConfiguring simulation...")

# Select buildings for this scenario
cm.select_buildings(["office_building_1", "office_building_2"])

# Assign controllers (per-building selection)
cm.select_controller_for_building("office_building_1", "mpc_controller")
cm.select_controller_for_building("office_building_2", "rule_based_controller")

# Select disturbances and global elements
cm.select_disturbances(["weather", "electricity_prices"])
cm.select_environment()
cm.select_grid()

# ------------------------------------------------------------------------------
# 7) Adjust parameters for this specific run (optional, convenient tweaking)
#    You can reach into a component or controller and change values ad-hoc.
# ------------------------------------------------------------------------------
cm.adjust_building_component_parameter( "office_building_1", "batteries",
                                        "main_battery", "battery_capacity", 120.0)

cm.adjust_building_controller_parameter("office_building_1", "prediction_horizon", 36)

print("✓ Configured simulation settings")

# ------------------------------------------------------------------------------
# 8) Validate and summarize current selection
# ------------------------------------------------------------------------------
warnings = cm.validate_configuration()
if warnings:
    print(f"\n⚠ Warnings found: {warnings}")
else:
    print("\n✓ Configuration validation passed")

cm.print_summary()

# ------------------------------------------------------------------------------
# 9) Save the final selected configuration for this scenario
# ------------------------------------------------------------------------------
cm.save_final_configuration("simulation_config.json")
print("✓ Saved simulation configuration as simulation_config.json")
