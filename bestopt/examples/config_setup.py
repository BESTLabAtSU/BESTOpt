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
from bestopt.env.core.data_structure import BatteryConfig

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
    class_path="bestopt.env.modules.ders.battery.BatteryModule"
)

# Example B: PV defined purely via parameters (no dataclass seed)
cm.add_building_component(
    "office_building_1", "pv_systems", "rooftop_pv",
    parameters={"pv_capacity": 50.0, "efficiency": 0.22, "tilt_angle": 30},
    class_path="bestopt.env.modules.ders.pv.PVModule"
)

# Example C: Central HVAC via parameters
cm.add_building_component(
    "office_building_1", "hvac_systems", "ideaHVAC",
    parameters={"cooling_capacity": 80.0, "heating_capacity": 60.0, "cop_cooling": 3.5},
    class_path="bestopt.env.modules.hvac.ideal.HVACModule"
)

# Example D: A thermal zone (geometry/thermal mass info)
cm.add_building_component(
    "office_building_1", "thermal_zones", "main_zone",
    parameters={"floor_area": 1000.0, "volume": 3000.0, "thermal_mass": 50000.0},
    class_path="bestopt.env.modules.building.dynamic.ThermalDynamicsModule"
)

# --- Office Building 2 (multiple batteries & PV arrays) ------------------------
cm.add_building_component(
    "office_building_2", "batteries", "battery_bank_1",
    parameters={"battery_capacity": 150.0, "battery_c_rate": 0.33},
    class_path="bestopt.env.modules.ders.battery.BatteryModule"
)
cm.add_building_component(
    "office_building_2", "batteries", "battery_bank_2",
    parameters={"battery_capacity": 150.0, "battery_c_rate": 0.33},
    class_path="bestopt.env.modules.ders.battery.BatteryModule"
)
cm.add_building_component(
    "office_building_2", "pv_systems", "south_array",
    parameters={"pv_capacity": 75.0, "efficiency": 0.21, "azimuth": 180},
    class_path="bestopt.env.modules.ders.pv.PVModule"
)
cm.add_building_component(
    "office_building_2", "pv_systems", "east_array",
    parameters={"pv_capacity": 40.0, "efficiency": 0.20, "azimuth": 90},
    class_path="bestopt.env.modules.ders.pv.PVModule"
)

# ------------------------------------------------------------------------------
# 3) Add controllers (building-domain-controllers)
# ------------------------------------------------------------------------------
# One controller for one domain per building
cm.add_controller(
    "electrical_rb",
    parameters={
        "domain": "electrical",
        "prediction_horizon": 24,
        "objectives": ["cost_minimization", "peak_shaving"]
    },
    class_path="bestopt.env.controllers.electrical.RuleBased"
)

cm.add_controller(
    "thermal_rb",
    parameters={
        "domain": "thermal",
        "prediction_horizon": 12,
        "comfort_bounds": {"min": 20, "max": 24}
    },
    class_path="bestopt.env.controllers.thermal.RuleBased"
)

cm.add_controller(
    "water_rb",
    parameters={
        "domain": "water",
        "tank_temp_range": [55, 65]
    },
    class_path="bestopt.env.controllers.water.RuleBased"
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
    class_path="bestopt.env.disturbances.weather.WeatherModule"
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
    class_path="bestopt.env.disturbances.price.PriceModule"
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
    class_path="bestopt.env.modules.grid.DistributionGrid"
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
cm.select_controller_for_building_domain("office_building_1", "electrical", "electrical_rb")
cm.select_controller_for_building_domain("office_building_1", "thermal", "thermal_rb")
cm.select_controller_for_building_domain("office_building_1", "water", "water_rb")

cm.select_controller_for_building_domain("office_building_2", "electrical", "electrical_rb")
cm.select_controller_for_building_domain("office_building_2", "thermal", "thermal_rb")
cm.select_controller_for_building_domain("office_building_2", "water", "water_rb")

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

cm.adjust_building_controller_parameter("office_building_1", "electrical",
                                        "prediction_horizon", 36)

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
cm.save_final_configuration("config_setup.json")
print("✓ Saved simulation configuration as simulation_config.json")
