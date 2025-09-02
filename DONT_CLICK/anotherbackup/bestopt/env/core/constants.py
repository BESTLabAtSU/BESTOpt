"""
This module provides basic physical constants and unit conversions.
"""

import math
from typing import Dict, Final

# Air properties (standard conditions)
AIR_DENSITY: Final[float] = 1.225  # kg/m³ at 15°C, 101.325 kPa
AIR_SPECIFIC_HEAT: Final[float] = 1005.0  # J/(kg·K)
AIR_THERMAL_CONDUCTIVITY: Final[float] = 0.0257  # W/(m·K)

# Water properties
WATER_DENSITY: Final[float] = 1000.0  # kg/m³
WATER_SPECIFIC_HEAT: Final[float] = 4186.0  # J/(kg·K)
WATER_THERMAL_CONDUCTIVITY: Final[float] = 0.6  # W/(m·K)

# Solar and atmospheric constants
SOLAR_CONSTANT: Final[float] = 1361.0  # W/m²
STEFAN_BOLTZMANN: Final[float] = 5.67e-8  # W/(m²·K⁴)
STANDARD_ATMOSPHERIC_PRESSURE: Final[float] = 101325.0  # Pa
STANDARD_TEMPERATURE: Final[float] = 273.15  # K (0°C)
STANDARD_GRAVITY: Final[float] = 9.80665  # m/s²


# Temperature conversions
def celsius_to_kelvin(temp_c: float) -> float:
    """Convert Celsius to Kelvin."""
    return temp_c + 273.15


def kelvin_to_celsius(temp_k: float) -> float:
    """Convert Kelvin to Celsius."""
    return temp_k - 273.15


def fahrenheit_to_celsius(temp_f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (temp_f - 32.0) * 5.0 / 9.0


def celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9.0 / 5.0 + 32.0


# Energy conversions
KWH_TO_J: Final[float] = 3.6e6  # J/kWh
J_TO_KWH: Final[float] = 1.0 / KWH_TO_J
BTU_TO_J: Final[float] = 1055.06  # J/BTU
J_TO_BTU: Final[float] = 1.0 / BTU_TO_J
THERM_TO_J: Final[float] = 1.05506e8  # J/therm (US)

# Power conversions
HP_TO_W: Final[float] = 745.7  # W/hp
W_TO_HP: Final[float] = 1.0 / HP_TO_W
TON_TO_W: Final[float] = 3516.85  # W/ton (refrigeration)

# Time conversions
SECONDS_PER_HOUR: Final[float] = 3600.0
HOURS_PER_DAY: Final[float] = 24.0
DAYS_PER_YEAR: Final[float] = 365  # @TODO leap year
SECONDS_PER_DAY: Final[float] = SECONDS_PER_HOUR * HOURS_PER_DAY

# Simulation default
SIMULATION_DEFAULTS: Final[Dict[str, float]] = {
    'timestep_minutes': 15,
}

# # Grid and power quality
# NOMINAL_VOLTAGE_RESIDENTIAL: Final[float] = 240.0  # V (US split-phase)
# NOMINAL_FREQUENCY: Final[float] = 60.0  # Hz (US)
# POWER_FACTOR_TYPICAL: Final[float] = 0.95
#
# # Battery technology constants
# BATTERY_TECH: Final[Dict[str, Dict[str, float]]] = {
#     'lithium_ion': {
#         'energy_density': 250.0,  # Wh/kg
#         'power_density': 1000.0,  # W/kg
#         'cycle_life': 5000.0,  # cycles
#         'efficiency': 0.95,  # round-trip efficiency
#         'self_discharge': 0.05,  # %/day
#         'temp_coefficient': -0.005,  # capacity change per °C
#     },
#     'lead_acid': {
#         'energy_density': 40.0,
#         'power_density': 180.0,
#         'cycle_life': 1000.0,
#         'efficiency': 0.80,
#         'self_discharge': 0.3,
#         'temp_coefficient': -0.008,
#     }
# }
#
# # PV technology constants
# PV_TECH: Final[Dict[str, Dict[str, float]]] = {
#     'silicon_crystalline': {
#         'efficiency_stc': 0.20,  # % at STC
#         'temp_coefficient': -0.004,  # efficiency change per °C
#         'irradiance_coefficient': 0.12,  # efficiency change with irradiance
#         'degradation_rate': 0.005,  # %/year
#     },
#     'silicon_thin_film': {
#         'efficiency_stc': 0.12,
#         'temp_coefficient': -0.0025,
#         'irradiance_coefficient': 0.08,
#         'degradation_rate': 0.006,
#     }
# }
#
# # Standard Test Conditions (STC) for PV
# PV_STC: Final[Dict[str, float]] = {
#     'irradiance': 1000.0,  # W/m²
#     'cell_temperature': 25.0,  # °C
#     'air_mass': 1.5,
# }
#
#
# # Predicted Mean Vote (PMV) model constants
# PMV_CONSTANTS: Final[Dict[str, float]] = {
#     'stefan_boltzmann': STEFAN_BOLTZMANN,
#     'evaporation_coefficient': 3.05e-3,  # kg/(Pa·s·m²)
#     'respiration_coefficient': 1.7e-5,  # m³/(s·m²)
#     'respiration_temp_coefficient': 5.73e-3,  # K⁻¹
#     'convection_coefficient': 3.96,  # W/(m²·K)
# }
#
# # Metabolic rates for different activities
# METABOLIC_RATES: Final[Dict[str, float]] = {
#     'sleeping': 0.7,  # met
#     'seated_quiet': 1.0,  # met (reference)
#     'standing_light': 1.2,
#     'office_work': 1.2,
#     'walking_slow': 2.0,
#     'walking_normal': 3.0,
#     'light_exercise': 4.0,
#     'heavy_work': 5.0,
# }
#
# # Clothing insulation values
# CLOTHING_INSULATION: Final[Dict[str, float]] = {
#     'nude': 0.0,  # clo
#     'underwear': 0.1,
#     'light_summer': 0.5,
#     'typical_indoor': 1.0,  # clo (reference)
#     'winter_indoor': 1.5,
#     'heavy_winter': 2.0,
# }
#
# # Typical utility rate structures (example values)
# UTILITY_RATES: Final[Dict[str, Dict[str, float]]] = {
#     'residential_tiered': {
#         'tier1_rate': 0.12,  # $/kWh
#         'tier1_limit': 500.0,  # kWh/month
#         'tier2_rate': 0.18,
#         'tier2_limit': 1000.0,
#         'tier3_rate': 0.25,
#     },
#     'time_of_use': {
#         'peak_rate': 0.30,  # $/kWh
#         'mid_peak_rate': 0.18,
#         'off_peak_rate': 0.10,
#     },
#     'demand_charges': {
#         'summer_demand': 20.0,  # $/kW
#         'winter_demand': 15.0,
#         'ratchet_factor': 0.8,  # fraction of peak demand
#     }
# }
#
# # Carbon emission factors
# CARBON_FACTORS: Final[Dict[str, float]] = {
#     'grid_average_us': 0.92,  # lbCO2/kWh
#     'natural_gas': 11.7,  # lbCO2/therm
#     'propane': 12.7,  # lbCO2/gallon
#     'fuel_oil': 22.4,  # lbCO2/gallon
#     'coal': 2.23,  # lbCO2/kWh
#     'nuclear': 0.0,  # lbCO2/kWh
#     'renewable': 0.0,  # lbCO2/kWh
# }
#
#
# # Typical simulation parameters

#
# # Numerical stability limits
# NUMERICAL_LIMITS: Final[Dict[str, float]] = {
#     'min_temperature': -50.0,  # °C
#     'max_temperature': 100.0,  # °C
#     'min_humidity': 0.0,  # %
#     'max_humidity': 100.0,  # %
#     'min_power': -1e6,  # W
#     'max_power': 1e6,  # W
#     'min_soc': 0.0,  # %
#     'max_soc': 100.0,  # %
# }
#
#
# def psychrometric_saturation_pressure(temp_c: float) -> float:
#     """Calculate saturation vapor pressure using Antoine equation.
#
#     Args:
#         temp_c: Temperature in Celsius
#
#     Returns:
#         Saturation vapor pressure in Pa
#     """
#     temp_k = celsius_to_kelvin(temp_c)
#     # Antoine equation for water vapor
#     log_p = 8.07131 - 1730.63 / (temp_k - 39.724)
#     return 10 ** log_p * 133.322  # Convert mmHg to Pa
#
#
# def air_density_correction(temp_c: float, pressure_pa: float = STANDARD_ATMOSPHERIC_PRESSURE) -> float:
#     """Calculate air density at given temperature and pressure.
#
#     Args:
#         temp_c: Temperature in Celsius
#         pressure_pa: Pressure in Pascal
#
#     Returns:
#         Air density in kg/m³
#     """
#     temp_k = celsius_to_kelvin(temp_c)
#     gas_constant_air = 287.0  # J/(kg·K)
#     return pressure_pa / (gas_constant_air * temp_k)
