"""
BEST_OPT - Thermal Storage Module
Copyright (c) 2025 Zixin Jiang, BEST Lab, Syracuse University

This module defines the configuration for a Thermal storage system to be used within the BESTOPT framework.
"""

from dataclasses import dataclass

# @TODO
# the current version controls charging/discharging power,
# which should be replaced by inlet/outlet temperature when HVAC module is ready
@dataclass
class TESConfig:
    """Thermal Storage Configuration"""
    capacity_kwh: float  # Storage capacity in kWh
    c_rate: float  # charging/discharging speed
    charge_efficiency: float = 0.85  # Charging efficiency
    discharge_efficiency: float = 0.85  # Discharging efficiency
    min_soc: float = 0.0  # Minimum state of charge (00%)
    max_soc: float = 1.0  # Maximum state of charge (100%)
    initial_soc: float = 0.5  # Initial state of charge