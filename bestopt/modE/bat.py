"""
BEST_OPT - Battery Module
Copyright (c) 2025 Zixin Jiang, BEST Lab, Syracuse University

This module defines the configuration for a Battery system to be used within the BESTOPT framework.
"""

from dataclasses import dataclass

@dataclass
class BatteryConfig:
    """
    Configuration class for a battery energy storage system.

    Attributes:
        capacity_kwh (float): Battery capacity in kilowatt-hours (kWh).
        c_rate (float): Maximum charge/discharge rate (per unit of capacity).
        charge_efficiency (float): Efficiency during charging [0-1].
        discharge_efficiency (float): Efficiency during discharging [0-1].
        min_soc (float): Minimum state of charge (SOC), as a fraction [0-1].
        max_soc (float): Maximum state of charge (SOC), as a fraction [0-1].
        initial_soc (float): Initial state of charge at the beginning of operation [0-1].
    """
    capacity_kwh: float
    c_rate: float # For example, 0.5C means the battery takes 2 hours to fully charge; 0.2C means 5 hours.
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    min_soc: float = 0.1
    max_soc: float = 0.9
    initial_soc: float = 0.5
