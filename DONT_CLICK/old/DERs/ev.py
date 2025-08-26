"""
BEST_OPT - EV Module
Copyright (c) 2025 Zixin Jiang, BEST Lab, Syracuse University

This module defines the configuration for an Electric Vehicle to be used within the BESTOPT framework.
"""

from dataclasses import dataclass

@dataclass
class EVConfig:
    """
    Configuration class for an Electric Vehicle.

    Attributes:
        capacity_kwh (float): EV capacity in kilowatt-hours (kWh).
        c_rate (float): Maximum charge/discharge rate (per unit of capacity).
        arrival_time (int):  Arrival time step (not clock time). For example, with 15-minute resolution, 8:00 AM = timestep 32.
        departure_time (int): Time step when the EV departs.
        arrival_soc (float): State of charge (SOC) upon arrival [0–1].
        required_departure_soc (float): Minimum required SOC at departure [0–1].
        charge_efficiency (float): Efficiency during charging [0–1].
        discharge_efficiency (float): Efficiency during discharging [0–1].
        min_soc (float): Minimum allowable SOC [0–1].
        max_soc (float): Maximum allowable SOC [0–1].
    """
    capacity_kwh: float
    c_rate: float  # For example, 0.5C means the battery takes 2 hours to fully charge; 0.2C means 5 hours.
    arrival_time: int # use TIMESTEP not CLOCKTIME, please check above
    departure_time: int
    arrival_soc: float
    required_departure_soc: float
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    min_soc: float = 0.1
    max_soc: float = 1.0
