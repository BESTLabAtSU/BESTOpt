"""
BEST_OPT - Global Module
Copyright (c) 2025 Zixin Jiang, BEST Lab, Syracuse University

This module defines the global variables to be used within the BESTOPT framework.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class GlobalConfig:
    """
    Configuration class for simulation environment settings.

    Attributes:
        T_amb (np.ndarray): Outdoor ambient temperature (°C), time series.
        Sol (np.ndarray): Solar radiation (W/m2), time series.
        TOU (np.ndarray): Time-of-Use electricity price (USD/W), time series.
        Load_thermal (np.ndarray): Thermal load (kW), time series.
        Load_ele (np.ndarray): Electric load (kW), time series.
        Res (int): Data resolution in minutes (e.g., 15 for 15-minute intervals).
        cpus (int): Number of CPU cores to allocate for parallel processing.
    """
    T_amb: np.ndarray
    Sol: np.ndarray
    TOU: np.ndarray
    Load_thermal: np.ndarray
    Load_ele: np.ndarray
    cop: np.ndarray
    Res: int
    cpus: int
