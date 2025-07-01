"""
BEST_OPT - Building Module
Copyright (c) 2025 Zixin Jiang, BEST Lab, Syracuse University

This module defines the configuration for a Building with PV-Battery-EV to be used within the BESTOPT framework.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from .pv import PVConfig
from .bat import BatteryConfig
from .ev import EVConfig

@dataclass
class BuildingConfig:
    """
    Configuration class for a building energy system.

    Attributes:
        load (np.ndarray): Building electrical load profile (kW), time series.
        pv (Optional[PVConfig]): Configuration for the PV system.
        battery (Optional[BatteryConfig]): Configuration for the battery system.
        ev (Optional[EVConfig]): Configuration for the EV.
    """
    load: np.ndarray
    pv: Optional[PVConfig] = None
    battery: Optional[BatteryConfig] = None
    ev: Optional[EVConfig] = None
