"""
BEST_OPT - PV Module
Copyright (c) 2025 Zixin Jiang, BEST Lab, Syracuse University

This module defines the configuration for a PV system to be used within the BESTOPT framework.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class PVConfig:
    """
    Configuration class for a PV system.

    Attributes:
        capacity_kw (float): Installed PV capacity in kilowatts.
        pv_gen (Optional[np.ndarray]): Optional array of PV power generation over time (in kW).
    """
    capacity_kw: float
    pv_gen: Optional[np.ndarray] = None
