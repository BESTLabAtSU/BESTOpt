"""
BEST_OPT - PV Module
Copyright (c) 2025 Zixin Jiang, BEST Lab, Syracuse University

This module defines the configuration for a PV system to be used within the BESTOPT framework.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

@dataclass
class PVConfig:
    """
    Configuration class for a PV system.

    Attributes:
        capacity_kw (float): Installed PV capacity in kilowatts.
        global_params (dict): Includes 'T_amb' (ambient temperature) and 'Sol' (solar radiation).
    """
    capacity_kw: float
    global_params: Optional[Dict[str, np.ndarray]] = None
    gen: Optional[np.ndarray] = field(init=False, default=None)
    pv_coefficient: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self):
        if self.global_params:
            self.gen = self._pv_generation()

    def _pv_generation(self) -> np.ndarray:
        """
        Calculate PV generation based on ambient temperature and solar radiation.
        Reference: https://doi.org/10.1016/j.apenergy.2022.119713, Section 2.1
        Can be replaced by any other module.

        Returns:
            np.ndarray: Estimated PV power generation profile (kW).
        """
        Isol = self.global_params['Sol']
        Tamb = self.global_params['T_amb']

        # PV performance parameters
        K = -3.7 / 1000  # Temperature coefficient (1/°C)
        Sol_ref = 1000  # Reference solar irradiance (W/m²)
        Temp_ref = 25  # Reference temperature (°C)

        self.pv_coefficient = (Isol / Sol_ref) * (1 + K * (Tamb + (0.0256 * Isol) - Temp_ref))
        return self.capacity_kw * self.pv_coefficient
