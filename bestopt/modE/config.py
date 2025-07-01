"""
BEST_OPT - Wrap-Up Module
Copyright (c) 2025 Zixin Jiang, BEST Lab, Syracuse University

This module wraps all electrical components for the BESTOPT framework.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from .globals import GlobalConfig
from .pv import PVConfig
from .bat import BatteryConfig
from .ev import EVConfig
from .bldg import BuildingConfig

@dataclass
class Config:
    """
    Wrapper class to configure and generate buildings for BEST_OPT framework.

    Attributes:
        global_config (GlobalConfig): Global environmental and simulation settings.
        pv_coefficient (np.ndarray): Pre-calculated PV generation coefficient.
    """
    global_config: GlobalConfig

    def __post_init__(self):
        self.pv_coefficient = self.calculate_pv_coefficient()

    def calculate_pv_coefficient(self) -> np.ndarray:
        """
        Pre-calculate PV coefficient based on ambient temperature and solar radiation.
        Reference: https://doi.org/10.1016/j.apenergy.2022.119713, Section 2.1
        Can be replaced by any other module

        Returns:
            np.ndarray: Coefficient array used to scale PV size to generation.
        """
        Isol = self.global_config.Sol
        Tamb = self.global_config.T_amb

        # PV performance parameters
        K = -3.7 / 1000  # Temperature coefficient
        Sol_ref = 1000   # Reference solar irradiance (W/m²)
        Temp_ref = 25    # Reference temperature (°C)

        return (Isol / Sol_ref) * (1 + K * (Tamb + (0.0256 * Isol) - Temp_ref))

    def get_pv_generation(self, pv_size: float) -> np.ndarray:
        """
        Calculate PV generation based on pre-calculated coefficient and size.

        Args:
            pv_size (float): PV system capacity in kW.

        Returns:
            np.ndarray: PV generation profile in kW.
        """
        if self.pv_coefficient is None:
            raise ValueError("PV coefficient not calculated correctly")
        return pv_size * self.pv_coefficient

    def create_building(
        self,
        load: np.ndarray,
        pv_params: Optional[dict] = None,
        battery_params: Optional[dict] = None,
        ev_params: Optional[dict] = None,
    ) -> BuildingConfig:
        """
        Create a BuildingConfig instance with optional PV, battery, and EV systems.

        Args:
            load (np.ndarray): Load profile of the building.
            pv_params (dict, optional): Parameters for PVConfig.
            battery_params (dict, optional): Parameters for BatteryConfig.
            ev_params (dict, optional): Parameters for EVConfig.

        Returns:
            BuildingConfig: Fully constructed building configuration.
        """
        if pv_params is not None:
            pv_size = pv_params['capacity_kw']
            pv_gen = self.get_pv_generation(pv_size)
            pv = PVConfig(**{**pv_params, 'pv_gen': pv_gen})
        else:
            pv = None

        battery = BatteryConfig(**battery_params) if battery_params else None
        ev = EVConfig(**ev_params) if ev_params else None

        return BuildingConfig(load=load, pv=pv, battery=battery, ev=ev)

    def create_multiple_buildings(self, config_list: List[dict]) -> List[BuildingConfig]:
        """
        Create multiple buildings from a list of config dictionaries.

        Args:
            config_list (List[dict]): Each dict must have at least a 'load' key, optional for PV-Battery-EV.

        Returns:
            List[BuildingConfig]: A list of building configurations.
        """
        return [self.create_building(**params) for params in config_list]
