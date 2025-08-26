"""
BEST_OPT - Building Module
Copyright (c) 2025 Zixin Jiang, BEST Lab, Syracuse University

This module defines the configuration for a Building with PV-Battery-EV-TES to be used within the BESTOPT framework.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from .pv import PVConfig
from .bat import BatteryConfig
from .ev import EVConfig
from .tes import TESConfig
from .globals import GlobalConfig

@dataclass
class BuildingConfig:
    """
    Configuration class for a building energy system.

    Attributes:
        globals (Optional[GlobalConfig]): Configuration for the global variables.
        pv (Optional[PVConfig]): Configuration for the PV system.
        battery (Optional[BatteryConfig]): Configuration for the battery system.
        ev (Optional[EVConfig]): Configuration for the EV.
        tes (Optional[TESConfig]): Configuration for the TES system.
    """
    globals: Optional[GlobalConfig] = None
    pv: Optional[PVConfig] = None
    battery: Optional[BatteryConfig] = None
    ev: Optional[EVConfig] = None
    tes: Optional[TESConfig] = None

    def get_params(self,
            # --- PV ---
            pv_capacity_kw=20,

            # --- Battery ---
            battery_capacity_kwh=100.0,
            battery_c_rate=0.25,
            battery_charge_eff=0.95,
            battery_discharge_eff=0.95,
            battery_min_soc=0.1,
            battery_max_soc=0.9,
            battery_initial_soc=0.5,

            # --- EV ---
            ev_capacity_kwh=50.0,
            ev_c_rate=0.25,
            ev_charge_eff=0.95,
            ev_discharge_eff=0.95,
            ev_min_soc=0.1,
            ev_max_soc=1.0,
            ev_departure_time=8 * 4,
            ev_arrival_time=18 * 4,
            ev_arrival_soc=0.6,
            ev_required_departure_soc=0.95,

            # --- TES ---
            tes_capacity_kwh=50.0,
            tes_c_rate=0.25,
            tes_charge_eff=0.85,
            tes_discharge_eff=0.85,
            tes_min_soc=0.0,
            tes_max_soc=1.0,
            tes_initial_soc=0.5,

            # --- Global ---
            T_amb=None,
            Sol=None,
            TOU=None,
            Load_thermal=None,
            Load_ele=None,
            cop=None,
            Res=15,
            cpus=1
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Returns parameter dictionaries for global, PV, battery, EV, and TES systems.
        """
        self.pv_params = {
            'capacity_kw': pv_capacity_kw
        } if pv_capacity_kw else None

        self.battery_params = {
            'capacity_kwh': battery_capacity_kwh,
            'c_rate': battery_c_rate,
            'charge_efficiency': battery_charge_eff,
            'discharge_efficiency': battery_discharge_eff,
            'min_soc': battery_min_soc,
            'max_soc': battery_max_soc,
            'initial_soc': battery_initial_soc
        } if battery_capacity_kwh else None

        self.ev_params = {
            'capacity_kwh': ev_capacity_kwh,
            'c_rate': ev_c_rate,
            'charge_efficiency': ev_charge_eff,
            'discharge_efficiency': ev_discharge_eff,
            'min_soc': ev_min_soc,
            'max_soc': ev_max_soc,
            'departure_time': ev_departure_time,
            'arrival_time': ev_arrival_time,
            'arrival_soc': ev_arrival_soc,
            'required_departure_soc': ev_required_departure_soc
        } if ev_capacity_kwh else None

        self.tes_params = {
            'capacity_kwh': tes_capacity_kwh,
            'c_rate': tes_c_rate,
            'charge_efficiency': tes_charge_eff,
            'discharge_efficiency': tes_discharge_eff,
            'min_soc': tes_min_soc,
            'max_soc': tes_max_soc,
            'initial_soc': tes_initial_soc
        } if tes_capacity_kwh else None

        self.global_params = {
            'T_amb': T_amb,
            'Sol': Sol,
            'TOU': TOU,
            'Load_thermal': Load_thermal,
            'Load_ele': Load_ele,
            'cop': cop,
            'Res': Res,
            'cpus': cpus
        } if T_amb is not None and Sol is not None else None


    def create_building(self):
        """
        Factory function to create a BuildingConfig instance.

        Args:
            global_params (dict): Shared global parameters.
            pv_params (dict): Parameters for PV system.
            battery_params (dict): Parameters for battery system.
            ev_params (dict): Parameters for EV system.
            tes_params (dict): Parameters for TES system.

        Returns:
            BuildingConfig: Configured building energy system.
        """
        self.pv = None
        if self.pv_params:
            pv_params = {**self.pv_params, 'global_params': self.global_params}
            self.pv = PVConfig(**pv_params)

        self.battery = BatteryConfig(**self.battery_params) if self.battery_params else None
        self.ev = EVConfig(**self.ev_params) if self.ev_params else None
        self.tes = TESConfig(**self.tes_params) if self.tes_params else None
        self.globals = GlobalConfig(**self.global_params) if self.global_params else None

