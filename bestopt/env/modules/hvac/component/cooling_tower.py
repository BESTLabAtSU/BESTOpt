"""
Cooling tower module.
"""

from typing import Dict, Any
import numpy as np

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import (
    ThermalAction,
    CoolingTowerState,
    ChillerState,
    WeatherData
)


class CoolingTowerModule(BaseModule):
    """
    Cooling Tower module that rejects condenser heat and estimates fan power.

    Inputs:
        - ChillerState.cooling_W                    : Chiller cooling load [W]
        - ChillerState.cop                          : Chiller COP [-]
        - ThermalAction.condenser_temp_c_sp         : CW supply temp setpoint [°C] (to chiller)
        - WeatherData.outdoor_wet_bulb_temperature  : Outdoor wet-bulb temp [°C]

    Outputs (to CoolingTowerState):
        - heat_rejected_W   : Q_cooling + Q_electric [W]
        - cw_supply_temp_c  : To chiller [°C]
        - cw_return_temp_c  : From chiller [°C]
        - cw_flow_m3s       : Estimated condenser water flow rate [m³/s]
        - fan_power_W       : Cooling tower fan power [W]
        - energy_J_cum      : Cumulative fan energy use [J]
    """

    def __init__(self, config: Dict[str, Any], name: str = "cooling_tower"):
        super().__init__(config, name)

        self.rated_capacity_W = float(config.get("rated_capacity_W", 200_000.0))  # 200 kW
        self.rated_fan_power_W = float(config.get("rated_fan_power_W", 5000.0))   # at full load
        self.min_approach_C = float(config.get("min_approach_C", 3.0))            # design approach
        self.max_approach_C = float(config.get("max_approach_C", 7.0))            # part-load approach

        self.temp_range_C = float(config.get("temp_range_C", 5.0))  # CW ΔT
        self.rho = 1000.0
        self.cp = 4180.0

    def initialize(self):
        self._initialized = True

    def step(
        self,
        state: "CoolingTowerState",
        action: "ThermalAction",
        chiller_state: "ChillerState",
        weather: "WeatherData",
        timestep: float
    ) -> Dict[str, Any]:

        # === Inputs ===
        q_cooling_W = chiller_state.cooling_W
        cop = max(chiller_state.cop, 0.1)
        wet_bulb_temp_c = float(getattr(weather, "outdoor_wet_bulb_temperature", 25.0))
        cw_supply_sp_c = float(getattr(action, "condenser_temp_c_sp", 30.0))

        # === Heat rejected to tower ===
        power_W = q_cooling_W / cop
        q_reject_W = q_cooling_W + power_W

        # === Load ratio and approach temp ===
        load_ratio = np.clip(q_reject_W / self.rated_capacity_W, 0.0, 1.0)
        approach = self.min_approach_C + (1.0 - load_ratio) * (self.max_approach_C - self.min_approach_C)

        # === CW temps ===
        cw_supply_temp_c = cw_supply_sp_c                      # to chiller
        cw_return_temp_c = max(wet_bulb_temp_c+approach, cw_supply_sp_c) + self.temp_range_C           # from chiller

        # === CW flow rate (Q = m*cp*dT) ===
        m_dot = q_reject_W / (self.cp * self.temp_range_C)
        cw_flow_m3s = m_dot / self.rho

        # === Fan power using cubic part-load model ===
        fan_power_W = self.rated_fan_power_W * load_ratio**3
        energy_J = fan_power_W * timestep

        # === Update state ===
        state.heat_rejected_W = q_reject_W
        state.cw_supply_temp_c = cw_supply_temp_c
        state.cw_return_temp_c = cw_return_temp_c
        state.cw_flow_m3s = cw_flow_m3s
        state.fan_power_W = fan_power_W
        state.energy_J_cum += energy_J

        # === Log ===
        self._record_state({
            "Q_rejected_W": q_reject_W,
            "cw_supply_temp_C": cw_supply_temp_c,
            "cw_return_temp_C": cw_return_temp_c,
            "cw_flow_m3s": cw_flow_m3s,
            "fan_power_W": fan_power_W,
            "energy_J": energy_J,
            "approach_C": approach,
            "load_ratio": load_ratio
        })

        return {}

    def reset(self):
        self._state_history.clear()
        self._initialized = False
        self.initialize()
