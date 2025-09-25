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
    Cooling Tower module that rejects condenser heat and calculates
    both fan power and condenser pump power.

    Inputs:
        - ChillerState.cooling_W                  : Chiller cooling load [W]
        - ChillerState.cop                        : Chiller COP [-]
        - ThermalAction.condenser_temp_c_sp       : CW supply temp setpoint [°C] (to chiller)
        - WeatherData.outdoor_wet_bulb_temperature: Outdoor wet-bulb temp [°C]

    Outputs (to CoolingTowerState):
        - heat_rejected_W       : Total heat rejected [W]
        - cw_supply_temp_c      : To chiller [°C]
        - cw_return_temp_c      : From chiller [°C]
        - cw_flow_m3s           : Estimated condenser water flow rate [m³/s]
        - fan_power_W           : Tower fan power [W]
        - pump_power_W          : Condenser pump power [W]
        - energy_J_cum          : Cumulative energy (fan + pump) [J]
    """

    def __init__(self, config: Dict[str, Any], name: str = "cooling_tower"):
        super().__init__(config, name)

        self.rated_capacity_W = float(config.get("rated_capacity_W", 200_000.0))  # 200 kW
        self.rated_fan_power_W = 0.03*self.rated_capacity_W   # Fan power: ~0.01 – 0.03 kW per kW of cooling rejected
        self.min_approach_C = float(config.get("min_approach_C", 3.0))            # design approach
        self.max_approach_C = float(config.get("max_approach_C", 7.0))            # part-load approach

        self.temp_range_C = float(config.get("temp_range_C", 5.0))

        self.rho = 1000.0  # kg/m³
        self.cp = 4180.0   # J/kg-K

        # Condenser pump
        self.pump_power_per_flow_W_per_m3s = float(config.get("pump_power_per_flow", 143_000.0))  # W per m³/s

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

        # === Heat rejected ===
        power_W = q_cooling_W / cop
        q_reject_W = q_cooling_W + power_W

        # === Load ratio and approach ===
        load_ratio = np.clip(q_reject_W / self.rated_capacity_W, 0.0, 1.0)
        approach = self.min_approach_C + (1.0 - load_ratio) * (self.max_approach_C - self.min_approach_C)

        # === CW temperatures ===
        cw_supply_temp_c = cw_supply_sp_c                                                              # to chiller
        cw_return_temp_c = max(wet_bulb_temp_c+approach, cw_supply_sp_c) + self.temp_range_C           # from chiller

        # === CW flow rate (Q = m*cp*dT) ===
        m_dot = q_reject_W / (self.cp * self.temp_range_C)
        cw_flow_m3s = m_dot / self.rho

        # === Fan power (cubic model) ===
        fan_power_W = self.rated_fan_power_W * load_ratio**3

        # === Pump power (linear with flow) ===
        pump_power_W = self.pump_power_per_flow_W_per_m3s * cw_flow_m3s

        # === Energy (fan + pump) ===
        total_power_W = fan_power_W + pump_power_W
        energy_J = total_power_W * timestep

        # === Update state ===
        state.heat_rejected_W = q_reject_W
        state.cw_supply_temp_c = cw_supply_temp_c
        state.cw_return_temp_c = cw_return_temp_c
        state.cw_flow_m3s = cw_flow_m3s
        state.fan_power_W = fan_power_W
        state.pump_power_W = pump_power_W
        state.energy_J_cum += energy_J

        # === Log ===
        self._record_state({
            "Q_rejected_W": q_reject_W,
            "cw_supply_temp_C": cw_supply_temp_c,
            "cw_return_temp_C": cw_return_temp_c,
            "cw_flow_m3s": cw_flow_m3s,
            "fan_power_W": fan_power_W,
            "pump_power_W": pump_power_W,
            "energy_J": energy_J,
            "approach_C": approach,
            "load_ratio": load_ratio
        })

        return {}

    def reset(self):
        self._state_history.clear()
        self._initialized = False
        self.initialize()
