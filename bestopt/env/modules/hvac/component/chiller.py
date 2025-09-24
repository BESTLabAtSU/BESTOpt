"""
Chiller module
"""

from typing import Dict, Any
import numpy as np

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import ThermalAction, ChillerState


class ChillerModule(BaseModule):
    """
    Chiller module with Carnot-based COP, cooling output, and CHW flow rate.

    Input (from action):
        - ThermalAction.chiller_cooling_W_sp  : Cooling demand [W]
        - ThermalAction.chws_temp_c_sp        : CHW supply temp setpoint [°C]
        - ThermalAction.condenser_temp_c_sp   : Condenser temp setpoint [°C]

    Output (written in-place to ChillerState):
        - cooling_W
        - cop
        - chws_temp_c
        - chw_flow_m3s
        - power_W
        - energy_J_cum
    """

    def __init__(self, config: Dict[str, Any], name: str = "chiller"):
        super().__init__(config, name)

        self.rated_capacity_W = float(config.get("rated_capacity_W", 150_000.0))  # 150 kW
        self.min_plr = float(config.get("min_plr", 0.15))
        self.max_plr = float(config.get("max_plr", 1.03))
        self.eta_carnot = float(config.get("eta_carnot", 0.4))
        self.min_cop = float(config.get("min_cop", 2.0))
        self.max_cop = float(config.get("max_cop", 10.0))
        self.min_chws_temp = float(config.get("min_chws_temp_c", 5.0))
        self.max_chws_temp = float(config.get("max_chws_temp_c", 10.0))
        self.rho = 1000.0  # kg/m³
        self.cp = 4180.0   # J/kg-K

    def initialize(self) -> None:
        self._initialized = True

    def step(
        self,
        state: "ChillerState",
        action: "ThermalAction",
        timestep: float
    ) -> Dict[str, Any]:
        q_W = float(getattr(action, "chiller_cooling_W_sp", 0.0))
        chws_sp = np.clip(float(getattr(action, "chws_temp_c_sp", 7.0)),
                          self.min_chws_temp, self.max_chws_temp)
        t_cond = float(getattr(action, "condenser_temp_c_sp", 35.0))
        t_evap = chws_sp

        plr = np.clip(q_W / self.rated_capacity_W, self.min_plr, self.max_plr)
        q_out_W = self.rated_capacity_W * plr

        # COP
        T_evap_K = t_evap + 273.15
        T_cond_K = t_cond + 273.15
        delta_T = max(T_cond_K - T_evap_K, 0.5)
        cop_carnot = T_evap_K / delta_T
        cop = np.clip(self.eta_carnot * cop_carnot, self.min_cop, self.max_cop)

        power_W = q_out_W / cop
        flow_m3s = q_out_W / (self.rho * self.cp * max(1e-3, (12.0 - t_evap)))

        energy_J = power_W * timestep if timestep > 0 else 0.0

        # Update state
        state.cooling_W = q_out_W
        state.cop = cop
        state.chws_temp_c = chws_sp
        state.chw_flow_m3s = flow_m3s
        state.power_W = power_W
        state.energy_J_cum += energy_J

        self._record_state({
            "cooling_W": q_out_W,
            "cop": cop,
            "power_W": power_W,
            "chws_temp_c": chws_sp,
            "chw_flow_m3s": flow_m3s,
            "energy_J": energy_J
        })

        return {}

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()
