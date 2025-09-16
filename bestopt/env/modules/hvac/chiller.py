"""
Chiller module.
"""

from typing import Dict, Any
import numpy as np

from ...core.base import BaseModule
from ...core.data_structure import ThermalAction, ChillerState, Disturbance


class ChillerModule(BaseModule):
    """
    Chiller module with Carnot-based COP, cooling output, and CHW flow rate.

    Input (from action):
        - ThermalAction.chiller_cooling_kw_sp  : Cooling demand [kW]
        - ThermalAction.chws_temp_c_sp         : Chilled water supply temp setpoint [°C]

    Input (from disturbance):
        - Disturbance.condenser_temp_c         : Condenser water temp [°C]

    Output (written in-place to ChillerState):
        - cooling_kw
        - cop
        - chws_temp_c (setpoint-following)
        - chw_flow_m3s
        - power_W
        - energy_J_cum
    """

    def __init__(self, config: Dict[str, Any], name: str = "chiller"):
        super().__init__(config, name)

        self.rated_capacity_kw: float = float(config.get("rated_capacity_kw", 150.0))
        self.min_plr: float = float(config.get("min_plr", 0.15))
        self.max_plr: float = float(config.get("max_plr", 1.03))
        self.eta_carnot: float = float(config.get("eta_carnot", 0.4))  # Carnot effectiveness
        self.min_cop: float = float(config.get("min_cop", 2.0))
        self.max_cop: float = float(config.get("max_cop", 10.0))
        self.min_chws_temp: float = float(config.get("min_chws_temp_c", 5.0))
        self.max_chws_temp: float = float(config.get("max_chws_temp_c", 10.0))
        self.rho: float = 1000.0  # kg/m³
        self.cp: float = 4180.0   # J/kg-K

    def initialize(self) -> None:
        self._initialized = True

    def step(
        self,
        state: "ChillerState",
        action: "ThermalAction",
        disturbance: "Disturbance",
        timestep: float
    ) -> Dict[str, Any]:
        """
        One timestep simulation of the chiller.

        Parameters:
            - state: ChillerState object to update in place
            - action: includes cooling setpoint (kW) and chws_temp_c_sp (°C)
            - disturbance: includes condenser_temp_c (°C)
            - timestep: seconds

        Returns: Empty dictionary (in-place update only)
        """

        # 1. Inputs
        q_kw = float(getattr(action, "chiller_cooling_kw_sp", 0.0))
        chws_sp = float(getattr(action, "chws_temp_c_sp", 7.0))
        chws_sp = np.clip(chws_sp, self.min_chws_temp, self.max_chws_temp)

        t_cond = float(getattr(disturbance, "condenser_temp_c", 35.0))
        t_evap = chws_sp

        # 2. Compute PLR
        plr = q_kw / self.rated_capacity_kw
        plr = np.clip(plr, self.min_plr, self.max_plr)
        q_out = self.rated_capacity_kw * plr  # Final cooling output [kW]

        # 3. COP (Carnot-based)
        T_evap_K = t_evap + 273.15
        T_cond_K = t_cond + 273.15
        delta_T = max(T_cond_K - T_evap_K, 0.5)
        cop_carnot = T_evap_K / delta_T
        cop = self.eta_carnot * cop_carnot
        cop = np.clip(cop, self.min_cop, self.max_cop)

        # 4. Power & Flow
        power_W = (q_out * 1000) / cop
        flow_m3s = (q_out * 1000) / (self.rho * self.cp * max(1e-3, (12.0 - t_evap)))

        # 5. Energy
        energy_J = power_W * timestep if timestep > 0 else 0.0

        # 6. In-place state update
        state.cooling_kw = q_out
        state.cop = cop
        state.chws_temp_c = chws_sp
        state.chw_flow_m3s = flow_m3s
        state.power_W = power_W
        state.energy_J_cum = getattr(state, "energy_J_cum", 0.0) + energy_J

        # 7. Logging
        self._record_state({
            "cooling_kw": q_out,
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
