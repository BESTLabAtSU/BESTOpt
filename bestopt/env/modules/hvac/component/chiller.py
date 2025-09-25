"""
Chiller module (Carnot-based COP with CHW flow and return temp inputs).
"""

from typing import Dict, Any
import numpy as np

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import ThermalAction, ChillerState, PumpState, CoilState


class ChillerModule(BaseModule):
    """
    Chiller module with Carnot-based COP, cooling output, and CHW flow rate.

    Input (from ThermalAction, CoilState and PumpState):
        - CoilState.water_outlet_temp_C       : CHW inlet temp [°C]
        - ThermalAction.chws_temp_c_sp        : CHW outlet temp setpoint [°C]
        - PumpState.waterflow_m3s             : CHW flow rate [m³/s]
        - ThermalAction.condenser_temp_c_sp   : Condenser inlet temp setpoint [°C]

    Output (written in-place to ChillerState):
        - cooling_W                           : Cooling output [W]
        - cop                                 : Coefficient of performance [-]
        - chws_temp_c                         : CHW outlet temperature [°C]
        - chw_flow_m3s                        : CHW flow rate [m³/s]
        - power_W                             : Electrical power consumption [W]
        - energy_J_cum                        : Cumulative energy consumption [J]
        
    Model:
      - Carnot-based COP with fixed effectiveness factor:
          COP = η_carnot * (T_cw / (T_cw - T_chw)), using Kelvin temps
      - Cooling output calculated as:
          Q = m_dot * c_p * (T_return - T_supply)
      - Power = Q / COP
      - All temperatures are assumed to be in °C and converted to K internally

    State:
      - Reads from:
          * CoilState.water_outlet_temp_C
          * PumpState.waterflow_m3s
          * ThermalAction.chws_temp_c_sp
          * ThermalAction.condenser_temp_c_sp
      - Writes IN-PLACE to ChillerState:
          * ChillerState.cooling_W
          * ChillerState.cop
          * ChillerState.chws_temp_c
          * ChillerState.chw_flow_m3s
          * ChillerState.power_W
          * ChillerState.energy_J_cum

    Action:
      - Expects ThermalAction instances with:
          * chws_temp_c_sp
          * condenser_temp_c_sp
      - Indirectly affects condenser-side heat rejection (used in cooling tower model)
    """

    def __init__(self, config: Dict[str, Any], name: str = "chiller"):
        super().__init__(config, name)

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
        coil_state: "CoilState",
        pump_state: "PumpState",
        timestep: float
    ) -> Dict[str, Any]:
        t_in = float(getattr(coil_state, "water_outlet_temp_C", 12.0))
        t_out_sp = np.clip(float(getattr(action, "chws_temp_c_sp", 7.0)),
                           self.min_chws_temp, self.max_chws_temp)
        flow_m3s = float(getattr(pump_state, "waterflow_m3s", 0.01))
        t_cond = float(getattr(action, "condenser_temp_c_sp", 35.0))

        mass_flow_kg_s = self.rho * flow_m3s
        q_cooling_W = mass_flow_kg_s * self.cp * (t_in - t_out_sp)
        q_cooling_W = max(q_cooling_W, 0.0)

        # COP Calculation (Carnot)
        T_evap_K = t_out_sp + 273.15
        T_cond_K = t_cond + 273.15
        delta_T = max(T_cond_K - T_evap_K, 0.5)
        cop_carnot = T_evap_K / delta_T
        cop = np.clip(self.eta_carnot * cop_carnot, self.min_cop, self.max_cop)

        power_W = q_cooling_W / cop if cop > 0 else 0.0
        energy_J = power_W * timestep

        # Update state
        state.cooling_W = q_cooling_W
        state.cop = cop
        state.chws_temp_c = t_out_sp
        state.chw_flow_m3s = flow_m3s
        state.power_W = power_W
        state.energy_J_cum += energy_J

        self._record_state({
            "cooling_W": q_cooling_W,
            "cop": cop,
            "power_W": power_W,
            "chws_temp_c": t_out_sp,
            "chw_flow_m3s": flow_m3s,
            "energy_J": energy_J
        })

        return {}

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()
