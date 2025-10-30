from typing import Dict, Any
from math import inf

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import (
    HVACSystemAction, HVACSystemState, PumpComponentAction, ComponentType
)

class PumpLocalController(BaseModule):

    def __init__(self, config: Dict[str, Any], name: str = "PumpLocalController"):
        super().__init__(config, name)

        self.Kp: float = float(config.get("Kp_flow_per_K", 2e-4))  # (m^3/s)/K
        self.flow_min: float = float(config.get("flow_min_m3s", 0.0))
        self.flow_max: float = float(config.get("pump_flowrate_max", inf))
        self.deadband: float = float(config.get("deadband_K", 0.2))  
        self.rate_limit: float = float(config.get("rate_limit_m3s_per_s", 5e-5))  

        if self.flow_max <= 0:
            raise ValueError("pump_flowrate_max / flow_max must be positive.")
        if self.Kp < 0:
            raise ValueError("Kp_flow_per_K must be non-negative.")

        self.current_flowrate: float | None = None

    def initialize(self) -> None:
        self._initialized = True
        self.current_flowrate = None  

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def step(
        self,
        state: HVACSystemState,
        action: HVACSystemAction,
        timestep: float
    ) -> PumpComponentAction:


        T_sa_prev = float(getattr(state, "air_outlet_temp_c", None)
                          if getattr(state, "air_outlet_temp_c", None) is not None else
                          getattr(state, "air_outlet_temp_C", 0.0))  
        T_sa_sp   = float(getattr(action, "supply_temp_setpoint_c", None)
                          if getattr(action, "supply_temp_setpoint_c", None) is not None else
                          getattr(action, "supply_temp_setpoint_C", T_sa_prev))  

        flow_prev = self.current_flowrate if self.current_flowrate is not None else self.flow_min


        e = T_sa_prev - T_sa_sp


        delta = 0.0 if abs(e) < self.deadband else self.Kp * e

        flow_target = self._clip(flow_prev + delta, self.flow_min, self.flow_max)


        if self.rate_limit is not None and self.rate_limit > 0 and timestep and timestep > 0:
            max_step = self.rate_limit * float(timestep)
            flow_lo  = flow_prev - max_step
            flow_hi  = flow_prev + max_step
            flow_cmd = self._clip(flow_target, max(flow_lo, self.flow_min), min(flow_hi, self.flow_max))
        else:
            flow_cmd = flow_target

        self.current_flowrate = flow_cmd

        local_action = PumpComponentAction(
            component_id=state.system_id,
            component_type=ComponentType.PUMP.value
        )
        local_action.flow_setpoint_m3s = flow_cmd

        return local_action

    def reset(self) -> None:
        self.clear_history()
        self.current_flowrate = None
        self._initialized = True
