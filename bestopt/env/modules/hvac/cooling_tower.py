"""
Cooling tower module.
"""

from typing import Dict, Any
import numpy as np

from ...core.base import BaseModule
from ...core.data_structure import ThermalAction, CoolingTowerState


class CoolingTowerModule(BaseModule):
    """
    Cooling Tower module that rejects condenser heat and calculates fan power.

    Inputs:
        - ThermalAction.cooling_tower_load_W_sp [W]
        - ThermalAction.wet_bulb_temp_c [°C]

    Outputs (to CoolingTowerState):
        - heat_rejected_W
        - outlet_temp_c
        - fan_power_W
        - energy_J_cum
    """

    def __init__(self, config: Dict[str, Any], name: str = "cooling_tower"):
        super().__init__(config, name)

        self.rated_capacity_W = float(config.get("rated_capacity_W", 200_000.0))   # 200 kW
        self.rated_fan_power_W = float(config.get("rated_fan_power_W", 5000.0))
        self.min_approach_C = float(config.get("min_approach_C", 3.0))
        self.max_approach_C = float(config.get("max_approach_C", 7.0))

    def initialize(self) -> None:
        self._initialized = True

    def step(self, state: "CoolingTowerState", action: "ThermalAction", timestep: float) -> Dict[str, Any]:
        q_reject_W = float(getattr(action, "cooling_tower_load_W_sp", 0.0))
        wet_bulb_temp_c = float(getattr(action, "wet_bulb_temp_c", 25.0))

        load_ratio = np.clip(q_reject_W / self.rated_capacity_W, 0.0, 1.0)

        approach = self.min_approach_C + (1.0 - load_ratio) * (self.max_approach_C - self.min_approach_C)
        outlet_temp_c = wet_bulb_temp_c + approach
        fan_power_W = self.rated_fan_power_W * (load_ratio ** 3)
        energy_J = fan_power_W * timestep

        # Update state in-place
        state.heat_rejected_W = q_reject_W
        state.outlet_temp_c = outlet_temp_c
        state.fan_power_W = fan_power_W
        state.energy_J_cum = getattr(state, "energy_J_cum", 0.0) + energy_J

        self._record_state({
            "heat_rejected_W": q_reject_W,
            "outlet_temp_c": outlet_temp_c,
            "fan_power_W": fan_power_W,
            "energy_J": energy_J
        })

        return {}

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()