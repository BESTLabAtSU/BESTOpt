"""
Pump module.

Since actual pump head is generally unknown, pump power is modeled proportional to the cube of the flow ratio: pump power ∝ (flow/nominal flow)³
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import logging

from ...core.base import BaseModule
from ...core.data_structure import State, Action, Disturbance, HVACMode


class PumpModule(BaseModule):
    """
    Pump model for closed-loop HVAC water systems.

    Assumptions:
    - Pump power scales with the cube of flow ratio.
    - Nominal power and flow are provided in config.
    
    Assumptions:
    - Pump power scales with cube of flow ratio (affinity law).
    - Flow is provided in the Action or falls back to nominal flow.

    """

    def __init__(self, config: Dict[str, Any], name: str = "Pump"):
        """
        Initialize Pump module.

        Args:
            config: Pump configuration parameters
            name: Module name
        """
        super().__init__(config, name)
        self.nominal_flow = config.get("nominal_flow", 1.0)     # m³/s (or consistent unit)
        self.nominal_power = config.get("nominal_power", 1.0)   # Watts (W)
        self.logger = logging.getLogger(f"{__name__}.{name}")

    def initialize(self) -> None:
        self.energy_total = 0.0  # cumulative J

    def step(self, state: State, action: Action,
            disturbance: Disturbance, timestep: float) -> Dict[str, Any]:
        # Get flow from action, fallback to nominal
        flow = getattr(action.thermal, "pump_flow", self.nominal_flow)
        # flow = max(0.0, flow)  # avoid negatives

        if self.nominal_flow <= 0:
            self.logger.warning("Nominal flow <= 0, cannot compute pump power.")
            power = 0.0
            flow_ratio = 0.0
        else:
            flow_ratio = flow / self.nominal_flow
            power = self.nominal_power * (flow_ratio ** 3) 

        # Energy in this step (J)
        energy_step = power * timestep
        self.energy_total += energy_step

        return {
            "pump_power": power,
            "pump_energy_step": energy_step,
            "pump_energy_total": self.energy_total,
            "pump_flow": flow,
            "pump_flow_ratio": flow_ratio
        }
        
    def reset(self) -> None:
        self.energy_total = 0.0