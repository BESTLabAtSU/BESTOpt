"""
Building thermal dynamic module.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import logging

from ...core.base import BaseModule
from ...core.data_structure import State, Action, Disturbance, BatteryMode


class ThermalDynamicsModule(BaseModule):
    """
    Building thermal dynamic.

    Features:
    # @TODO
    -
    -
    -
    """

    def __init__(self, config: Dict[str, Any], name: str = ""):
        """
        Initialize building thermal dynamic module.

        Args:
            config: building thermal dynamic configuration parameters
            name: Module name
        """
        super().__init__(config, name)
        pass

    def initialize(self) -> None:
        pass

    def step(self, state: State, action: Action,
             disturbance: Disturbance, timestep: float) -> Dict[str, Any]:
        pass

    def reset(self) -> None:
        pass
