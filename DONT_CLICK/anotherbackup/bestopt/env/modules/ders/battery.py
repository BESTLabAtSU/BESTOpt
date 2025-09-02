"""
Battery Energy Storage System (BESS) module.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import logging

from ...core.base import BaseModule
from ...core.data_structure import State, Action, Disturbance, BatteryMode


class BatteryStatus(Enum):
    """Battery operational status."""
    NORMAL = "normal"
    WARNING = "warning"
    FAULT = "fault"
    MAINTENANCE = "maintenance"


class BatteryModule(BaseModule):
    """
    Battery energy storage model.

    Features:
    # @TODO
    -
    -
    -
    """

    def __init__(self, config: Dict[str, Any], name: str = "Battery"):
        """
        Initialize battery module.

        Args:
            config: Battery configuration parameters
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
