"""
Ideal HVAC module.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from ...core.base import BaseModule
from ...core.data_structure import State, Action, Disturbance


class HVACModule(BaseModule):
    """
    Ideal HVAC model.

    Features:
    # @TODO
    -
    -
    -
    """

    def __init__(self, config: Dict[str, Any], name: str = "HVAC"):
        """
        Initialize HVAC module.

        Args:
            config: HVAC configuration parameters
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
