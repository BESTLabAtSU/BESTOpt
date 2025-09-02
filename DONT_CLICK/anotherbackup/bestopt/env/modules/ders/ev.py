"""
EV module.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import logging

from ...core.base import BaseModule
from ...core.data_structure import State, Action, Disturbance, EVMode


class EVModule(BaseModule):
    """
    EV energy storage model.

    Features:
    # @TODO
    - 
    - 
    - 
    """

    def __init__(self, config: Dict[str, Any], name: str = "EV"):
        """
        Initialize EV module.

        Args:
            config: EV configuration parameters
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
