"""
Disturbance grid module.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from ..core.base import BaseModule
from ..core.data_structure import State, Action, Disturbance, PriceSignals


class GridModule(BaseModule):
    """
    Generate power-outage event.

    Features:
    # @TODO
    -
    -
    -
    """

    def __init__(self, config: Dict[str, Any], name: str = "Grid"):
        """
        """
        super().__init__(config, name)

    def initialize(self) -> None:
        pass

    def step(self, current_step: int) -> None:
        pass

    def reset(self) -> None:
        pass
