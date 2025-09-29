"""
Disturbance price module.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from ..core.base import BaseModule
from ..core.data_structure import State, Action, Disturbance, PriceSignals


class PriceModule(BaseModule):
    """
    Price energy storage model.

    Features:
    # @TODO
    -
    -
    -
    """

    def __init__(self, config: Dict[str, Any], name: str = "Price"):
        """
        Initialize price module.

        Args:
            config: Price configuration parameters
            name: Module name
        """
        super().__init__(config, name)
        self.current_price = PriceSignals()

    def initialize(self) -> None:
        self.peak_start = 17 * 4
        self.peak_end = 21 * 4
        daily_price = np.ones(96) * 10
        daily_price[self.peak_start:self.peak_end] = 15
        self.daily_price = daily_price
        daily_peaksignal = np.zeros(96, dtype=bool)
        daily_peaksignal[self.peak_start:self.peak_end] = True
        self.daily_peaksignal = daily_peaksignal

    def step(self, current_step: int) -> Optional[PriceSignals]:
        step_of_day = current_step % 96
        self.current_price.electricity_price = self.daily_price[step_of_day]
        self.current_price.peaksignal = self.daily_peaksignal[step_of_day]
        self.current_price.peak_start = self.peak_start
        self.current_price.peak_end = self.peak_end

        return self.current_price

    def reset(self) -> None:
        pass
