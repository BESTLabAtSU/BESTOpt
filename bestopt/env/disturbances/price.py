"""
Disturbance price module — fixed for multi-day episodes.
"""
import numpy as np
from typing import Dict, Any, Optional
from ..core.base import BaseModule
from ..core.data_structure import PriceSignals


class PriceModule(BaseModule):
    """
    Price signal model with proper multi-day support.

    Uses modulo indexing so any episode length works.
    """

    def __init__(self, config: Dict[str, Any], name: str = "Price"):
        super().__init__(config, name)
        self.price_signal = PriceSignals()
        self.steps_per_day = config.get("steps_per_day", 96)
        self.forecast_horizon = config.get("forecast_horizon", 96)

    def initialize(self) -> None:
        self.peak_start = 17 * 4   # step 68
        self.peak_end = 21 * 4     # step 84

        # Off-peak and peak prices (¢/kWh)
        self.off_peak_price = 10.0
        self.peak_price = 15.0

        # Build daily patterns
        self.daily_price = np.ones(self.steps_per_day) * self.off_peak_price
        self.daily_price[self.peak_start:self.peak_end] = self.peak_price

        self.daily_peaksignal = np.zeros(self.steps_per_day, dtype=bool)
        self.daily_peaksignal[self.peak_start:self.peak_end] = True

    def step(self, current_step: int) -> Optional[PriceSignals]:
        step_of_day = current_step % self.steps_per_day

        self.price_signal.electricity_price = self.daily_price[step_of_day]
        self.price_signal.peaksignal = bool(self.daily_peaksignal[step_of_day])
        self.price_signal.peak_start = self.peak_start
        self.price_signal.peak_end = self.peak_end

        # Forecast: wrap around daily pattern
        forecast_indices = [(step_of_day + i) % self.steps_per_day
                           for i in range(self.forecast_horizon)]
        self.price_signal.forecast_electricity_price = self.daily_price[forecast_indices]
        self.price_signal.forecast_peaksignal = self.daily_peaksignal[forecast_indices]

        return self.price_signal

    def reset(self) -> None:
        pass