"""
Disturbance occupancy module — PATCHED for cross-case reproducibility.

CHANGE: initialize() now uses a deterministic seed so that all 4 cases
produce the exact same daily occupancy pattern.

The seed can be passed via config["occupancy_seed"].  If not provided,
falls back to np.random (original behavior, but non-reproducible).
"""
import numpy as np
from typing import Dict, Any, Optional
from ..core.base import BaseModule
from ..core.data_structure import OccupancyDisturbance


class OccupancyModule(BaseModule):
    """
    Occupancy model with proper multi-day support.
    Generates a daily occupancy pattern (away during work hours)
    and wraps correctly for any episode length via modulo indexing.
    """

    def __init__(self, config: Dict[str, Any], name: str = "Occupancy"):
        super().__init__(config, name)
        self.occ = OccupancyDisturbance()
        self.steps_per_day = config.get("steps_per_day", 96)
        self.forecast_horizon = config.get("forecast_horizon", 96)
        self.occupancy_seed = config.get("occupancy_seed", None)  # ← NEW

    def initialize(self) -> None:
        # Use a local RNG so we don't pollute the global np.random state
        if self.occupancy_seed is not None:
            rng = np.random.RandomState(self.occupancy_seed)
        else:
            rng = np.random  # fallback: original (non-reproducible) behavior

        leave = int(rng.normal(8 * 4, 2 * 4))
        back  = int(rng.normal(17 * 4, 2 * 4))

        leave = max(0, min(leave, self.steps_per_day - 1))
        back  = max(leave + 1, min(back, self.steps_per_day))

        self.daily = np.ones(self.steps_per_day)
        self.daily[leave:back] = 0

        self.logger.info(f"Occupancy pattern: leave={leave} ({leave/4:.1f}h), "
                         f"back={back} ({back/4:.1f}h), seed={self.occupancy_seed}")

    def step(self, current_step: int) -> Optional[OccupancyDisturbance]:
        step_of_day = current_step % self.steps_per_day
        self.occ.occupancy_fraction = self.daily[step_of_day]
        self.occ.step_of_day = step_of_day

        forecast_indices = [(step_of_day + i) % self.steps_per_day
                           for i in range(self.forecast_horizon)]
        self.occ.occupancy_forecast = self.daily[forecast_indices]
        return self.occ

    def reset(self) -> None:
        # Re-randomize daily pattern on reset for training variety
        self.initialize()