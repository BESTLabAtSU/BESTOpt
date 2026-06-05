"""
Building electric dynamic module.
"""
import numpy as np
import torch
import pickle
import pandas as pd
from datetime import timedelta
from typing import Dict, Any, Optional, Tuple
from collections import deque
import logging
from ...core.base import BaseModule
from ...core.data_structure import ElectricalZoneComponentState, DomainAction, Disturbance, BuildingSystemState
import os


class ElectricalDynamicModule(BaseModule):
    """
    Building electrical dynamic module.
    """

    def __init__(self, config: Dict[str, Any], name: str = ""):
        super().__init__(config, name)
        self.lighting_day = config['lighting']['daytime']
        self.lighting_night = config['lighting']['nighttime']
        self.cooking = config['appliance']['cooking']
        self.pc = config['appliance']['pc']
        self.tv = config['appliance']['tv']

        # Logging
        self.logger = logging.getLogger(f"ElectricalDynamics.{name}")

    def initialize(self) -> None:
        pass


    def step(self, disturbance: Disturbance, timestep: int) -> Dict[str, Any]:
        occ_status = disturbance.occupancy.occupancy_fraction

        # Use time-of-day step, not global timestep
        step_of_day = timestep % (24 * 4)  # assuming 15-min resolution = 96 steps/day

        lighting_power = 0
        if step_of_day >= 18 * 4 and step_of_day <= 24 * 4:
            lighting_power = self.lighting_night
        if step_of_day >= 6 * 4 and step_of_day <= 10 * 4:
            lighting_power = self.lighting_day

        cooking_power = 0
        if step_of_day >= 7 * 4 + np.random.randint(-4, 4) and step_of_day <= 9 * 4 + np.random.randint(-4, 4):
            cooking_power = self.cooking
        if step_of_day >= 17 * 4 + np.random.randint(-4, 4) and step_of_day <= 20 * 4 + np.random.randint(-4, 4):
            cooking_power = self.cooking

        building_power = occ_status * (lighting_power + self.pc + self.tv + cooking_power)
        self.building_power = building_power
        self.lighting_power = lighting_power*occ_status
        self.pc_power = self.pc * occ_status
        self.tv_power = self.tv * occ_status
        self.cooking_power = cooking_power * occ_status
        return building_power

    def reset(self) -> None:
        pass
