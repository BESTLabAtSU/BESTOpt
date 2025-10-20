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


        # Logging
        self.logger = logging.getLogger(f"ElectricalDynamics.{name}")

    def initialize(self) -> None:
        pass


    def step(self, state: ElectricalZoneComponentState, action: DomainAction,
             disturbance: Disturbance, timestep: int) -> Dict[str, Any]:
        pass

    def reset(self) -> None:
        pass
