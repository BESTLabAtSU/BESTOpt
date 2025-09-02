from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class StateVariable:
    """Container for state variables"""
    name: str
    value: float
    unit: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def update(self, new_value: float):
        """Update state value with bounds checking"""
        if self.min_val is not None and new_value < self.min_val:
            self.value = self.min_val
        elif self.max_val is not None and new_value > self.max_val:
            self.value = self.max_val
        else:
            self.value = new_value


@dataclass
class ControlVariable:
    """Container for control variables"""
    name: str
    value: float
    unit: str
    min_val: float = 0.0
    max_val: float = 1.0


@dataclass
class DisturbanceVariable:
    """Container for disturbances variables"""
    name: str
    value: float
    unit: str


class BaseModule(ABC):
    """Base class for all modules"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.states: Dict[str, StateVariable] = {}
        self.controls: Dict[str, ControlVariable] = {}
        self.disturbances: Dict[str, DisturbanceVariable] = {}
        self._initialize_variables()

    @abstractmethod
    def _initialize_variables(self):
        """Module Initialization"""
        pass

    @abstractmethod
    def update(self,
               dt: float,
               signal_bus: Dict[str, Any],
               thermal_bus: Dict[str, Any],
               electric_bus: Dict[str, Any]):
        """Update module state based on inputs from three buses"""
        pass

    def get_state(self, name: str) -> float:
        """Get state variable value"""
        return self.states[name].value if name in self.states else None

    def set_control(self, name: str, value: float):
        """Set control variable value"""
        if name in self.controls:
            self.controls[name].value = np.clip(value,
                                                self.controls[name].min_val,
                                                self.controls[name].max_val)