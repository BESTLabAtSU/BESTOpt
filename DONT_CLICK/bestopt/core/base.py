from abc import ABC, abstractmethod
from typing import Dict, Any


class Module(ABC):
    """Base class for all runtime simulation modules."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the module with configuration."""
        pass

    @abstractmethod
    def step(self,
             control_action: Dict[str, float],
             disturbance: Dict[str, float],
             dt: float) -> Dict[str, float]:
        """Perform one-step simulation."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, float]:
        """Get current state of the module."""
        pass

    def reset(self) -> None:
        """Reset module to initial state."""
        self.initialize()


class DynamicModule(Module):
    """Base class for dynamic modules with state variables."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.state = {}
        self.state_bounds = {}

    @abstractmethod
    def state_transition(self, current_state: Dict[str, float],
                         control_action: Dict[str, float],
                         disturbance: Dict[str, float],
                         dt: float) -> Dict[str, float]:
        """Calculate next state based on current state, control, and disturbances."""
        pass

    def step(self, control_action: Dict[str, float],
             disturbance: Dict[str, float],
             dt: float) -> Dict[str, float]:
        """Update state and return outputs."""
        self.state = self.state_transition(self.state, control_action, disturbance, dt)
        return self.get_output()

    @abstractmethod
    def get_output(self) -> Dict[str, float]:
        """Get output based on current state."""
        pass


class StaticModule(Module):
    """Base class for static modules without state variables."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def step(self, control_action: Dict[str, float],
             disturbance: Dict[str, float],
             dt: float) -> Dict[str, float]:
        """Calculate output directly from inputs."""
        return self.calculate_output(control_action, disturbance)

    @abstractmethod
    def calculate_output(self, control_action: Dict[str, float],
                         disturbance: Dict[str, float]) -> Dict[str, float]:
        """Calculate output based on control and disturbances."""
        pass

    def get_state(self) -> Dict[str, float]:
        """Static modules have no state."""
        return {}