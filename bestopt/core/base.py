"""
This code defines the BaseClass for bestopt dynamic modules to inherit from,
In order to ensure a consistent format.

  - initialize(): prepare configuration / parameters
  - step(): run one simulation timestep
  - reset(): return to initial conditions

It also provides optional utilities for later use such as input validation, state save/load,
and state history recording.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

# Shared data structures that define common inputs/outputs between modules
from .data_structures import (
    BuildingState, ControlAction, WeatherData,
    PriceSignal, SimulationConfig
)


class BaseModule(ABC):
    """Abstract base class for all dynamic modules.

    Every dynamic module (e.g., Building, HVAC, DERs...) should inherit from
    this class and implement initialize(), step(), and reset() methods.
    """

    def __init__(self, config: Dict[str, Any], name: str = "BaseModule"):
        """Initialize base module.

        Args:
            config: Module configuration dictionary
            name: Module name, also used for namespacing the logger.
        """
        self.config = config
        self.name = name

        # Module-level logger
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Tracks if the module has been initialized before stepping
        self._initialized = False

        # Optional: store time-series of states if history enabled
        self._state_history: List[Dict[str, Any]] = []
        self._enable_history = config.get('enable_history', False)

    # ------------------- ABSTRACT INTERFACE -------------------

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the module with configuration.

        Example: allocate arrays, set initial conditions, load parameters.
        Must be called before simulation starts.
        """
        pass

    @abstractmethod
    def step(self, state: State, action: Action,
             disturbance: Disturbance, timestep: float) -> Dict[str, Any]:
        """Execute one simulation timestep.

        Args:
            state: Current state of dynamic module (temperature, SOC, etc.)
            action: Control action to apply (charge/discharge power, cooling/heating etc.)
            disturbance: Disturbance at current step (weather, utility price, etc.)
            timestep: Simulation timestep (default: 15 minutes)

        Returns:
            Dictionary containing module outputs (e.g., updated state variables,
            energy flows, KPIs). These outputs will be merged by the environment.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset module to its initial state.

        Ensures reproducibility between episodes or experiments.
        Example: set SOC = 100%.
        """
        pass

    # ------------------- OPTIONAL HOOKS -------------------

    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters.

        Args:
            **kwargs: Arbitrary key-value pairs for inputs.

        Returns:
            True if inputs are valid (default always True).
            Override in child classes for input sanity checks.
        """
        return True

    def get_state(self) -> Dict[str, Any]:
        """Get current module state as dictionary.

        Example keys: {"indoor_temp": 22.5, "SOC": 0.8}
        Used for evaluation, saving, and debugging.
        """
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state from dictionary.

        Args:
            state: State dictionary to restore.
        This allows checkpointing and restoring simulations.
        """
        pass

    # ------------------- STATE RECORDING -------------------

    def save_state(self, filepath: str) -> None:
        """Save module state to file using pickle."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, filepath: str) -> None:
        """Load module state from file using pickle and apply to module."""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.set_state(state)

    # ------------------- HISTORY HANDLING -------------------

    def _record_state(self, state: Dict[str, Any]) -> None:
        """Record state in history if enabled (internal use only)."""
        if self._enable_history:
            # Copy ensures stored history is not affected by later modifications
            self._state_history.append(state.copy())

    def get_history(self) -> List[Dict[str, Any]]:
        """Get recorded state history.

        Returns:
            List of state dicts from previous timesteps.
        """
        return self._state_history.copy()

    def clear_history(self) -> None:
        """Clear state history (useful for restarting episodes)."""
        self._state_history.clear()
