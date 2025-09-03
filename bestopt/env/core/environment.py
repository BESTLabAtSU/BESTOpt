"""
Runtime environment for bestopt framework.
Manages module execution and simulation state.
"""

import logging
import importlib
from typing import Dict, Any, List, Optional, Tuple

from .base import BaseModule
from .data_structure import State, Action, Disturbance, Observation


class BestOptEnvironment:
    """Runtime environment that manages all modules."""

    def __init__(self, configuration: Dict[str, Any]):
        """Initialize environment with configuration.

        Args:
            configuration: Configuration dictionary (from ConfigurationManager.get_final_configuration())
            expected keys:
               - environment: {resolution:int, duration:int, ...}
               - modules: {<name>: {class_path:str, parameters:dict} }
               - controllers: {<name>: {class_path, parameters}, selected:str }
               - disturbances: {<name>: {class_path, parameters}, selected:str }
        """
        self.config = configuration
        self.logger = logging.getLogger("BestOptEnvironment")

        # Environment config
        env_config = self.config.get('environment', {})
        self.res = env_config.get("resolution")
        if self.res is None:
            raise ValueError("Missing required config key: 'resolution'")
        if not isinstance(self.res, int) or self.res <= 0:
            raise ValueError(f"'resolution' must be a positive int (seconds); got {self.res}")

        self.dur = env_config.get("duration", 24 * 60 * 60)  # 1 day simulation as default
        if not isinstance(self.dur, int) or self.dur <= 0:
            raise ValueError(f"'duration' must be a positive int (seconds); got {self.dur}")

        # warning if duration not divisible, only accept 'int' steps
        self.total_step = self.dur // self.res
        if self.dur % self.res != 0:
            self.logger.warning(
                f"duration ({self.dur}) not divisible by resolution ({self.res}); "
                f"sim will run {self.total_step} steps (= floor)."
            )

        # Simulation data flow
        self.state = State()
        self.observation = Observation()
        self.disturbance = Disturbance()
        self.action = Action()

        # Tracking
        self.current_step = 0
        self.done = False

        # Instance registration
        self.modules: Dict[str, BaseModule] = {}
        self.controller: Dict[str, BaseModule] = {}
        self.disturbance: Dict[str, BaseModule] = {}

        # Build runtime
        self._generate_components()  # include modules/controller/disturbance

        self.logger.info(f"Environment configuration finished")

    # Module generation help functions
    def _generate_components(self) -> None:
        """Generate all components (modules, controller, disturbances)."""
        # Generate modules
        modules_config = self.config.get('modules', {})
        for module_name, module_info in modules_config.items():
            instance = self._create_instance(
                name=module_name,
                config=module_info,
                component_type="module",
                must_subclass=BaseModule
            )
            if instance:
                self.modules[module_name] = instance

        # Generate controller
        controller_config = self.config.get('controller', {})
        active_name = controller_config.get('_active')
        if active_name and controller_config:
            ctrl_config = {k: v for k, v in controller_config.items() if k != '_active'}
            instance = self._create_instance(
                name=active_name,
                config=ctrl_config,
                component_type="controller",
                must_subclass=BaseModule
            )
            if instance:
                self.controller = instance
        else:
            self.logger.info("No controller activated; external actions required.")

        # Generate disturbances
        disturbances_config = self.config.get('disturbances', {})
        for dist_name, dist_info in disturbances_config.items():
            instance = self._create_instance(
                name=dist_name,
                config=dist_info,
                component_type="disturbance",
                must_subclass=BaseModule
            )
            if instance:
                self.disturbance[dist_name] = instance

    def _create_instance(
            self,
            name: str,
            config: Dict[str, Any],
            component_type: str,
            must_subclass: Optional[type] = None
    ) -> Optional[BaseModule]:
        """Generic function to create component instances.

        Args:
            name: Component name
            config: Component configuration dict with 'class_path' and 'parameters'
            component_type: Type for logging ('module', 'controller', 'disturbance')
            must_subclass: Base class that the component must inherit from

        Returns:
            Created instance or None if creation failed
        """
        if not config:
            self.logger.warning(f"{component_type.title()} '{name}' has empty config; skipping.")
            return None

        # Get class path
        class_path = config.get('class_path')
        if not class_path:
            self.logger.warning(f"{component_type.title()} '{name}' missing 'class_path'; skipping.")
            return None

        # Import the class
        try:
            component_class = self._import_class(class_path, must_subclass)
        except Exception as e:
            self.logger.error(f"Failed to import {component_type} '{name}' from {class_path}: {e}")
            return None

        # Get parameters
        params = config.get('parameters', {})

        # Create instance
        try:
            # Register and initialize
            instance = component_class(config=params, name=name)
            instance.initialize()
            instance._initialized = True

            self.logger.info(f"Generated {component_type}: {name} ({class_path})")
            return instance

        except Exception as e:
            self.logger.error(f"Failed to create {component_type} '{name}': {e}")
            raise

    @staticmethod
    def _import_class(class_path: str, must_subclass: Optional[type] = None):
        parts = class_path.split('.')
        module_path = '.'.join(parts[:-1])
        class_name = parts[-1]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        # To keep consistency, all modules need to inherit from base modules
        if must_subclass and not issubclass(cls, must_subclass):
            raise TypeError(f"{class_path} must inherit from {must_subclass.__name__}")
        return cls

    def _reset_components(self, components: dict, component_type: str) -> None:
        """Helper function to reset modules, controllers, or disturbances."""
        for name, comp in components.items():
            try:
                comp.reset()
                self.logger.debug(f"Reset {component_type}: {name}")
            except Exception as e:
                self.logger.error(f"Error resetting {component_type} {name}: {e}")
                raise

    def reset(self) -> Observation:
        """
        Reset the environment to initial state.
        """
        self.logger.info("Resetting environment")

        # Reset tracking
        self.current_step = 0
        self.done = False

        # Reset all modules
        self._reset_components(self.modules, "module")
        self._reset_components(self.controller, "controller")
        self._reset_components(self.disturbance, "disturbance")

        # Reset state
        self.state = State()
        self.observation = Observation()
        self.disturbance = Disturbance()
        self.action = Action()

        # return @TODO return the initial observation

    def step(self, action: Action) -> Tuple[Observation, bool, Dict[str, Any]]:
        """Execute one simulation step.

        Args:
            action: Control actions for this step

        Returns:
           Tuple of (observation, done, info):
           - observation: Current system observation for the controller
           - done: Whether simulation is complete
           - info: Additional simulation information / metrics
        """
        # Check simulation status
        if self.done:
            self.logger.warning("Environment is done. Call reset() to restart.")
            return self.observation, True, {}

        # Get control actions
        try:
            # @TODO the format of action need to be carefully defined
            self.action = self.controller.step(
                state=self.state,
                action=self.action,
                disturbance=self.disturbance,
                resolution=self.res,
                timestep=self.current_step
            )
        except Exception as e:
            self.logger.error(f"Controller step failed: {e}")

        # Get disturbance
        try:
            self.disturbance = self.disturbance.step(
                resolution=self.res,
                timestep=self.current_step
            )
        except Exception as e:
            self.logger.error(f"Disturbance step failed: {e}")

        # Get modules update
        module_outputs = {}
        for module_name, module in self.modules.items():
            try:
                output = module.step(
                    state=self.state,
                    action=self.action,
                    disturbance=self.disturbance,
                    resolution=self.res,
                    timestep=self.current_step
                )
                module_outputs[module_name] = output
            except Exception as e:
                self.logger.error(f"Module {module_name} step failed: {e}")

        # Get observation
        # @TODO self._update_observation()

        # Get tracking update
        self.current_step += 1
        if self.current_step >= self.total_step:
            self.done = True
            self.logger.info(f"Simulation completed after {self.current_step} steps")

        # Grab all information
        # @TODO info = self._collect_step_info(module_outputs)

