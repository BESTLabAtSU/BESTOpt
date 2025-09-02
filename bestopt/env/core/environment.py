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

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute one simulation step.

        Args:
            action: Control actions for this step

        Returns:
            Tuple of (observation, reward, done, info)
        """

        # Check if environment is initialized
        if not self.modules:
            raise RuntimeError("No modules loaded. Check configuration.")

        # Get current disturbance
        self.disturbance = self._get_disturbance()

        # Execute all modules
        module_outputs = {}
        for name, module in self.modules.items():
            try:
                outputs = module.step(
                    state=self.state,
                    action=action,
                    disturbance=self.disturbance,
                    timestep=self.timestep
                )
                module_outputs[name] = outputs

                # Update global state with module outputs
                self._update_state(name, outputs)

            except Exception as e:
                self.logger.error(f"Error in module {name}.step(): {e}")
                raise

        # Calculate reward
        reward = self._calculate_reward(module_outputs)
        self.episode_reward += reward

        # Update step counter
        self.current_step += 1
        self.state.step = self.current_step

        # Check termination
        self.done = self.current_step >= self.max_steps

        # Get next observation
        self.observation = self._get_observation()

        # Prepare info
        info = {
            'step': self.current_step,
            'timestep': self.timestep,
            'modules': list(self.modules.keys()),
            'module_outputs': module_outputs,
            'state': self.state.to_dict(),
            'episode_reward': self.episode_reward
        }

        return self.observation, reward, self.done, info

    def _update_state(self, module_name: str, outputs: Dict[str, Any]) -> None:
        """Update global state based on module outputs.

        Args:
            module_name: Name of the module
            outputs: Module output dictionary
        """
        # Map module outputs to state variables
        # Customize this based on your specific modules

        if module_name == 'battery':
            if 'soc' in outputs:
                self.state.electrical.battery_soc = outputs['soc']
            if 'power' in outputs:
                self.state.electrical.battery_to_building = outputs['power']
            if 'temperature' in outputs:
                self.state.electrical.battery_temperature = outputs['temperature']

        elif module_name == 'pv':
            if 'generation' in outputs:
                self.state.electrical.pv_building = outputs['generation']

        elif module_name == 'ev':
            if 'soc' in outputs:
                self.state.electrical.ev_soc = outputs['soc']
            if 'power' in outputs:
                self.state.electrical.ev_to_building = outputs['power']

        elif module_name == 'building':
            if 'zone_temperatures' in outputs:
                self.state.thermal.zone_temperatures = outputs['zone_temperatures']
            if 'plug_loads' in outputs:
                self.state.electrical.plug_loads_power = outputs['plug_loads']

        elif module_name == 'hvac':
            if 'power' in outputs:
                self.state.electrical.hvac_power = outputs['power']

        elif module_name == 'grid':
            if 'import_power' in outputs:
                self.state.electrical.grid_building = outputs['import_power']

        # Add more mappings as you develop modules

    def _get_disturbance(self) -> Disturbance:
        """Get current disturbance values.

        Returns:
            Disturbance object with current values
        """
        dist = Disturbance()

        # Calculate time-based values
        hour = (self.current_step * self.timestep / 3600) % 24

        # Example disturbance patterns (replace with data loading)
        # Weather
        dist.weather.outdoor_temperature = 20 + 5 * (1 - abs(hour - 14) / 10)
        dist.weather.solar_radiation = max(0, 800 * (1 - abs(hour - 12) / 6)) if 6 <= hour <= 18 else 0

        # Prices (example time-of-use pricing)
        if 16 <= hour <= 21:  # Peak hours
            dist.prices.electricity_price = 0.25
        elif 6 <= hour <= 16:  # Mid-peak
            dist.prices.electricity_price = 0.15
        else:  # Off-peak
            dist.prices.electricity_price = 0.10

        # Occupancy
        if 8 <= hour <= 18:  # Work hours
            dist.occupancy.occupancy_fraction = 0.8
            dist.occupancy.is_occupied = True
        else:
            dist.occupancy.occupancy_fraction = 0.1
            dist.occupancy.is_occupied = False

        return dist

    def _get_observation(self) -> Observation:
        """Construct observation from current state.

        Returns:
            Observation object
        """
        obs = Observation()

        # Time information
        obs.time_of_day = (self.current_step * self.timestep / 3600) % 24
        obs.day_of_week = int((self.current_step * self.timestep / 86400)) % 7 + 1
        obs.day_of_year = int((self.current_step * self.timestep / 86400)) % 365 + 1

        # Current state information
        obs.extras['battery_soc'] = self.state.electrical.battery_soc
        obs.extras['ev_soc'] = self.state.electrical.ev_soc
        obs.extras['grid_power'] = self.state.electrical.grid_building

        # Forecasts (simplified - replace with actual forecasting)
        forecast_hours = 24
        obs.outdoor_temp_forecast = [20.0] * forecast_hours
        obs.solar_forecast = [0.0] * forecast_hours
        obs.price_forecast = [0.1] * forecast_hours
        obs.occupancy_forecast = [0.5] * forecast_hours

        return obs

    def _calculate_reward(self, module_outputs: Dict[str, Dict]) -> float:
        """Calculate reward for this step.

        Args:
            module_outputs: Outputs from all modules

        Returns:
            Scalar reward value
        """
        reward = 0.0

        # Get reward weights from config
        opt_config = self.config.get('optimization', {})
        objective = opt_config.get('objective', 'minimize_cost')

        if objective == 'minimize_cost':
            # Energy cost
            price = self.disturbance.prices.electricity_price
            grid_power = self.state.electrical.grid_building
            energy_cost = -abs(grid_power) * price * (self.timestep / 3600)
            reward += energy_cost

        elif objective == 'minimize_carbon':
            # Carbon emissions
            carbon_intensity = self.disturbance.prices.carbon_intensity
            grid_power = max(0, self.state.electrical.grid_building)  # Only imports
            carbon_cost = -grid_power * carbon_intensity * (self.timestep / 3600) / 1000
            reward += carbon_cost

        elif objective == 'maximize_self_consumption':
            # Self-consumption of renewable energy
            if 'pv' in module_outputs:
                pv_gen = module_outputs['pv'].get('generation', 0)
                grid_export = min(0, self.state.electrical.grid_building)
                self_consumption = pv_gen + grid_export  # Consumption = generation - export
                reward += self_consumption * 0.1

        # Add comfort penalty if available
        if 'building' in module_outputs:
            comfort_violation = module_outputs['building'].get('comfort_violation', 0)
            reward -= comfort_violation * 10

        return reward

    def get_info(self) -> Dict[str, Any]:
        """Get current environment information.

        Returns:
            Dictionary with environment status
        """
        return {
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'timestep': self.timestep,
            'episode_reward': self.episode_reward,
            'modules': list(self.modules.keys()),
            'done': self.done
        }

    def get_module(self, name: str) -> Optional[BaseModule]:
        """Get a specific module instance.

        Args:
            name: Module name

        Returns:
            Module instance or None if not found
        """
        return self.modules.get(name)

    def render(self, mode: str = 'human') -> Optional[Dict]:
        """Render the environment state.

        Args:
            mode: Rendering mode ('human' for text, 'dict' for dictionary)

        Returns:
            State dictionary if mode='dict', None otherwise
        """
        if mode == 'human':
            print(f"\n=== Step {self.current_step}/{self.max_steps} ===")
            print(f"Time: {self.observation.time_of_day:.1f}:00")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print(f"Active Modules: {', '.join(self.modules.keys())}")

            # Module states
            if 'battery' in self.modules:
                print(f"Battery SOC: {self.state.electrical.battery_soc:.1%}")
            if 'ev' in self.modules:
                print(f"EV SOC: {self.state.electrical.ev_soc:.1%}")
            if 'pv' in self.modules:
                print(f"PV Generation: {self.state.electrical.pv_building:.1f} kW")

            print(f"Grid Power: {self.state.electrical.grid_building:.1f} kW")

        elif mode == 'dict':
            return self.state.to_dict()

        return None

    def close(self) -> None:
        """Clean up resources."""
        for name, module in self.modules.items():
            if hasattr(module, 'close'):
                module.close()
                self.logger.debug(f"Closed module: {name}")

        self.modules.clear()
        self.logger.info("Environment closed")
