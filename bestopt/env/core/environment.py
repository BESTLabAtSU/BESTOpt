"""
Runtime environment for bestopt framework.
Manages module execution and simulation state.
"""

import logging
import importlib
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from .base import BaseModule
from .data_structure import (
    State, Action, Disturbance, Observation,
    BatteryState, PVState, BLDGEState, EVState,
    HVACState, BLDGTState, TESState,
    WaterHeaterState,
    ThermalAction, ElectricalAction, WaterAction
)
import torch
import random
import os
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
seed_value = 142857
set_seed(seed_value)

class BESTOptEnvironment:
    """Runtime environment that manages all modules."""

    def __init__(self, configuration: Dict[str, Any]):
        """Initialize environment with configuration.

        Args:
            configuration: Configuration dictionary (from ConfigurationManager.get_final_configuration())
            expected keys:
               - buildings: {<building_id>: {components...}}
               - controllers: {<controller_key>: {class_path, parameters}}
               -- active_controllers: {<building_id>: <controller_key>}
               - disturbances: {<name>: {class_path, parameters}}
               - environment: {parameters: {resolution, duration, ...}}
        """
        self.config = configuration
        self.logger = logging.getLogger("BestOptEnvironment")

        # Environment config
        env_config = self.config.get('environment', {})
        env_params = env_config.get('parameters', {})

        self.res = env_params.get("resolution")
        if self.res is None:
            raise ValueError("Missing required config key: 'resolution'")
        if not isinstance(self.res, int) or self.res <= 0:
            raise ValueError(f"'resolution' must be a positive int (seconds); got {self.res}")

        self.dur = env_config.get("duration", 24 * 60 * 60)  # Run 1 day simulation if 'duration' is missing
        if not isinstance(self.dur, int) or self.dur <= 0:
            raise ValueError(f"'duration' must be a positive int (seconds); got {self.dur}")

        # Warning if duration not divisible, only accept 'int' steps
        self.total_step = self.dur // self.res
        if self.dur % self.res != 0:
            self.logger.warning(
                f"duration ({self.dur}) not divisible by resolution ({self.res}); "
                f"sim will run {self.total_step} steps (= floor)."
            )

        self.simulation_start_time = configuration.get('environment', {}).get('parameters', {}).get(
            'simulation_start_time', '2024-01-01 00:00:00')

        # Get building configurations
        buildings_config = self.config.get('buildings', {})
        if not buildings_config:
            raise ValueError("No buildings configured in environment")

        # Initialize simulation data state
        self.states: Dict[str, State] = {}
        for building_id in buildings_config.keys():
            self.states[building_id] = State(building_id=building_id)
            self.logger.info(f"Initialized state for building: {building_id}")

        self.observations: Dict[str, Observation] = {
            bldg: Observation() for bldg in buildings_config.keys()
        }
        self.disturbances: Dict[str, Disturbance] = {
            bldg: Disturbance() for bldg in buildings_config.keys()
        }
        self.actions: Dict[str, Action] = {
            bldg: Action() for bldg in buildings_config.keys()
        }

        # Tracking
        self.current_step = 0
        self.done = False

        # Setup hierarchical structure for multi-scale-components
        self.building_modules: Dict[str, Dict[str, Dict[str, BaseModule]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.disturbance_modules: Dict[str, BaseModule] = {}
        self.controllers: Dict[str, BaseModule] = {}
        self.local_controllers: Dict[str, BaseModule] = {}
        self.grid_module: Optional[BaseModule] = None  # @TODO

        # Build runtime
        self._generate_components()  # include modules/controller/disturbance
        self._initialize_component_states()
        self._validate_controller_assignments()
        self.logger.info(f"Environment configuration finished")

    # Module generation
    def _generate_components(self) -> None:
        """Generate all components (modules, controllers, disturbances)."""
        # Generate modules
        buildings_config = self.config.get('buildings', {})
        thermal_modules_to_warmup = []
        for building_id, building_info in buildings_config.items():
            # Process each component type in the building
            for component_type in ['batteries', 'pv_systems', 'hvac_systems',
                                   'thermal_zones']:  # TODO need to be adaptive
                components = building_info.get(component_type, {})
                for component_id, component_config in components.items():
                    instance = self._create_instance(
                        name=component_id,
                        config=component_config,
                        component_type=f"{component_type}.{component_id}",
                        must_subclass=BaseModule
                    )
                    if instance:
                        self.building_modules[building_id][component_type][component_id] = instance
                        component_path = f"{building_id}.{component_type}.{component_id}"
                        self._create_local_controller_if_exists(component_path, instance)
                        if component_type == 'thermal_zones':
                            thermal_modules_to_warmup.append((building_id, component_id, instance))

        self._warmup_thermal_modules(thermal_modules_to_warmup)
        # Generate controller
        controllers_config = self.config.get('controllers', {})
        active_controllers = self.config.get('active_controllers', {})

        for controller_key, controller_info in controllers_config.items():
            instance = self._create_instance(
                name=controller_key,
                config=controller_info,
                component_type="controller",
                must_subclass=BaseModule
            )
            if instance:
                self.controllers[controller_key] = instance
        if not active_controllers:
            self.logger.warning("No active controllers assigned to buildings")

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
                self.disturbance_modules[dist_name] = instance

    def _create_local_controller_if_exists(self, component_path: str, module_instance: BaseModule) -> None:
        """Create local controller for a component if configured."""
        local_controllers_config = self.config.get('local_controllers', {})

        if component_path in local_controllers_config:
            controller_config = local_controllers_config[component_path]
            controller_instance = self._create_instance(
                name=f"{component_path}_local",
                config=controller_config,
                component_type="local_controller",
                must_subclass=BaseModule
            )
            if controller_instance:
                self.local_controllers[component_path] = controller_instance
                module_instance.local_controller = controller_instance
                self.logger.info(f"Attached local controller to {component_path}")

    def _warmup_thermal_modules(self, thermal_modules: List[Tuple[str, str, BaseModule]]) -> None:
        """Warmup thermal dynamics modules immediately after creation."""
        self.logger.info(f"Starting automatic warmup for {len(thermal_modules)} thermal dynamics modules...")

        for building_id, component_id, module in thermal_modules:
            try:
                module_name = f"{building_id}.{component_id}"
                self.logger.info(f"Warming up thermal module: {module_name}")

                # Prepare the module for simulation
                module.prepare_for_simulation(self.simulation_start_time)

                # Log warmup status
                if hasattr(module, 'get_state_summary'):
                    summary = module.get_state_summary()
                    buffer_size = summary.get('buffer_size', 0)
                    self.logger.info(f"Warmup completed for {module_name}: {buffer_size} timesteps loaded")

            except Exception as e:
                self.logger.error(f"Failed to warmup {module_name}: {e}")
                raise RuntimeError(f"Thermal dynamics warmup failed for {module_name}: {e}")

        self.logger.info(f"Successfully warmed up {len(thermal_modules)} thermal dynamics modules")

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

    def _initialize_component_states(self) -> None:
        """Initialize state objects for all created modules."""
        buildings_config = self.config.get('buildings', {})

        for building_id, building_info in buildings_config.items():
            state = self.states[building_id]

            # Initialize electrical component states
            for component_type in ['batteries', 'pv_systems', 'bldg_e_loads']:
                components = building_info.get(component_type, {})
                for component_id, component_config in components.items():
                    if component_type == 'batteries':
                        # Create battery state from config
                        battery_config = component_config.get('parameters', {})
                        initial_soc = battery_config.get('initial_soc', 0.5)
                        state.electrical.batteries[component_id] = BatteryState(
                            component_id=component_id,
                            battery_soc=initial_soc
                        )
                    elif component_type == 'pv_systems':
                        state.electrical.pv_systems[component_id] = PVState(
                            component_id=component_id
                        )
                    elif component_type == 'bldg_e_loads':
                        state.electrical.bldg_e_loads[component_id] = BLDGEState(
                            component_id=component_id
                        )

            # Initialize thermal component states
            for component_type in ['hvac_systems', 'thermal_zones', 'thermal_storage']:
                components = building_info.get(component_type, {})
                for component_id, component_config in components.items():
                    if component_type == 'hvac_systems':
                        state.thermal.hvac_systems[component_id] = HVACState(
                            component_id=component_id
                        )
                    elif component_type == 'thermal_zones':
                        # Initialize with default or config values
                        zone_config = component_config.get('parameters', {})
                        initial_temp = zone_config.get('initial_temperature', 22.0)
                        state.thermal.thermal_zones[component_id] = BLDGTState(
                            component_id=component_id,
                            temperature=initial_temp
                        )
                    elif component_type == 'thermal_storage':
                        tes_config = component_config.get('parameters', {})
                        initial_temp = tes_config.get('initial_temperature', 22.0)
                        initial_soc = tes_config.get('initial_soc', 0.5)
                        state.thermal.thermal_storage[component_id] = TESState(
                            component_id=component_id,
                            temperature=initial_temp,
                            tes_soc=initial_soc
                        )

            # Initialize water component states
            for component_type in ['water_heaters']:
                components = building_info.get(component_type, {})
                for component_id, component_config in components.items():
                    if component_type == 'water_heaters':
                        heater_config = component_config.get('parameters', {})
                        initial_temp = heater_config.get('initial_temperature', 60.0)
                        state.water.water_heaters[component_id] = WaterHeaterState(
                            component_id=component_id,
                            tank_temperature=initial_temp
                        )

    def _validate_controller_assignments(self):
        """Validate that controller assignments match the expected domain structure."""
        active_controllers = self.config.get('active_controllers', {})

        for building_id in self.states.keys():
            if building_id not in active_controllers:
                self.logger.warning(f"Building {building_id} has no domain controllers assigned")
            else:
                domains_assigned = list(active_controllers[building_id].keys())
                self.logger.info(f"Building {building_id} has controllers for domains: {domains_assigned}")

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

    def reset(self) -> Dict[str, Observation]:
        """
        Reset the environment to initial state.

        Returns:
            Dictionary of observations for each building
        """
        self.logger.info("Resetting environment")

        # Reset tracking
        self.current_step = 0
        self.done = False

        # Reset all building modules
        for building_id, building_modules in self.building_modules.items():
            for component_type, components in building_modules.items():
                self._reset_components(components, f"{building_id}.{component_type}")

        # Reset controllers and disturbances
        self._reset_components(self.controllers, "controller")
        self._reset_components(self.local_controllers, "local_controller")
        self._reset_components(self.disturbance_modules, "disturbance_modules")

        # Reset states for each building
        for building_id in self.states.keys():
            self.states[building_id] = State(building_id=building_id)
            self.observations[building_id] = Observation()
            self.disturbances[building_id] = Disturbance()
            self.actions[building_id] = Action()

        self._initialize_component_states()

        return self.observations  # TODO return the initial observation

    def step(self, external_actions: Optional[Dict[str, Dict[str, Any]]] = None) -> (
            Tuple)[Dict[str, Observation], bool, Dict[str, Any]]:
        """Execute one simulation step.

        Args:
            external_actions: Optional external actions per building per domain
                            e.g., {"building_1": {"electrical": ElectricalAction(...),
                                                  "thermal": ThermalAction(...)}}
                            If None, uses internal controllers

        Returns:
            Tuple of (observations, done, info):
            - observations: Current system observations for each building
            - done: Whether simulation is complete
            - info: Additional simulation information
        """
        # Check simulation status
        if self.done:
            self.logger.warning("Environment is done. Call reset() to restart.")
            return self.observations, True, {}

        # Process each building
        for building_id in self.states.keys():
            # Get actions for this building from domain controllers
            building_action = self._get_building_actions(
                building_id,
                external_actions.get(building_id) if external_actions else None
            )

            # Store the combined action
            self.actions[building_id] = building_action

            # Update disturbances
            self._update_building_disturbances(building_id)

            # Execute building modules with the actions
            self._execute_building_modules(building_id)

            # Update observations for next step
            self._update_building_observation(building_id)

        # Update simulation tracking
        self.current_step += 1
        if self.current_step >= self.total_step:
            self.done = True
            self.logger.info(f"Simulation completed after {self.current_step} steps")

        # Collect step info
        info = self._collect_step_info()

        return self.observations, self.done, info

    def _get_building_actions(self,
                              building_id: str,
                              external_actions: Optional[Dict[str, Any]] = None) -> Action:
        """Get control actions for a building from domain controllers or external input.

        Args:
            building_id: ID of the building
            external_actions: Optional external actions for this building's domains

        Returns:
            Combined Action object with all domain actions
        """
        # Initialize domain actions
        thermal_action = ThermalAction()
        electrical_action = ElectricalAction()
        water_action = WaterAction()

        if external_actions:
            # Use provided external actions
            if "thermal" in external_actions:
                thermal_action = external_actions["thermal"]
            if "electrical" in external_actions:
                electrical_action = external_actions["electrical"]
            if "water" in external_actions:
                water_action = external_actions["water"]
        else:
            # Use internal domain controllers
            active_controllers = self.config.get('active_controllers', {})
            building_controllers = active_controllers.get(building_id, {})

            # Get thermal action
            if "thermal" in building_controllers:
                thermal_action = self._get_domain_action(
                    building_id, "thermal", building_controllers["thermal"]
                )

            # Get electrical action
            if "electrical" in building_controllers:
                electrical_action = self._get_domain_action(
                    building_id, "electrical", building_controllers["electrical"]
                )

            # Get water action
            if "water" in building_controllers:
                water_action = self._get_domain_action(
                    building_id, "water", building_controllers["water"]
                )

        # Combine into single Action object
        return Action(
            thermal=thermal_action,
            electrical=electrical_action,
            water=water_action
        )

    def _get_domain_action(self,
                           building_id: str,
                           domain: str,
                           controller_key: str) -> Any:
        """Get action from a specific domain controller.

        Args:
            building_id: ID of the building
            domain: Domain name ('thermal', 'electrical', 'water')
            controller_key: Key to the controller instance

        Returns:
            Domain-specific action object
        """
        if controller_key not in self.controllers:
            self.logger.warning(f"Controller {controller_key} not found")
            # Return default action for the domain
            if domain == "thermal":
                return ThermalAction()
            elif domain == "electrical":
                return ElectricalAction()
            elif domain == "water":
                return WaterAction()

        controller = self.controllers[controller_key]

        try:
            # Pass domain-specific state and observation to the controller
            domain_state = getattr(self.states[building_id], domain)

            # Call controller's step method
            action = controller.step(
                state=domain_state,
                observation=self.observations[building_id],
                disturbance=self.disturbances[building_id],
                timestep=self.current_step
            )

            return action

        except Exception as e:
            self.logger.error(f"Error getting action from {domain} controller for {building_id}: {e}")
            # Return default action on error
            if domain == "thermal":
                return ThermalAction()
            elif domain == "electrical":
                return ElectricalAction()
            elif domain == "water":
                return WaterAction()

    def _update_building_disturbances(self, building_id: str) -> None:
        """Update disturbances for a building.

        Args:
            building_id: ID of the building
        """
        # Update from each disturbance module
        for dist_name, dist_module in self.disturbance_modules.items():
            try:
                # Get disturbance update
                dist_update = dist_module.step(
                    current_step=self.current_step
                )

                # Update specific fields based on disturbance type
                if dist_name == "weather":
                    self.disturbances[building_id].weather = dist_update
                elif dist_name == "electricity_prices":
                    self.disturbances[building_id].prices = dist_update
                elif dist_name == "occupancy":
                    self.disturbances[building_id].occupancy = dist_update

            except Exception as e:
                self.logger.error(f"Disturbance {dist_name} update failed for {building_id}: {e}")

    def _execute_building_modules(self, building_id: str) -> None:
        """Execute all modules for a building with domain-specific actions.

        Args:
            building_id: ID of the building
        """
        # @TODO not decide yet, if we should pass whole state or just specific state when we excute the module, or just hybrid
        building_modules = self.building_modules.get(building_id, {})
        action = self.actions[building_id]
        state = self.states[building_id]
        disturbance = self.disturbances[building_id]

        # Execute electrical domain modules
        if "batteries" in building_modules:
            for battery_id, battery_module in building_modules["batteries"].items():
                try:
                    # @TODO need to adaptive
                    battery_module.step(
                        state=state.electrical.batteries[battery_id],
                        action=action.electrical,
                        disturbance=disturbance,
                        timestep=self.current_step
                    )
                except Exception as e:
                    self.logger.error(f"Battery {battery_id} step failed: {e}")

        if "pv_systems" in building_modules:
            for pv_id, pv_module in building_modules["pv_systems"].items():
                try:
                    # @TODO need to adaptive
                    pv_module.step(
                        state=state.electrical.pv_systems[pv_id],
                        action=action.electrical,  # Pass electrical action
                        disturbance=disturbance,
                        resolution=self.res,
                        timestep=self.current_step
                    )
                except Exception as e:
                    self.logger.error(f"PV {pv_id} step failed: {e}")

        # Execute thermal domain modules
        if "hvac_systems" in building_modules:
            for hvac_id, hvac_module in building_modules["hvac_systems"].items():
                try:
                    supervisory_action = action.thermal

                    if hasattr(hvac_module, 'local_controller'):
                        # Use local controller to translate supervisory action to component action
                        local_action = hvac_module.local_controller.step(
                            supervisory_action=supervisory_action,
                            state=state.thermal.hvac_systems[hvac_id],
                            disturbance=disturbance,
                            timestep=self.current_step
                        )
                    else:
                        # No local controller, pass supervisory action directly
                        local_action = supervisory_action
                    hvac_module.step(
                        state=state.thermal.hvac_systems[hvac_id],
                        action=local_action,
                        disturbance=disturbance,
                        timestep=self.current_step
                    )
                except Exception as e:
                    self.logger.error(f"HVAC {hvac_id} step failed: {e}")



        if "thermal_zones" in building_modules:
            for zone_id, zone_module in building_modules["thermal_zones"].items():
                try:
                    # wrap up thermal load
                    # @TODO need to match ID
                    thermal_load = (13 - state.thermal.thermal_zones[zone_id].temperature)*hvac_module.current_flow_rate*1.2*1005
                    # @Need rename to thermal load in action data structure, check later
                    action.thermal.hvac_power = thermal_load
                    zone_module.step(
                        state=state.thermal.thermal_zones[zone_id],
                        action=action.thermal,  # Pass thermal action
                        disturbance=disturbance,
                        timestep=self.current_step
                    )
                except Exception as e:
                    self.logger.error(f"Thermal zone {zone_id} step failed: {e}")

        # Execute water domain modules
        if "water_heaters" in building_modules:
            for heater_id, heater_module in building_modules["water_heaters"].items():
                try:
                    heater_module.step(
                        state=state.water.water_heaters[heater_id],
                        action=action.water,  # Pass water action
                        disturbance=disturbance,
                        resolution=self.res,
                        timestep=self.current_step
                    )
                except Exception as e:
                    self.logger.error(f"Water heater {heater_id} step failed: {e}")

        # Update state aggregations after all modules have executed
        state.update_all_aggregations()

    def _update_building_observation(self, building_id: str) -> None:
        """Update observation for a building based on current state.

        Args:
            building_id: ID of the building
        """
        state = self.states[building_id]
        disturbance = self.disturbances[building_id]

        # Create new observation from current state
        obs = Observation()

        # Add time information
        obs.time_of_day = (self.current_step * self.res / 3600) % 24  # hours
        obs.day_of_week = int((self.current_step * self.res / 86400)) % 7 + 1
        obs.day_of_year = int((self.current_step * self.res / 86400)) % 365 + 1

        # Add forecasts (these would come from disturbance modules)
        # This is a simplified example - you'd get these from forecast modules
        obs.outdoor_temp_forecast = [disturbance.weather.outdoor_temperature] * 4
        obs.solar_forecast = [disturbance.weather.solar_radiation] * 4
        obs.price_forecast = [disturbance.prices.electricity_price] * 4
        obs.occupancy_forecast = [disturbance.occupancy.occupancy_fraction] * 4

        # Add comfort references
        obs.comfort_temp_min = disturbance.occupancy.comfort_temp_min
        obs.comfort_temp_max = disturbance.occupancy.comfort_temp_max

        # Add current state info to extras
        obs.extras = {
            "electrical_load": state.electrical.total_demand,
            "thermal_load": state.thermal.total_heating_load + state.thermal.total_cooling_load,
            "grid_import": state.electrical.grid_import,
            "grid_export": state.electrical.grid_export
        }

        self.observations[building_id] = obs

    def _collect_step_info(self) -> Dict[str, Any]:
        """Collect information about the current step.

        Returns:
            Dictionary with step information
        """
        info = {
            "step": self.current_step,
            "time_hours": self.current_step * self.res / 3600,
            "buildings": {}
        }

        # Add per-building info
        for building_id in self.states.keys():
            state = self.states[building_id]
            action = self.actions[building_id]

            info["buildings"][building_id] = {
                # @ Refine later, what info are we going to collect?
                "electrical": {
                    "total_generation": state.electrical.total_generation,
                    "total_demand": state.electrical.total_demand,
                    "grid_import": state.electrical.grid_import,
                    "grid_export": state.electrical.grid_export,
                },
                "thermal": {
                    "heating_load": state.thermal.total_heating_load,
                    "cooling_load": state.thermal.total_cooling_load,
                },
                "water": {
                    "heating_power": state.water.total_water_heating_power,
                }
            }

        return info
