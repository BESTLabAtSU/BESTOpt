"""
Runtime environment that manages submodules (building, HVAC, DERs, etc.)
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Type, Tuple
from dataclasses import asdict
import time
from .base import BaseModule
from .data_structure import (
    State, Action, Disturbance, Observation, SimulationConfig,
    ThermalState, ElectricalState, WeatherData, PriceSignals, OccupancyData
)
from .constants import SIMULATION_DEFAULTS


class Environment:
    """
    Main simulation environment that coordinates all subsystem modules.
    """

    def __init__(self, config: SimulationConfig, modules: Optional[Dict[str, BaseModule]] = None):
        """
        Initialize the simulation environment.

        Args:
            config: Simulation configuration parameters
            modules: Pre-initialized modules
        """
        self.config = config
        if not config.validate():
            raise ValueError("Invalid simulation configuration")

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.Environment")

        # Simulation state
        self.current_state = State()
        self.current_disturbance = Disturbance()
        self.timestep_count = 0
        self.simulation_time = config.start_time  # hours from start of year

        # Module registry
        self.modules: Dict[str, BaseModule] = modules or {}
        self.module_order = []  # Execution order for modules

        # Data storage for analysis
        self.state_history: List[State] = []
        self.action_history: List[Action] = []
        self.disturbance_history: List[Disturbance] = []
        self.kpi_history: List[Dict[str, float]] = []

        # Performance tracking
        self.computation_times: Dict[str, List[float]] = {}
        self.convergence_info: List[Dict[str, Any]] = []

        # Simulation status
        self.is_initialized = False
        self.is_done = False

        # Initialize internal components
        self._setup_default_state()
        self._setup_logging()

    def register_module(self, name: str, module: BaseModule, execution_order: int = 0) -> None:
        """
        Register a simulation module.

        Args:
            name: Unique module name
            module: Module instance inheriting from BaseModule
            execution_order: Order in which modules are executed (lower first)
        """
        if name in self.modules:
            self.logger.warning(f"Replacing existing module: {name}")

        self.modules[name] = module

        # Update execution order
        self.module_order = sorted(
            [(n, getattr(self.modules[n], '_execution_order', execution_order))
             for n in self.modules.keys()],
            key=lambda x: x[1]
        )
        self.module_order = [name for name, _ in self.module_order]

        self.logger.info(f"Registered module '{name}' with execution order {execution_order}")

    def initialize(self) -> None:
        """Initialize all modules and prepare for simulation."""
        self.logger.info("Initializing environment and all modules...")

        # Initialize all modules
        for module_name in self.module_order:
            module = self.modules[module_name]
            start_time = time.time()

            try:
                module.initialize()
                init_time = time.time() - start_time

                self.logger.info(f"Module '{module_name}' initialized in {init_time:.3f}s")

                if module_name not in self.computation_times:
                    self.computation_times[module_name] = []

            except Exception as e:
                self.logger.error(f"Failed to initialize module '{module_name}': {e}")
                raise RuntimeError(f"Module initialization failed: {module_name}") from e

        # Validate initial state
        self._validate_state(self.current_state)

        self.is_initialized = True
        self.logger.info("Environment initialization complete")

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one simulation timestep.

        Args:
            action: Control action to apply

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if not self.is_initialized:
            raise RuntimeError("Environment must be initialized before stepping")

        if self.is_done:
            self.logger.warning("Step called on finished simulation")
            return self.get_observation(), 0.0, True, {'error': 'simulation_finished'}

        step_start_time = time.time()

        # Store action in history
        self.action_history.append(action)

        # Update disturbances first
        self._update_disturbances()

        # Execute modules in order, accumulating state updates
        module_outputs = {}
        step_info = {}

        for module_name in self.module_order:
            module = self.modules[module_name]
            module_start_time = time.time()

            try:
                # Execute module step
                output = module.step(
                    state=self.current_state,
                    action=action,
                    disturbance=self.current_disturbance,
                    timestep=self.config.timestep
                )

                module_outputs[module_name] = output
                module_time = time.time() - module_start_time
                self.computation_times[module_name].append(module_time)

            except Exception as e:
                self.logger.error(f"Module '{module_name}' step failed: {e}")
                step_info['module_errors'] = step_info.get('module_errors', [])
                step_info['module_errors'].append(f"{module_name}: {str(e)}")
                # Continue with other modules

        # Update system state based on module outputs
        self._update_state(module_outputs)

        # Calculate reward and KPIs
        reward = self._calculate_reward(action)
        kpis = self._calculate_kpis()

        # Update simulation time and check termination
        self.timestep_count += 1
        self.simulation_time += self.config.timestep

        total_hours = self.config.simulation_days * 24
        self.is_done = self.simulation_time >= (self.config.start_time + total_hours)

        # Store history
        self.state_history.append(self._copy_state(self.current_state))
        self.disturbance_history.append(self._copy_disturbance(self.current_disturbance))
        self.kpi_history.append(kpis)

        # Prepare step info
        step_time = time.time() - step_start_time
        step_info.update({
            'timestep': self.timestep_count,
            'simulation_time': self.simulation_time,
            'step_computation_time': step_time,
            'kpis': kpis,
            'module_outputs': module_outputs
        })

        # Validate final state
        try:
            self._validate_state(self.current_state)
        except ValueError as e:
            self.logger.error(f"Invalid state after step: {e}")
            step_info['state_validation_error'] = str(e)

        return self.get_observation(), reward, self.is_done, step_info

    def reset(self) -> Observation:
        """
        Reset the simulation to initial conditions.

        Returns:
            Initial observation
        """
        self.logger.info("Resetting environment...")

        # Reset simulation state
        self.timestep_count = 0
        self.simulation_time = self.config.start_time
        self.is_done = False

        # Clear history
        self.state_history.clear()
        self.action_history.clear()
        self.disturbance_history.clear()
        self.kpi_history.clear()
        self.convergence_info.clear()

        # Reset to default state
        self._setup_default_state()
        self._update_disturbances()

        # Reset all modules
        for module_name in self.modules:
            try:
                self.modules[module_name].reset()
            except Exception as e:
                self.logger.error(f"Failed to reset module '{module_name}': {e}")

        self.logger.info("Environment reset complete")
        return self.get_observation()

    def get_observation(self) -> Observation:
        """
        Get current observation for the controller.

        Returns:
            Current observation containing relevant state information
        """
        obs = Observation()

        # Current measurements
        main_zone_temp = self.current_state.thermal.zone_temperatures.get('main', 22.0)
        obs.zone_temp = main_zone_temp
        obs.zone_humidity = self.current_state.thermal.zone_humidity.get('main', 50.0)
        obs.battery_soc = self.current_state.electrical.battery_soc
        obs.ev_soc = self.current_state.electrical.ev_soc
        obs.pv_power = self.current_state.electrical.pv_power
        obs.grid_power = self.current_state.electrical.grid_power

        # Comfort requirements from occupancy
        obs.comfort_temp_min = self.current_disturbance.occupancy.comfort_temp_min
        obs.comfort_temp_max = self.current_disturbance.occupancy.comfort_temp_max

        # Time information
        obs.time_of_day = self.simulation_time % 24
        obs.day_of_week = int((self.simulation_time // 24) % 7) + 1
        obs.day_of_year = int((self.simulation_time // 24) % 365) + 1

        # Simple forecasts (in practice, these would come from disturbances modules)
        forecast_horizon = min(24, int(24 / self.config.timestep))  # 24-hour forecast

        obs.outdoor_temp_forecast = [
            self.current_disturbance.weather.outdoor_temperature + np.random.normal(0, 1)
            for _ in range(forecast_horizon)
        ]

        obs.solar_forecast = [
            max(0, self.current_disturbance.weather.solar_irradiance + np.random.normal(0, 50))
            for _ in range(forecast_horizon)
        ]

        obs.price_forecast = [
            max(0.05, self.current_disturbance.prices.electricity_price + np.random.normal(0, 0.02))
            for _ in range(forecast_horizon)
        ]

        obs.occupancy_forecast = [
            self.current_disturbance.occupancy.occupancy_fraction
            for _ in range(forecast_horizon)
        ]

        return obs

    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete current state as dictionary."""
        return {
            'state': self.current_state.to_dict(),
            'disturbances': asdict(self.current_disturbance),
            'simulation_time': self.simulation_time,
            'timestep_count': self.timestep_count,
            'is_done': self.is_done
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get simulation performance summary."""
        if not self.computation_times:
            return {'message': 'No performance data available'}

        summary = {}
        for module_name, times in self.computation_times.items():
            if times:
                summary[module_name] = {
                    'mean_time': np.mean(times),
                    'max_time': np.max(times),
                    'min_time': np.min(times),
                    'std_time': np.std(times),
                    'total_time': np.sum(times),
                    'call_count': len(times)
                }

        if self.kpi_history:
            # Calculate KPI statistics
            kpi_summary = {}
            all_kpis = {key: [] for key in self.kpi_history[0].keys()}

            for kpi_dict in self.kpi_history:
                for key, value in kpi_dict.items():
                    all_kpis[key].append(value)

            for key, values in all_kpis.items():
                kpi_summary[key] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values),
                    'final': values[-1] if values else 0.0
                }

            summary['kpis'] = kpi_summary

        return summary

    # ==================== PRIVATE METHODS ====================

    def _setup_default_state(self) -> None:
        """Initialize default system state."""
        self.current_state = State()

        # Default thermal state
        self.current_state.thermal.zone_temperatures = {'main': 22.0}
        self.current_state.thermal.zone_humidity = {'main': 50.0}
        self.current_state.thermal.hvac_supply_temp = 20.0
        self.current_state.thermal.hvac_return_temp = 22.0

        # Default electrical state
        self.current_state.electrical.battery_soc = 50.0
        self.current_state.electrical.ev_soc = 50.0
        self.current_state.electrical.grid_voltage = 240.0
        self.current_state.electrical.grid_frequency = 60.0

        # Update metadata
        self.current_state.timestamp = self.simulation_time
        self.current_state.step_count = self.timestep_count

    def _setup_logging(self) -> None:
        """Configure logging for the simulation."""
        log_level = logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('simulation.log')
            ]
        )

    def _update_disturbances(self) -> None:
        """Update disturbances values for current timestep."""
        # This is simplified - in practice, disturbances modules would provide this

        # Simple weather model (sinusoidal daily pattern)
        hour_of_day = self.simulation_time % 24
        day_of_year = (self.simulation_time // 24) % 365

        # Temperature variation
        daily_avg_temp = 20.0 + 10.0 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal
        daily_temp_variation = 8.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Daily

        self.current_disturbance.weather.outdoor_temperature = (
                daily_avg_temp + daily_temp_variation + np.random.normal(0, 1)
        )

        # Solar irradiance (simple model)
        if 6 <= hour_of_day <= 18:  # Daytime
            solar_elevation = max(0, np.sin(np.pi * (hour_of_day - 6) / 12))
            self.current_disturbance.weather.solar_irradiance = (
                    1000.0 * solar_elevation * (0.7 + 0.3 * np.random.random())
            )
        else:
            self.current_disturbance.weather.solar_irradiance = 0.0

        # Simple occupancy model
        is_weekday = ((self.simulation_time // 24) % 7) < 5
        if is_weekday and 8 <= hour_of_day <= 18:  # Work hours
            self.current_disturbance.occupancy.occupancy_fraction = 0.2
            self.current_disturbance.occupancy.is_occupied = False
        elif 18 <= hour_of_day <= 23 or 6 <= hour_of_day <= 8:  # Evening/morning
            self.current_disturbance.occupancy.occupancy_fraction = 1.0
            self.current_disturbance.occupancy.is_occupied = True
        else:
            self.current_disturbance.occupancy.occupancy_fraction = 1.0
            self.current_disturbance.occupancy.is_occupied = True

        # Simple electricity pricing (peak/off-peak)
        if 16 <= hour_of_day <= 20:  # Peak hours
            self.current_disturbance.prices.electricity_price = 0.25
        elif 22 <= hour_of_day <= 6:  # Off-peak
            self.current_disturbance.prices.electricity_price = 0.08
        else:  # Mid-peak
            self.current_disturbance.prices.electricity_price = 0.15

    def _update_state(self, module_outputs: Dict[str, Dict[str, Any]]) -> None:
        """Update system state based on module outputs."""
        # This method would merge outputs from all modules
        # For now, just update timestamp and step count
        self.current_state.timestamp = self.simulation_time
        self.current_state.step_count = self.timestep_count

        # In practice, you would merge thermal and electrical states from modules
        # Example structure for module outputs:
        # {
        #   'building_thermal': {'zone_temp': 23.2, 'hvac_power': 2.5},
        #   'battery': {'soc': 65.3, 'power': -1.2},
        #   'pv': {'power': 4.8, 'voltage': 245.0}
        # }

    def _calculate_reward(self, action: Action) -> float:
        """Calculate reward signal for RL training."""
        reward = 0.0

        # Comfort penalty
        zone_temp = self.current_state.thermal.zone_temperatures.get('main', 22.0)
        comfort_min = self.current_disturbance.occupancy.comfort_temp_min
        comfort_max = self.current_disturbance.occupancy.comfort_temp_max

        if zone_temp < comfort_min:
            reward -= 10.0 * (comfort_min - zone_temp) ** 2
        elif zone_temp > comfort_max:
            reward -= 10.0 * (zone_temp - comfort_max) ** 2

        # Energy cost penalty
        grid_power = abs(self.current_state.electrical.grid_power)
        electricity_price = self.current_disturbance.prices.electricity_price
        energy_cost = grid_power * self.config.timestep * electricity_price
        reward -= energy_cost

        # Battery cycling penalty
        battery_power = abs(self.current_state.electrical.battery_power)
        reward -= 0.01 * battery_power  # Small cycling cost

        return reward

    def _calculate_kpis(self) -> Dict[str, float]:
        """Calculate Key Performance Indicators."""
        kpis = {}

        # Energy KPIs
        kpis['grid_energy'] = abs(self.current_state.electrical.grid_power) * self.config.timestep
        kpis['pv_energy'] = self.current_state.electrical.pv_power * self.config.timestep
        kpis['battery_energy'] = abs(self.current_state.electrical.battery_power) * self.config.timestep

        # Comfort KPIs
        zone_temp = self.current_state.thermal.zone_temperatures.get('main', 22.0)
        comfort_min = self.current_disturbance.occupancy.comfort_temp_min
        comfort_max = self.current_disturbance.occupancy.comfort_temp_max

        kpis['comfort_violation'] = max(0, zone_temp - comfort_max) + max(0, comfort_min - zone_temp)
        kpis['zone_temperature'] = zone_temp

        # Economic KPIs
        electricity_price = self.current_disturbance.prices.electricity_price
        kpis['electricity_cost'] = (
                max(0, self.current_state.electrical.grid_power) *
                self.config.timestep * electricity_price
        )

        # Efficiency KPIs
        total_load = (self.current_state.electrical.hvac_power +
                      self.current_state.electrical.lighting_power +
                      self.current_state.electrical.plug_loads_power)

        if self.current_state.electrical.pv_power > 0:
            kpis['pv_utilization'] = min(1.0, total_load / self.current_state.electrical.pv_power)
        else:
            kpis['pv_utilization'] = 0.0

        return kpis

    def _validate_state(self, state: State) -> None:
        """Validate state values are within reasonable bounds."""
        # Temperature bounds
        for zone, temp in state.thermal.zone_temperatures.items():
            if not (-10 <= temp <= 50):
                raise ValueError(f"Zone {zone} temperature {temp}°C out of bounds")

        # SOC bounds
        if not (0 <= state.electrical.battery_soc <= 100):
            raise ValueError(f"Battery SOC {state.electrical.battery_soc}% out of bounds")

        if not (0 <= state.electrical.ev_soc <= 100):
            raise ValueError(f"EV SOC {state.electrical.ev_soc}% out of bounds")

    def _copy_state(self, state: State) -> State:
        """Create a deep copy of the state."""
        # Simple implementation - in practice, you might need more sophisticated copying
        import copy
        return copy.deepcopy(state)

    def _copy_disturbance(self, disturbance: Disturbance) -> Disturbance:
        """Create a deep copy of the disturbances."""
        import copy
        return copy.deepcopy(disturbance)