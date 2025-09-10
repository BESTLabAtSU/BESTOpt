# Building Simulation Framework
# Project Structure and Base Classes

"""
Project Structure:
building_simulation/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base.py
│   └── loops.py
├── runtime/
│   ├── __init__.py
│   ├── hvac/
│   │   ├── __init__.py
│   │   ├── source.py
│   │   ├── distribution.py
│   │   └── terminal.py
│   ├── ders/
│   │   ├── __init__.py
│   │   ├── pv.py
│   │   ├── battery.py
│   │   ├── ev.py
│   │   └── thermal_storage.py
│   ├── building/
│   │   ├── __init__.py
│   │   └── dynamic.py
│   └── appliance/
│       ├── __init__.py
│       └── appliance.py
├── controller/
│   ├── __init__.py
│   ├── base.py
│   ├── pid.py
│   └── mpc.py
├── disturbances/
│   ├── __init__.py
│   ├── weather.py
│   ├── price.py
│   └── predictor.py
└── utils/
    ├── __init__.py
    └── helpers.py
"""

# core/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class SimulationModule(ABC):
    """Base class for all simulation modules."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the module with configuration."""
        pass

    @abstractmethod
    def step(self, control_action: Dict[str, float],
             disturbance: Dict[str, float],
             dt: float) -> Dict[str, float]:
        """Perform one simulation step."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, float]:
        """Get current state of the module."""
        pass

    def reset(self) -> None:
        """Reset module to initial state."""
        self.initialize()


class DynamicModule(SimulationModule):
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


class StaticModule(SimulationModule):
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


# core/loops.py
class ThermalLoop:
    """Manages thermal energy transfer between modules."""

    def __init__(self):
        self.connections = {}
        self.thermal_flows = {}

    def connect(self, source: str, sink: str, capacity: float):
        """Connect two modules in the thermal loop."""
        if source not in self.connections:
            self.connections[source] = []
        self.connections[source].append({'sink': sink, 'capacity': capacity})

    def calculate_flows(self, module_states: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate thermal flows between connected modules."""
        flows = {}
        for source, sinks in self.connections.items():
            source_temp = module_states[source].get('temperature', 20.0)
            for sink_info in sinks:
                sink = sink_info['sink']
                capacity = sink_info['capacity']
                sink_temp = module_states[sink].get('temperature', 20.0)
                flow = capacity * (source_temp - sink_temp)
                flows[f"{source}_to_{sink}"] = flow
        return flows


class ElectricLoop:
    """Manages electrical power transfer between modules."""

    def __init__(self):
        self.connections = {}
        self.power_flows = {}

    def connect(self, source: str, sink: str, max_power: float):
        """Connect two modules in the electric loop."""
        if source not in self.connections:
            self.connections[source] = []
        self.connections[source].append({'sink': sink, 'max_power': max_power})

    def balance_power(self, generation: Dict[str, float],
                      demand: Dict[str, float]) -> Dict[str, float]:
        """Balance power between generation and demand."""
        total_generation = sum(generation.values())
        total_demand = sum(demand.values())

        if total_generation >= total_demand:
            return {'grid_import': 0, 'grid_export': total_generation - total_demand}
        else:
            return {'grid_import': total_demand - total_generation, 'grid_export': 0}


# runtime/building/dynamic.py
class BuildingDynamics(DynamicModule):
    """Building thermal dynamics module."""

    def initialize(self):
        self.state = {
            'indoor_temperature': self.config.get('initial_temp', 20.0),
            'wall_temperature': self.config.get('initial_temp', 20.0)
        }
        self.thermal_mass = self.config.get('thermal_mass', 1e6)  # J/K
        self.ua_value = self.config.get('ua_value', 500)  # W/K
        self.initialized = True

    def state_transition(self, current_state: Dict[str, float],
                         control_action: Dict[str, float],
                         disturbance: Dict[str, float],
                         dt: float) -> Dict[str, float]:
        """Calculate building temperature dynamics."""
        T_in = current_state['indoor_temperature']
        T_wall = current_state['wall_temperature']
        T_out = disturbance.get('outdoor_temperature', 15.0)
        Q_hvac = control_action.get('hvac_power', 0.0)
        Q_solar = disturbance.get('solar_gain', 0.0)

        # Simple lumped capacitance model
        dT_in_dt = (Q_hvac + Q_solar - self.ua_value * (T_in - T_out)) / self.thermal_mass

        new_state = {
            'indoor_temperature': T_in + dT_in_dt * dt,
            'wall_temperature': T_wall + 0.1 * (T_in - T_wall) * dt
        }

        return new_state

    def get_output(self) -> Dict[str, float]:
        return {
            'indoor_temperature': self.state['indoor_temperature'],
            'thermal_demand': self.ua_value * (self.state['indoor_temperature'] - 15.0)
        }


# runtime/ders/pv.py
class PVSystem(StaticModule):
    """Photovoltaic system module."""

    def initialize(self):
        self.capacity = self.config.get('capacity', 5000)  # W
        self.efficiency = self.config.get('efficiency', 0.2)
        self.area = self.config.get('area', 25)  # m²
        self.initialized = True

    def calculate_output(self, control_action: Dict[str, float],
                         disturbance: Dict[str, float]) -> Dict[str, float]:
        """Calculate PV power output."""
        irradiance = disturbance.get('solar_irradiance', 0)  # W/m²
        temperature = disturbance.get('ambient_temperature', 25)  # °C
        curtailment = control_action.get('curtailment', 0)  # 0-1

        # Simple PV model with temperature derating
        temp_coefficient = -0.004  # per °C
        temp_factor = 1 + temp_coefficient * (temperature - 25)

        power = self.area * irradiance * self.efficiency * temp_factor * (1 - curtailment)
        power = min(power, self.capacity)

        return {'power_output': power, 'available_power': power / (1 - curtailment)}


# runtime/ders/battery.py
class Battery(DynamicModule):
    """Battery energy storage module."""

    def initialize(self):
        self.capacity = self.config.get('capacity', 10000)  # Wh
        self.max_power = self.config.get('max_power', 5000)  # W
        self.efficiency = self.config.get('efficiency', 0.95)
        self.state = {
            'soc': self.config.get('initial_soc', 0.5),  # State of charge (0-1)
            'power': 0.0
        }
        self.initialized = True

    def state_transition(self, current_state: Dict[str, float],
                         control_action: Dict[str, float],
                         disturbance: Dict[str, float],
                         dt: float) -> Dict[str, float]:
        """Update battery state of charge."""
        soc = current_state['soc']
        power_setpoint = control_action.get('power_setpoint', 0)  # Positive = discharge

        # Apply power limits
        power = np.clip(power_setpoint, -self.max_power, self.max_power)

        # Apply SOC limits
        if power > 0 and soc <= 0.1:  # Discharging
            power = 0
        elif power < 0 and soc >= 0.9:  # Charging
            power = 0

        # Update SOC
        energy_change = power * dt / 3600  # Wh
        if power < 0:  # Charging
            energy_change *= self.efficiency
        else:  # Discharging
            energy_change /= self.efficiency

        new_soc = soc - energy_change / self.capacity
        new_soc = np.clip(new_soc, 0, 1)

        return {'soc': new_soc, 'power': power}

    def get_output(self) -> Dict[str, float]:
        return {
            'soc': self.state['soc'],
            'power': self.state['power'],
            'energy_stored': self.state['soc'] * self.capacity
        }


# runtime/hvac/source.py
class HeatPump(StaticModule):
    """Heat pump module for heating/cooling."""

    def initialize(self):
        self.capacity = self.config.get('capacity', 10000)  # W
        self.cop_heating = self.config.get('cop_heating', 3.5)
        self.cop_cooling = self.config.get('cop_cooling', 4.0)
        self.initialized = True

    def calculate_output(self, control_action: Dict[str, float],
                         disturbance: Dict[str, float]) -> Dict[str, float]:
        """Calculate heat pump performance."""
        mode = control_action.get('mode', 'off')  # 'heating', 'cooling', 'off'
        load_fraction = control_action.get('load_fraction', 0)  # 0-1
        outdoor_temp = disturbance.get('outdoor_temperature', 15)

        if mode == 'off':
            return {'thermal_output': 0, 'power_consumption': 0}

        # Adjust COP based on outdoor temperature
        if mode == 'heating':
            cop = self.cop_heating * (1 - 0.02 * (20 - outdoor_temp))
        else:  # cooling
            cop = self.cop_cooling * (1 - 0.02 * (outdoor_temp - 25))

        thermal_output = self.capacity * load_fraction
        power_consumption = thermal_output / cop

        return {
            'thermal_output': thermal_output,
            'power_consumption': power_consumption,
            'cop': cop
        }


# controller/base.py
class Controller(ABC):
    """Base controller class."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.setpoint = config.get('setpoint', {})

    @abstractmethod
    def compute_control(self, state: Dict[str, float],
                        reference: Dict[str, float],
                        disturbance: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Compute control action based on state, reference, and disturbances."""
        pass

    def update_setpoint(self, setpoint: Dict[str, float]):
        """Update controller setpoint."""
        self.setpoint.update(setpoint)


# controller/pid.py
class PIDController(Controller):
    """PID controller implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kp = config.get('kp', 1.0)
        self.ki = config.get('ki', 0.1)
        self.kd = config.get('kd', 0.01)
        self.integral = 0
        self.last_error = 0
        self.output_limits = config.get('output_limits', (-np.inf, np.inf))

    def compute_control(self, state: Dict[str, float],
                        reference: Dict[str, float],
                        disturbance: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Compute PID control action."""
        # Simple single-variable PID for demonstration
        controlled_var = list(reference.keys())[0]
        error = reference[controlled_var] - state.get(controlled_var, 0)

        # PID calculation
        self.integral += error
        derivative = error - self.last_error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Apply limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        self.last_error = error

        return {'control_output': output}


# disturbances/weather.py
class WeatherDisturbance:
    """Weather disturbances generator."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_source = config.get('data_source', 'synthetic')
        self.data = None

    def load(self, filepath: str):
        """Load weather data from file."""
        # Placeholder for loading weather data
        pass

    def get_disturbance(self, timestamp: float) -> Dict[str, float]:
        """Get weather disturbances at given timestamp."""
        # Simple synthetic weather for demonstration
        hour = (timestamp / 3600) % 24

        # Temperature: sinusoidal daily pattern
        outdoor_temp = 20 + 5 * np.sin(2 * np.pi * (hour - 6) / 24)

        # Solar irradiance: peaked at noon
        if 6 <= hour <= 18:
            solar_irradiance = 800 * np.sin(np.pi * (hour - 6) / 12)
        else:
            solar_irradiance = 0

        return {
            'outdoor_temperature': outdoor_temp,
            'solar_irradiance': solar_irradiance,
            'wind_speed': 5 + 2 * np.random.random()
        }


# disturbances/price.py
class PriceDisturbance:
    """Electricity price disturbances generator."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.peak_price = config.get('peak_price', 0.25)  # $/kWh
        self.off_peak_price = config.get('off_peak_price', 0.10)  # $/kWh

    def get_disturbance(self, timestamp: float) -> Dict[str, float]:
        """Get price signal at given timestamp."""
        hour = (timestamp / 3600) % 24

        # Simple time-of-use pricing
        if 14 <= hour <= 20:  # Peak hours
            price = self.peak_price
        else:
            price = self.off_peak_price

        return {'electricity_price': price}


# Example usage
if __name__ == "__main__":
    # Create building dynamics
    building_config = {
        'name': 'Office Building',
        'thermal_mass': 5e6,
        'ua_value': 1000,
        'initial_temp': 22
    }
    building = BuildingDynamics(building_config)
    building.initialize()

    # Create PV system
    pv_config = {
        'name': 'Rooftop PV',
        'capacity': 50000,
        'area': 250
    }
    pv = PVSystem(pv_config)
    pv.initialize()

    # Create battery
    battery_config = {
        'name': 'Battery Storage',
        'capacity': 100000,
        'max_power': 50000,
        'initial_soc': 0.5
    }
    battery = Battery(battery_config)
    battery.initialize()

    # Create controller
    controller_config = {
        'name': 'Temperature Controller',
        'kp': 1000,
        'ki': 10,
        'kd': 100,
        'output_limits': (-10000, 10000)
    }
    controller = PIDController(controller_config)

    # Create disturbances
    weather = WeatherDisturbance({})
    price = PriceDisturbance({})

    # Simulation loop example
    dt = 300  # 5 minute timestep
    timestamp = 0

    for step in range(10):
        # Get disturbances
        weather_dist = weather.get_disturbance(timestamp)
        price_dist = price.get_disturbance(timestamp)

        # Get building state
        building_state = building.get_state()

        # Compute control
        control = controller.compute_control(
            building_state,
            {'indoor_temperature': 22},
            weather_dist
        )

        # Apply control to building
        building_output = building.step(
            {'hvac_power': control['control_output']},
            weather_dist,
            dt
        )

        # PV generation
        pv_output = pv.step({}, weather_dist, dt)

        # Battery operation
        battery_control = {'power_setpoint': pv_output['power_output'] - control['control_output']}
        battery_output = battery.step(battery_control, {}, dt)

        print(f"Step {step}: T_in={building_state['indoor_temperature']:.1f}°C, "
              f"PV={pv_output['power_output']:.0f}W, "
              f"Battery SOC={battery_output['soc']:.2f}")

        timestamp += dt