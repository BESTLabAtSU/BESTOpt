"""
Modular Physics-Informed Machine Learning (PIML) Framework
for Building Energy Systems

This framework supports scalable, generalizable, and physically consistent
modeling of building energy systems across diverse applications.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Base Classes and Interfaces
# =============================================================================

class ModuleType(Enum):
    """Enumeration of module types in the framework"""
    OCCUPANCY = "occupancy"
    DISTURBANCE = "disturbance"
    HVAC = "hvac"
    DER = "ders"
    EQUIPMENT = "equipment"
    THERMAL_DYNAMIC = "thermal_dynamic"
    POWER_DYNAMIC = "power_dynamic"
    CONTROL = "control"


@dataclass
class ModuleInput:
    """Standard input structure for modules"""
    timestamp: float
    data: Dict[str, Any]
    module_outputs: Dict[str, Any] = None


@dataclass
class ModuleOutput:
    """Standard output structure for modules"""
    timestamp: float
    values: Dict[str, Any]
    metadata: Dict[str, Any] = None


class PIMLModule(ABC):
    """Abstract base class for all PIML modules"""

    def __init__(self, module_id: str, module_type: ModuleType):
        self.module_id = module_id
        self.module_type = module_type
        self.is_trained = False
        self.dependencies = []
        self.parameters = {}

    @abstractmethod
    def forward(self, inputs: ModuleInput) -> ModuleOutput:
        """Forward pass through the module"""
        pass

    @abstractmethod
    def train(self, training_data: pd.DataFrame) -> None:
        """Train the module using physics-informed ML"""
        pass

    def add_dependency(self, module_id: str):
        """Add a dependency to another module"""
        self.dependencies.append(module_id)

    def validate_inputs(self, inputs: ModuleInput) -> bool:
        """Validate input data"""
        return inputs is not None and inputs.data is not None


# =============================================================================
# Occupancy Module
# =============================================================================

@dataclass
class OccupancyState:
    """Occupancy state representation"""
    mobility: Dict[str, float]  # location probabilities
    comfort: Dict[str, float]  # comfort preferences
    activity: Dict[str, float]  # activity patterns


class OccupancyModule(PIMLModule):
    """
    Occupancy module representing daily occupant behavior through:
    - Mobility: occupant location and presence
    - Comfort: thermal comfort range impacting setpoints
    - Activity: HVAC adjustments, lighting, appliance usage
    """

    def __init__(self, module_id: str = "occupancy_main"):
        super().__init__(module_id, ModuleType.OCCUPANCY)
        self.mobility_model = None
        self.comfort_model = None
        self.activity_model = None

    def forward(self, inputs: ModuleInput) -> ModuleOutput:
        """Generate occupancy predictions"""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid inputs for occupancy module")

        # Extract time-based features
        hour = inputs.data.get('hour', 0)
        day_of_week = inputs.data.get('day_of_week', 0)

        # Mobility prediction
        mobility = self._predict_mobility(hour, day_of_week)

        # Comfort-based setpoint prediction
        setpoint = self._predict_setpoint(mobility, inputs.data.get('external_temp', 20))

        # Activity prediction
        activity = self._predict_activity(mobility, hour)

        return ModuleOutput(
            timestamp=inputs.timestamp,
            values={
                'occupancy_count': mobility['occupancy_count'],
                'presence_probability': mobility['presence_prob'],
                'thermal_setpoint': setpoint,
                'appliance_usage': activity['appliance_usage'],
                'lighting_usage': activity['lighting_usage']
            },
            metadata={'module_type': 'occupancy'}
        )

    def train(self, training_data: pd.DataFrame) -> None:
        """Train occupancy models using historical data"""
        logger.info(f"Training occupancy module {self.module_id}")

        # Train mobility model (simplified)
        self.mobility_model = self._train_mobility_model(training_data)

        # Train comfort model
        self.comfort_model = self._train_comfort_model(training_data)

        # Train activity model
        self.activity_model = self._train_activity_model(training_data)

        self.is_trained = True
        logger.info(f"Occupancy module {self.module_id} training completed")

    def _predict_mobility(self, hour: int, day_of_week: int) -> Dict[str, float]:
        """Predict mobility patterns"""
        # Simplified mobility prediction
        if 9 <= hour <= 17 and day_of_week < 5:  # Working hours
            return {'occupancy_count': 1.0, 'presence_prob': 0.8}
        else:
            return {'occupancy_count': 0.5, 'presence_prob': 0.3}

    def _predict_setpoint(self, mobility: Dict, external_temp: float) -> float:
        """Predict comfort-based setpoint"""
        base_setpoint = 22.0  # Base comfort temperature
        if mobility['presence_prob'] > 0.5:
            # Adjust based on external temperature
            adjustment = (external_temp - 20) * 0.1
            return base_setpoint + adjustment
        return base_setpoint

    def _predict_activity(self, mobility: Dict, hour: int) -> Dict[str, float]:
        """Predict activity patterns"""
        base_usage = mobility['presence_prob']
        return {
            'appliance_usage': base_usage * (1.0 if 18 <= hour <= 22 else 0.5),
            'lighting_usage': base_usage * (1.0 if hour < 8 or hour > 18 else 0.3)
        }

    def _train_mobility_model(self, data: pd.DataFrame):
        """Train mobility prediction model"""
        # Placeholder for actual ML model training
        return {"trained": True, "model_type": "mobility"}

    def _train_comfort_model(self, data: pd.DataFrame):
        """Train comfort prediction model"""
        return {"trained": True, "model_type": "comfort"}

    def _train_activity_model(self, data: pd.DataFrame):
        """Train activity prediction model"""
        return {"trained": True, "model_type": "activity"}


# =============================================================================
# Disturbance Module
# =============================================================================

class DisturbanceModule(PIMLModule):
    """
    Disturbance module for weather-related factors and external signals:
    - Ambient air temperature
    - Solar radiation
    - Real-time utility prices
    """

    def __init__(self, module_id: str = "disturbance_main"):
        super().__init__(module_id, ModuleType.DISTURBANCE)
        self.weather_model = None
        self.price_model = None

    def forward(self, inputs: ModuleInput) -> ModuleOutput:
        """Generate disturbance predictions"""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid inputs for disturbance module")

        timestamp = inputs.timestamp

        # Weather predictions
        weather = self._predict_weather(timestamp)

        # Price predictions
        price = self._predict_price(timestamp)

        return ModuleOutput(
            timestamp=timestamp,
            values={
                'ambient_temperature': weather['temperature'],
                'solar_radiation': weather['solar_radiation'],
                'wind_speed': weather['wind_speed'],
                'humidity': weather['humidity'],
                'electricity_price': price['electricity'],
                'gas_price': price['gas']
            },
            metadata={'module_type': 'disturbance'}
        )

    def train(self, training_data: pd.DataFrame) -> None:
        """Train disturbance prediction models"""
        logger.info(f"Training disturbance module {self.module_id}")

        # Train weather model
        self.weather_model = self._train_weather_model(training_data)

        # Train price model
        self.price_model = self._train_price_model(training_data)

        self.is_trained = True
        logger.info(f"Disturbance module {self.module_id} training completed")

    def _predict_weather(self, timestamp: float) -> Dict[str, float]:
        """Predict weather conditions"""
        # Simplified weather prediction
        hour = int((timestamp % 86400) / 3600)  # Hour of day

        return {
            'temperature': 20 + 10 * np.sin(2 * np.pi * hour / 24),
            'solar_radiation': max(0, 800 * np.sin(np.pi * hour / 12)),
            'wind_speed': 5 + 3 * np.random.random(),
            'humidity': 0.5 + 0.2 * np.random.random()
        }

    def _predict_price(self, timestamp: float) -> Dict[str, float]:
        """Predict utility prices"""
        hour = int((timestamp % 86400) / 3600)

        # Time-of-use pricing
        if 6 <= hour <= 10 or 18 <= hour <= 22:
            electricity_price = 0.25  # Peak hours
        else:
            electricity_price = 0.15  # Off-peak hours

        return {
            'electricity': electricity_price,
            'gas': 0.05  # Simplified gas price
        }

    def _train_weather_model(self, data: pd.DataFrame):
        """Train weather prediction model"""
        return {"trained": True, "model_type": "weather"}

    def _train_price_model(self, data: pd.DataFrame):
        """Train price prediction model"""
        return {"trained": True, "model_type": "price"}


# =============================================================================
# HVAC Module
# =============================================================================

class HVACModule(PIMLModule):
    """
    HVAC module for heating, ventilation, and air conditioning systems
    Outputs: thermal load, electric load
    """

    def __init__(self, module_id: str = "hvac_main"):
        super().__init__(module_id, ModuleType.HVAC)
        self.performance_model = None
        self.efficiency_curves = {}

    def forward(self, inputs: ModuleInput) -> ModuleOutput:
        """Generate HVAC system outputs"""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid inputs for HVAC module")

        # Extract inputs
        setpoint = inputs.data.get('setpoint', 22)
        ambient_temp = inputs.data.get('ambient_temperature', 20)
        indoor_temp = inputs.data.get('indoor_temperature', 22)

        # Calculate thermal load
        thermal_load = self._calculate_thermal_load(setpoint, indoor_temp, ambient_temp)

        # Calculate electric load
        electric_load = self._calculate_electric_load(thermal_load, ambient_temp)

        return ModuleOutput(
            timestamp=inputs.timestamp,
            values={
                'thermal_load': thermal_load,
                'electric_load': electric_load,
                'cop': self._calculate_cop(ambient_temp),
                'mass_flow_rate': self._calculate_mass_flow_rate(thermal_load)
            },
            metadata={'module_type': 'hvac'}
        )

    def train(self, training_data: pd.DataFrame) -> None:
        """Train HVAC performance models"""
        logger.info(f"Training HVAC module {self.module_id}")

        # Train performance model
        self.performance_model = self._train_performance_model(training_data)

        # Train efficiency curves
        self.efficiency_curves = self._train_efficiency_curves(training_data)

        self.is_trained = True
        logger.info(f"HVAC module {self.module_id} training completed")

    def _calculate_thermal_load(self, setpoint: float, indoor_temp: float,
                                ambient_temp: float) -> float:
        """Calculate thermal load based on temperature difference"""
        temp_diff = setpoint - indoor_temp
        load_factor = abs(temp_diff) * 1000  # Simplified load calculation
        return load_factor

    def _calculate_electric_load(self, thermal_load: float, ambient_temp: float) -> float:
        """Calculate electric load based on thermal load and efficiency"""
        cop = self._calculate_cop(ambient_temp)
        return thermal_load / cop if cop > 0 else 0

    def _calculate_cop(self, ambient_temp: float) -> float:
        """Calculate coefficient of performance"""
        # Simplified COP calculation
        return 3.5 - 0.05 * abs(ambient_temp - 20)

    def _calculate_mass_flow_rate(self, thermal_load: float) -> float:
        """Calculate mass flow rate"""
        return thermal_load / 1000  # Simplified calculation

    def _train_performance_model(self, data: pd.DataFrame):
        """Train HVAC performance model"""
        return {"trained": True, "model_type": "performance"}

    def _train_efficiency_curves(self, data: pd.DataFrame):
        """Train efficiency curves"""
        return {"trained": True, "model_type": "efficiency"}


# =============================================================================
# DER Module
# =============================================================================

class DERType(Enum):
    """Types of Distributed Energy Resources"""
    PV = "photovoltaic"
    WIND = "wind"
    DIESEL = "diesel_generator"
    BATTERY = "battery_storage"
    THERMAL_STORAGE = "thermal_storage"
    EV = "electric_vehicle"


class DERModule(PIMLModule):
    """
    Distributed Energy Resources module for generation and storage systems
    Output: electric load (positive for consumption, negative for generation)
    """

    def __init__(self, module_id: str, der_type: DERType):
        super().__init__(module_id, ModuleType.DER)
        self.der_type = der_type
        self.capacity = 0
        self.efficiency = 0.9
        self.state_of_charge = 0.5  # For storage systems

    def forward(self, inputs: ModuleInput) -> ModuleOutput:
        """Generate DER outputs"""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid inputs for DER module")

        if self.der_type == DERType.PV:
            output = self._calculate_pv_output(inputs)
        elif self.der_type == DERType.BATTERY:
            output = self._calculate_battery_output(inputs)
        elif self.der_type == DERType.WIND:
            output = self._calculate_wind_output(inputs)
        else:
            output = {'electric_load': 0, 'generation_potential': 0}

        return ModuleOutput(
            timestamp=inputs.timestamp,
            values=output,
            metadata={'module_type': 'ders', 'der_type': self.der_type.value}
        )

    def train(self, training_data: pd.DataFrame) -> None:
        """Train DER models"""
        logger.info(f"Training DER module {self.module_id}")

        # Train based on DER type
        if self.der_type == DERType.PV:
            self._train_pv_model(training_data)
        elif self.der_type == DERType.BATTERY:
            self._train_battery_model(training_data)
        elif self.der_type == DERType.WIND:
            self._train_wind_model(training_data)

        self.is_trained = True
        logger.info(f"DER module {self.module_id} training completed")

    def _calculate_pv_output(self, inputs: ModuleInput) -> Dict[str, float]:
        """Calculate PV generation output"""
        solar_radiation = inputs.data.get('solar_radiation', 0)
        generation = self.capacity * solar_radiation / 1000 * self.efficiency

        return {
            'electric_load': -generation,  # Negative for generation
            'generation_potential': generation
        }

    def _calculate_battery_output(self, inputs: ModuleInput) -> Dict[str, float]:
        """Calculate battery storage output"""
        control_signal = inputs.data.get('battery_control', 0)  # +charge, -discharge

        # Update state of charge
        self.state_of_charge = np.clip(
            self.state_of_charge + control_signal * 0.1, 0, 1
        )

        return {
            'electric_load': control_signal * self.capacity,
            'state_of_charge': self.state_of_charge
        }

    def _calculate_wind_output(self, inputs: ModuleInput) -> Dict[str, float]:
        """Calculate wind generation output"""
        wind_speed = inputs.data.get('wind_speed', 0)

        # Simplified wind power curve
        if wind_speed < 3:
            generation = 0
        elif wind_speed > 25:
            generation = 0
        else:
            generation = self.capacity * min(1, (wind_speed - 3) / 10)

        return {
            'electric_load': -generation,  # Negative for generation
            'generation_potential': generation
        }

    def _train_pv_model(self, data: pd.DataFrame):
        """Train PV model"""
        pass

    def _train_battery_model(self, data: pd.DataFrame):
        """Train battery model"""
        pass

    def _train_wind_model(self, data: pd.DataFrame):
        """Train wind model"""
        pass


# =============================================================================
# Thermal Dynamic Module
# =============================================================================

class ThermalDynamicModule(PIMLModule):
    """
    Thermal dynamic module for building thermal dynamics
    Uses building dynamic model for thermal loop
    """

    def __init__(self, module_id: str = "thermal_main"):
        super().__init__(module_id, ModuleType.THERMAL_DYNAMIC)
        self.thermal_mass = 1000  # Building thermal mass
        self.ua_value = 500  # Overall heat transfer coefficient
        self.indoor_temperature = 22  # Initial indoor temperature

    def forward(self, inputs: ModuleInput) -> ModuleOutput:
        """Calculate thermal dynamics"""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid inputs for thermal dynamic module")

        # Extract inputs
        ambient_temp = inputs.data.get('ambient_temperature', 20)
        thermal_load = inputs.data.get('thermal_load', 0)
        solar_gain = inputs.data.get('solar_radiation', 0) * 0.1  # Simplified

        # Calculate heat transfer
        heat_transfer = self._calculate_heat_transfer(ambient_temp, thermal_load, solar_gain)

        # Update indoor temperature
        self.indoor_temperature = self._update_indoor_temperature(heat_transfer)

        return ModuleOutput(
            timestamp=inputs.timestamp,
            values={
                'indoor_temperature': self.indoor_temperature,
                'heat_transfer_rate': heat_transfer,
                'thermal_mass_temperature': self.indoor_temperature
            },
            metadata={'module_type': 'thermal_dynamic'}
        )

    def train(self, training_data: pd.DataFrame) -> None:
        """Train thermal dynamic models"""
        logger.info(f"Training thermal dynamic module {self.module_id}")

        # Train thermal parameters
        self._train_thermal_parameters(training_data)

        self.is_trained = True
        logger.info(f"Thermal dynamic module {self.module_id} training completed")

    def _calculate_heat_transfer(self, ambient_temp: float, thermal_load: float,
                                 solar_gain: float) -> float:
        """Calculate heat transfer rate"""
        # Heat transfer through envelope
        envelope_transfer = self.ua_value * (ambient_temp - self.indoor_temperature)

        # Total heat transfer
        total_transfer = envelope_transfer + thermal_load + solar_gain

        return total_transfer

    def _update_indoor_temperature(self, heat_transfer: float) -> float:
        """Update indoor temperature based on heat transfer"""
        dt = 1  # Time step in hours
        temp_change = heat_transfer * dt / self.thermal_mass
        return self.indoor_temperature + temp_change

    def _train_thermal_parameters(self, data: pd.DataFrame):
        """Train thermal parameters"""
        pass


# =============================================================================
# Power Dynamic Module
# =============================================================================

class PowerDynamicModule(PIMLModule):
    """
    Power dynamic module for power balance and electric loop
    """

    def __init__(self, module_id: str = "power_main"):
        super().__init__(module_id, ModuleType.POWER_DYNAMIC)
        self.grid_connection = True

    def forward(self, inputs: ModuleInput) -> ModuleOutput:
        """Calculate power balance"""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid inputs for power dynamic module")

        # Extract electric loads from all sources
        hvac_load = inputs.data.get('hvac_electric_load', 0)
        der_loads = inputs.data.get('der_electric_loads', [])
        appliance_load = inputs.data.get('appliance_load', 0)

        # Calculate total load
        total_load = hvac_load + sum(der_loads) + appliance_load

        # Calculate grid interaction
        grid_power = self._calculate_grid_power(total_load)

        return ModuleOutput(
            timestamp=inputs.timestamp,
            values={
                'total_electric_load': total_load,
                'grid_power': grid_power,
                'power_balance': 0,  # Should be zero if balanced
                'power_factor': 0.95
            },
            metadata={'module_type': 'power_dynamic'}
        )

    def train(self, training_data: pd.DataFrame) -> None:
        """Train power dynamic models"""
        logger.info(f"Training power dynamic module {self.module_id}")
        self.is_trained = True
        logger.info(f"Power dynamic module {self.module_id} training completed")

    def _calculate_grid_power(self, total_load: float) -> float:
        """Calculate grid power requirement"""
        if self.grid_connection:
            return total_load  # Grid supplies what's needed
        else:
            return 0  # Islanded mode


# =============================================================================
# Control Module
# =============================================================================

class ControlModule(PIMLModule):
    """
    Control module providing interactive control evaluation and
    optimal control policy generation
    """

    def __init__(self, module_id: str = "control_main"):
        super().__init__(module_id, ModuleType.CONTROL)
        self.control_strategy = "rule_based"
        self.optimization_objective = "energy_cost"

    def forward(self, inputs: ModuleInput) -> ModuleOutput:
        """Generate control signals"""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid inputs for control module")

        # Extract system state
        indoor_temp = inputs.data.get('indoor_temperature', 22)
        setpoint = inputs.data.get('setpoint', 22)
        electricity_price = inputs.data.get('electricity_price', 0.15)

        # Generate control signals
        control_signals = self._generate_control_signals(
            indoor_temp, setpoint, electricity_price
        )

        return ModuleOutput(
            timestamp=inputs.timestamp,
            values=control_signals,
            metadata={'module_type': 'control'}
        )

    def train(self, training_data: pd.DataFrame) -> None:
        """Train control policies"""
        logger.info(f"Training control module {self.module_id}")

        # Train control policy
        self._train_control_policy(training_data)

        self.is_trained = True
        logger.info(f"Control module {self.module_id} training completed")

    def _generate_control_signals(self, indoor_temp: float, setpoint: float,
                                  price: float) -> Dict[str, float]:
        """Generate control signals based on current state"""
        # Simple rule-based control
        temp_error = setpoint - indoor_temp

        # HVAC control
        if abs(temp_error) > 0.5:
            hvac_signal = 1.0 if temp_error > 0 else -1.0
        else:
            hvac_signal = 0.0

        # Battery control (charge during low price)
        battery_signal = -1.0 if price < 0.2 else 1.0

        return {
            'hvac_control': hvac_signal,
            'battery_control': battery_signal,
            'lighting_control': 1.0
        }

    def _train_control_policy(self, data: pd.DataFrame):
        """Train optimal control policy"""
        pass


# =============================================================================
# Framework Integration
# =============================================================================

class PIMLFramework:
    """
    Main framework class that integrates all modules
    """

    def __init__(self):
        self.modules = {}
        self.execution_order = []
        self.signal_loop = {}
        self.thermal_loop = {}
        self.electric_loop = {}

    def add_module(self, module: PIMLModule):
        """Add a module to the framework"""
        self.modules[module.module_id] = module
        logger.info(f"Added module: {module.module_id} ({module.module_type.value})")

    def remove_module(self, module_id: str):
        """Remove a module from the framework"""
        if module_id in self.modules:
            del self.modules[module_id]
            logger.info(f"Removed module: {module_id}")

    def set_execution_order(self, order: List[str]):
        """Set the execution order of modules"""
        self.execution_order = order

    def run_simulation(self, time_steps: int, dt: float = 1.0) -> Dict[str, List]:
        """Run simulation for specified time steps"""
        results = {module_id: [] for module_id in self.modules.keys()}

        for step in range(time_steps):
            timestamp = step * dt
            module_outputs = {}

            # Execute modules in order
            for module_id in self.execution_order:
                if module_id in self.modules:
                    module = self.modules[module_id]

                    # Prepare inputs
                    inputs = self._prepare_module_inputs(
                        module_id, timestamp, module_outputs
                    )

                    # Execute module
                    output = module.forward(inputs)
                    module_outputs[module_id] = output
                    results[module_id].append(output)

            # Log progress
            if step % 24 == 0:  # Log every 24 steps (assuming hourly steps)
                logger.info(f"Simulation step {step}/{time_steps}")

        return results

    def _prepare_module_inputs(self, module_id: str, timestamp: float,
                               module_outputs: Dict) -> ModuleInput:
        """Prepare inputs for a specific module"""
        module = self.modules[module_id]
        input_data = {'timestamp': timestamp}

        # Add dependency outputs
        for dep_id in module.dependencies:
            if dep_id in module_outputs:
                dep_output = module_outputs[dep_id]
                input_data.update(dep_output.values)

        return ModuleInput(
            timestamp=timestamp,
            data=input_data,
            module_outputs=module_outputs
        )


# =============================================================================
# Example Usage
# =============================================================================

def create_example_framework():
    """Create an example PIML framework setup"""

    # Initialize framework
    framework = PIMLFramework()

    # Create modules
    occupancy = OccupancyModule()
    disturbance = DisturbanceModule()
    hvac = HVACModule()
    pv = DERModule("pv_system", DERType.PV)
    battery = DERModule("battery_system", DERType.BATTERY)
    thermal = ThermalDynamicModule()
    power = PowerDynamicModule()
    control = ControlModule()

    # Set capacities
    pv.capacity = 5000  # 5kW PV system
    battery.capacity = 10000  # 10kWh battery

    # Add modules to framework
    framework.add_module(occupancy)
    framework.add_module(disturbance)
    framework.add_module(hvac)
    framework.add_module(pv)
    framework.add_module(battery)
    framework.add_module(thermal)
    framework.add_module(power)
    framework.add_module(control)

    # Set dependencies
    hvac.add_dependency("occupancy_main")
    hvac.add_dependency("disturbance_main")
    pv.add_dependency("disturbance_main")
    battery.add_dependency("control_main")
    thermal.add_dependency("hvac_main")
    thermal.add_dependency("disturbance_main")
    power.add_dependency("hvac_main")
    power.add_dependency("pv_system")
    power.add_dependency("battery_system")
    control.add_dependency("thermal_main")
    control.add_dependency("disturbance_main")

    # Set execution order
    framework.set_execution_order([
        "disturbance_main",
        "occupancy_main",
        "control_main",
        "hvac_main",
        "pv_system",
        "battery_system",
        "thermal_main",
        "power_main"
    ])

    return framework


if __name__ == "__main__":
    # Create and run example framework
    framework = create_example_framework()

    # Run simulation for 48 hours
    results = framework.run_simulation(time_steps=48, dt=1.0)

    print("Simulation completed successfully!")
    print(f"Generated results for {len(results)} modules")