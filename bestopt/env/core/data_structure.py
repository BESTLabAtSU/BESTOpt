"""
Data structure used in bestopt runtime environment.

This module defines the key dataclasses used throughout the simulation:
- State: System state variables (temperatures, SOCs, etc.)
- Action: Control actions (charging/discharging power, HVAC thermal load, etc.)
- Disturbance: External disturbances (weather, prices, occupancy)
- Observation: What the controller observes
- SimulationConfig: Configuration parameters
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import numpy as np


class HVACMode(Enum):
    """HVAC operation modes."""
    OFF = "off"
    HEATING = "heating"
    COOLING = "cooling"
    DUAL = "dual"
    VENTILATION = "ventilation"
    AUTO = "auto"


class BatteryMode(Enum):
    """Battery operation modes."""
    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"


class TESMode(Enum):
    """Thermal energy storage operation modes."""
    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"


class EVMode(Enum):
    """Electric Vehicle operation modes."""
    CONNECTED_CHARGING = "connected_charging"
    CONNECTED_IDLE = "connected_idle"
    CONNECTED_V2G = "connected_v2g"
    CONNECTED_V2B = "connected_v2b"
    DISCONNECTED = "disconnected"


@dataclass
class ThermalState:
    """State variables of thermal loop."""
    # Building thermal zones
    zone_temperatures: Dict[str, float] = field(default_factory=dict)  # °C
    # @TODO zone_humidity: Dict[str, float] = field(default_factory=dict)  # %

    # HVAC system
    # @TODO Depends on HVAC settings
    hvac_supply_temp: float = 20.0  # °C
    hvac_return_temp: float = 22.0  # °C
    hvac_airflow_rate: float = 0.0  # m³/s

    # Thermal storage
    # @TODO
    thermal_storage_temp: Optional[float] = None  # °C
    thermal_storage_soc: Optional[float] = None  # %


@dataclass
class ElectricalState:
    """State variables of electrical loop."""
    # Battery
    battery_soc: float = 0.5  # State of charge
    battery_to_building: float = 0.0  # kW
    battery_to_grid: float = 0.0  # kW
    battery_to_ev: float = 0.0  # kW
    battery_temperature: float = 25.0  # °C

    # Electric Vehicle
    ev_soc: float = 0.5  # State of charge
    ev_connected: bool = False
    ev_to_building: float = 0.0  # kW
    ev_to_battery: float = 0.0  # kW
    ev_to_grid: float = 0.0  # kW

    # Grid
    grid_connected: bool = True
    grid_building: float = 0.0  # kW
    grid_battery: float = 0.0  # kW
    grid_ev: float = 0.0  # kW

    # Renewable generation
    pv_building: float = 0.0  # kW
    pv_battery: float = 0.0  # kW
    pv_grid: float = 0.0  # kW
    pv_ev: float = 0.0  # kW
    # @TODO Wind turbine

    # Building load breakdown
    hvac_power: float = 0.0  # kW
    lighting_power: float = 0.0  # kW
    plug_loads_power: float = 0.0  # kW


@dataclass
# @TODO
class WaterState:
    pass


@dataclass
class State:
    """Complete system state combining thermal, electrical and water loops."""
    thermal: ThermalState = field(default_factory=ThermalState)
    electrical: ElectricalState = field(default_factory=ElectricalState)
    water: WaterState = field(default_factory=WaterState)

    # Simulation metadata
    timestamp: float = 0.0  # simulation time in hours
    step_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for logging/saving."""
        return {
            'thermal': self.thermal.__dict__,
            'electrical': self.electrical.__dict__,
            'water':self.water.__dict__,
            'timestamp': self.timestamp,
            'step_count': self.step_count
        }


@dataclass
class ThermalAction:
    """Thermal control actions."""
    # HVAC controls
    hvac_mode: HVACMode = HVACMode.AUTO
    heating_setpoint: float = 20.0  # °C
    cooling_setpoint: float = 24.0  # °C
    airflow_setpoint: float = 0.0  # m³/s

    # Thermal storage control
    thermal_storage_power: float = 0.0  # kW (+ = charging, - = discharging)


@dataclass
class ElectricalAction:
    """Electrical control actions."""
    # Battery control
    battery_power_setpoint: float = 0.0  # kW
    battery_mode: BatteryMode = BatteryMode.IDLE

    # EV control
    ev_power_setpoint: float = 0.0  # kW
    ev_mode: EVMode = EVMode.DISCONNECTED

    # Grid interaction
    grid_power_limit: Optional[float] = None  # kW
    export_limit: Optional[float] = None  # kW


@dataclass
class Action:
    """Complete control action combining thermal and electrical domains."""
    thermal: ThermalAction = field(default_factory=ThermalAction)
    electrical: ElectricalAction = field(default_factory=ElectricalAction)

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            'thermal': self.thermal.__dict__,
            'electrical': self.electrical.__dict__
        }


@dataclass
class WeatherData:
    """Weather disturbance data."""
    outdoor_temperature: float = 20.0  # °C
    humidity: float = 50.0  # %
    solar_irradiance: float = 0.0  # W/m²
    wind_speed: float = 0.0  # m/s
    wind_direction: float = 0.0  # degrees
    precipitation: float = 0.0  # mm/h
    cloud_cover: float = 0.0  # %


@dataclass
class PriceSignals:
    """Electricity and energy price signals."""
    electricity_price: float = 0.10  # $/kWh
    demand_charge: float = 15.0  # $/kW
    carbon_intensity: float = 500.0  # gCO2/kWh
    gas_price: Optional[float] = None  # $/therm


@dataclass
class OccupancyData:
    """Occupancy and comfort requirements."""
    occupancy_count: int = 0
    occupancy_fraction: float = 0.0  # 0-1
    comfort_temp_min: float = 20.0  # °C
    comfort_temp_max: float = 26.0  # °C
    activity_level: float = 1.2  # metabolic rate multiplier
    clothing_level: float = 1.0  # clo

    # Schedule flags
    is_occupied: bool = False
    is_weekend: bool = False
    is_holiday: bool = False


@dataclass
class Disturbance:
    """External disturbances affecting the system."""
    weather: WeatherData = field(default_factory=WeatherData)
    prices: PriceSignals = field(default_factory=PriceSignals)
    occupancy: OccupancyData = field(default_factory=OccupancyData)

    # Additional disturbances
    base_electrical_load: float = 2.0  # kW (non-HVAC electrical load)
    internal_heat_gains: float = 1.0  # kW (people, equipment, lighting)


@dataclass
class Observation:
    """What the controller observes (subset of full state + disturbances)."""
    # Current measurements
    zone_temp: float = 22.0  # °C
    zone_humidity: float = 50.0  # %
    battery_soc: float = 50.0  # %
    ev_soc: float = 50.0  # %
    pv_power: float = 0.0  # kW
    grid_power: float = 0.0  # kW

    # Weather forecast (could be arrays for multi-step)
    outdoor_temp_forecast: List[float] = field(default_factory=list)
    solar_forecast: List[float] = field(default_factory=list)

    # Price forecast
    price_forecast: List[float] = field(default_factory=list)

    # Comfort requirements
    comfort_temp_min: float = 20.0  # °C
    comfort_temp_max: float = 26.0  # °C
    occupancy_forecast: List[float] = field(default_factory=list)

    # Time information
    time_of_day: float = 12.0  # hours (0-24)
    day_of_week: int = 1  # 1-7
    day_of_year: int = 1  # 1-365

    def to_array(self) -> np.ndarray:
        """Convert observation to numpy array for ML models."""
        # Flatten all scalar values into array
        scalar_values = [
            self.zone_temp, self.zone_humidity, self.battery_soc,
            self.ev_soc, self.pv_power, self.grid_power,
            self.comfort_temp_min, self.comfort_temp_max,
            self.time_of_day, self.day_of_week, self.day_of_year
        ]

        # Add forecast arrays (pad/truncate to fixed length if needed)
        forecast_arrays = [
            self.outdoor_temp_forecast[:24] if len(self.outdoor_temp_forecast) >= 24
            else self.outdoor_temp_forecast + [0.0] * (24 - len(self.outdoor_temp_forecast)),
            self.solar_forecast[:24] if len(self.solar_forecast) >= 24
            else self.solar_forecast + [0.0] * (24 - len(self.solar_forecast)),
            self.price_forecast[:24] if len(self.price_forecast) >= 24
            else self.price_forecast + [0.0] * (24 - len(self.price_forecast)),
        ]

        all_values = scalar_values
        for arr in forecast_arrays:
            all_values.extend(arr)

        return np.array(all_values, dtype=np.float32)


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    # Time parameters
    timestep: float = 0.25  # hours (15 minutes)
    simulation_days: int = 7
    start_time: float = 0.0  # hours from start of year

    # Building parameters
    building_area: float = 200.0  # m²
    building_volume: float = 600.0  # m³
    building_thermal_mass: float = 50000.0  # J/K
    building_ua_value: float = 300.0  # W/K (overall heat transfer coefficient)

    # HVAC parameters
    hvac_capacity: float = 10.0  # kW
    hvac_cop_cooling: float = 3.0
    hvac_cop_heating: float = 2.5

    # Battery parameters
    battery_capacity: float = 20.0  # kWh
    battery_max_power: float = 5.0  # kW
    battery_efficiency: float = 0.95
    battery_soc_min: float = 10.0  # %
    battery_soc_max: float = 90.0  # %

    # EV parameters
    ev_capacity: float = 60.0  # kWh
    ev_max_power: float = 7.0  # kW
    ev_efficiency: float = 0.90

    # PV parameters
    pv_capacity: float = 8.0  # kW
    pv_tilt: float = 30.0  # degrees
    pv_azimuth: float = 180.0  # degrees (south-facing)

    # Comfort parameters
    comfort_temp_deadband: float = 2.0  # °C
    comfort_temp_violation_penalty: float = 100.0  # $/°C-hour

    # Economic parameters
    demand_charge_window: int = 96  # timesteps (24 hours for 15-min timesteps)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        checks = [
            self.timestep > 0,
            self.simulation_days > 0,
            self.building_area > 0,
            self.battery_capacity > 0,
            self.battery_soc_min < self.battery_soc_max,
            0 <= self.pv_tilt <= 90,
            0 <= self.pv_azimuth <= 360,
        ]
        return all(checks)


# Utility functions for data structure manipulation
def merge_states(state1: State, state2: State, weight1: float = 0.5) -> State:
    """Merge two states with given weights (useful for ensemble methods)."""
    # This is a simplified implementation - in practice you'd need more sophisticated merging
    merged = State()
    merged.timestamp = state1.timestamp * weight1 + state2.timestamp * (1 - weight1)
    merged.step_count = max(state1.step_count, state2.step_count)
    return merged


def state_difference(state1: State, state2: State) -> Dict[str, float]:
    """Calculate difference between two states for analysis."""
    diff = {}
    diff['temp_diff'] = abs(state1.thermal.zone_temperatures.get('main', 22.0) -
                            state2.thermal.zone_temperatures.get('main', 22.0))
    diff['battery_soc_diff'] = abs(state1.electrical.battery_soc - state2.electrical.battery_soc)
    diff['power_diff'] = abs(state1.electrical.grid_power - state2.electrical.grid_power)
    return diff
