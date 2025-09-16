"""
Data structure used in bestopt runtime environment.

This module defines the key dataclasses used throughout the simulation:
- State: System state variables (temperatures, SOCs, etc.)
- Action: Control actions (charging/discharging power, HVAC thermal load, etc.)
- Disturbance: External disturbances (weather, prices, occupancy)
- Observation: What the controller observes
- Configuration: Configuration parameters used for initialization
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import numpy as np


# Operation mode class
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
    DISCONNECT_IDLE = "disconnected_idle"
    DISCONNECT_DRIVING = "disconnected_driving"


class GRIDMode(Enum):
    """Grid operation modes."""
    CONNECT = "connected"
    ISLAND = "island"


# Base component class
@dataclass
class ComponentState:
    """Base class for all component states."""
    component_id: str = ""
    component_type: str = ""
    domain: str = "" # 'electrical', 'thermal', 'water'
    timestamp: float = 0.0
    is_active: bool = True


# Electric state variables @TODO the current variables include action variables as well, which need to be removed
@dataclass
class BatteryState(ComponentState):
    """Battery storage component."""
    battery_soc: float = 0.5  # State of charge (0-1)
    battery_to_building: float = 0.0  # kW
    battery_to_grid: float = 0.0  # kW
    battery_to_ev: float = 0.0  # kW
    battery_temperature: float = 25.0  # °C
    battery_health: float = 1.0  # State of health (0-1)
    battery_power_loss: float = 0.0  # kW
    battery_mode: BatteryMode = BatteryMode.IDLE

    # @TODO Update later

    def __post_init__(self):
        self.domain = "electrical"
        self.component_type = "battery"


@dataclass
class EVState(ComponentState):
    """EV storage component."""
    ev_soc: float = 0.5  # State of charge
    ev_to_building: float = 0.0  # kW
    ev_to_battery: float = 0.0  # kW
    ev_to_grid: float = 0.0  # kW
    ev_driving_loss: float = 0.0  # kW
    ev_mode: EVMode = EVMode.CONNECTED_CHARGING

    # @TODO Update later

    def __post_init__(self):
        self.domain = "electrical"
        self.component_type = "ev"


@dataclass
class PVState(ComponentState):
    """Photovoltaic generation component."""
    pv_building: float = 0.0  # kW
    pv_battery: float = 0.0  # kW
    pv_grid: float = 0.0  # kW
    pv_ev: float = 0.0  # kW

    def __post_init__(self):
        self.domain = "electrical"
        self.component_type = "pv"


@dataclass
class BLDGEState(ComponentState):
    """Building load demand component."""
    hvac_power: float = 0.0  # kW
    lighting_power: float = 0.0  # kW
    plug_loads_power: float = 0.0  # kW

    def __post_init__(self):
        self.domain = "electrical"
        self.component_type = "bldg_e"  # E stands for electrical


# Thermal state variables
@dataclass
class BLDGTState(ComponentState):
    """Thermal zone component."""
    temperature: float = 22.0  # °C
    humidity: float = 50.0  # %
    internal_heat_gain: float = 0.0  # kW

    def __post_init__(self):
        self.domain = "thermal"
        self.component_type = "bldg_t"  # T stands for thermal


@dataclass
class TESState(ComponentState):
    """Thermal zone component."""
    temperature: float = 22.0  # °C
    tes_soc: float = 0.5  # State of charge (0-1)
    tes_mode: TESMode = TESMode.IDLE

    def __post_init__(self):
        self.domain = "thermal"
        self.component_type = "tes"  # T stands for thermal


@dataclass
class HVACState(ComponentState):
    """HVAC component."""
    thermal_load: float = 0.0  # kW

    def __post_init__(self):
        self.domain = "thermal"
        self.component_type = "hvac"


# Water state variables
@dataclass
class WaterHeaterState(ComponentState):
    """Water heating component."""
    tank_temperature: float = 60.0  # °C
    energy_content: float = 0.0  # kWh

    def __post_init__(self):
        self.domain = "water"
        self.component_type = "water_heater"


# Domain-Level State Aggregation
@dataclass
class ElectricalDomainState:
    """Aggregated electrical domain state for a building."""
    # Component collections
    batteries: Dict[str, BatteryState] = field(default_factory=dict)
    evs: Dict[str, EVState] = field(default_factory=dict)
    pv_systems: Dict[str, PVState] = field(default_factory=dict)
    bldg_e_loads: Dict[str, BLDGEState] = field(default_factory=dict)

    # Domain-level aggregations
    total_generation: float = 0.0  # kW
    total_demand: float = 0.0  # kW
    grid_import: float = 0.0  # kw
    grid_export: float = 0.0  # kw

    # @TODO

    def update_aggregations(self):
        """Update domain-level aggregated values."""
        # @TODO
        # self.total_generation = sum()
        # self.total_demand = sum()


@dataclass
class ThermalDomainState:
    """Aggregated thermal domain state for a building."""
    # Component collections
    hvac_systems: Dict[str, HVACState] = field(default_factory=dict)
    thermal_zones: Dict[str, BLDGTState] = field(default_factory=dict)
    thermal_storage: Dict[str, TESState] = field(default_factory=dict)

    # Domain-level aggregations
    total_heating_load: float = 0.0  # kW
    total_cooling_load: float = 0.0  # kW

    # @TODO

    def update_aggregations(self):
        """Update thermal domain aggregations."""
        # @TODO
        # self.total_heating_load = sum()
        # self.total_cooling_load = sum()


@dataclass
class WaterDomainState:
    """Aggregated water domain state for a building."""
    # Component collections
    water_heaters: Dict[str, WaterHeaterState] = field(default_factory=dict)

    # Domain-level aggregations
    total_water_heating_power: float = 0.0  # kW

    # @TODO

    def update_aggregations(self):
        """Update water domain aggregations."""
        # @TODO
        # self.total_water_heating_power = sum()


# Building-Level State Aggregation
@dataclass
class State:
    """Complete building state organized by physical domains."""
    building_id: str

    # Domain states (your original approach)
    electrical: ElectricalDomainState = field(default_factory=ElectricalDomainState)
    thermal: ThermalDomainState = field(default_factory=ThermalDomainState)
    water: WaterDomainState = field(default_factory=WaterDomainState)

    # Building-level aggregations
    # @TODO variables defined later
    total_electrical_load: float = 0.0  # kW
    total_thermal_load: float = 0.0  # kW
    net_energy_flow: float = 0.0  # kW

    def update_all_aggregations(self):
        """Update all domain and building-level aggregations."""
        self.electrical.update_aggregations()
        self.thermal.update_aggregations()
        self.water.update_aggregations()
        # @TODO

        # Building-level aggregations
        # @TODO
        # self.total_electrical_load =
        # self.total_thermal_load =
        # self.net_energy_flow =

    def get_component_by_id(self, component_id: str) -> Optional[ComponentState]:
        """Get any component by ID across all domains."""
        # @TODO the searching domain is not complete yet
        # Search electrical domain
        for components in [self.electrical.batteries, self.electrical.pv_systems, self.electrical.bldg_e_loads]:
            if component_id in components:
                return components[component_id]

        # Search thermal domain
        for components in [self.thermal.hvac_systems, self.thermal.thermal_zones, self.thermal.thermal_storage]:
            if component_id in components:
                return components[component_id]

        # Search water domain
        for components in [self.water.water_heaters]:
            if component_id in components:
                return components[component_id]

        return None

    def get_components_by_domain(self, domain: str) -> Dict[str, ComponentState]:
        """Get all components in a specific domain."""
        # @TODO not complete yet
        if domain == "electrical":
            result = {}
            result.update(self.electrical.batteries)
            result.update(self.electrical.pv_systems)
            result.update(self.electrical.bldg_e_loads)
            return result
        elif domain == "thermal":
            result = {}
            result.update(self.thermal.hvac_systems)
            result.update(self.thermal.thermal_zones)
            result.update(self.thermal.thermal_storage)
            return result
        elif domain == "water":
            result = {}
            result.update(self.water.water_heaters)
            return result
        return {}


# Action Variables
@dataclass
class ThermalAction:
    """Thermal control actions."""

    hvac_power: float = 0.0
    hvac_mode: HVACMode = HVACMode.OFF

    pump_flow_sp: Optional[float] = None    # m³/s water flow for circulation    
    supplyfan_flow_sp: Optional[float] = None
    
    # @TODO HVAC controls
    # @TODO Thermal storage control


@dataclass
class ElectricalAction:
    """Electrical control actions."""
    pass
    # @TODO Battery control
    # @TODO PV control
    # @TODO EV control


@dataclass
class WaterAction:
    """Water control actions."""
    pass
    # @TODO Battery control
    # @TODO PV control
    # @TODO EV control


@dataclass
class Action:
    """Complete control action combining thermal, electrical and water domains."""
    thermal: ThermalAction = field(default_factory=ThermalAction)
    electrical: ElectricalAction = field(default_factory=ElectricalAction)
    water: WaterAction = field(default_factory=WaterAction)

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            'thermal': self.thermal.__dict__,
            'electrical': self.electrical.__dict__,
            'water': self.water.__dict__
        }


# Disturbance Variables
@dataclass
class WeatherData:
    """Weather disturbances data."""
    outdoor_temperature: float = 20.0  # °C
    solar_radiation: float = 0.0  # W/m²
    # @TODO add more in the future


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
    # @TODO add behavior variables later

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


# Observation
@dataclass
class Observation:
    """What the controller observes."""
    # Current measurements
    # @TODO develop later

    # Time information
    time_of_day: float = 12.0  # hours (0-24)
    day_of_week: int = 1  # 1-7
    day_of_year: int = 1  # 1-365

    # Disturbances (could be arrays for multi-step, from historical data to future)
    # Weather
    outdoor_temp_forecast: List[float] = field(default_factory=list)
    solar_forecast: List[float] = field(default_factory=list)
    # Price
    price_forecast: List[float] = field(default_factory=list)
    # Occupancy
    occupancy_forecast: List[float] = field(default_factory=list)

    # Reference (what energy service we need to provide, for example, comfort)
    comfort_temp_min: float = 20.0  # °C
    comfort_temp_max: float = 26.0  # °C
    # TODO more reference signals needed

    # Placeholder for extensibility
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """Convert observation to numpy array for ML models."""
        # @TODO
        pass


@dataclass
class EnvConfig:
    # @TODO
    pass

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """
        helper function to validate config file
        """
        # @TODO
        pass


@dataclass
class BuildingConfig:
    # @TODO
    pass

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """
        helper function to validate config file
        """
        # @TODO
        pass


@dataclass
class HVACConfig:
    # @TODO
    pass

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """
        helper function to validate config file
        """
        # @TODO
        pass


@dataclass
class BatteryConfig:
    battery_capacity: float = 20.0  # kWh
    battery_c_rate: float = 0.25  # 0.5C means the battery takes 2 hours to fully charge; 0.2C means 5 hours.
    battery_charge_efficiency: float = 0.95
    battery_discharge_efficiency: float = 0.95
    battery_soc_min: float = 0.1
    battery_soc_max: float = 0.9

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """
        helper function to validate config file
        """
        # @TODO Add more later
        if not (0.0 <= self.battery_soc_min < self.battery_soc_max <= 1.0):
            raise ValueError("battery_soc_min/max must satisfy 0 ≤ min < max ≤ 1")


@dataclass
class EVConfig:
    ev_capacity: float = 60.0
    ev_c_rate: float = 0.25
    ev_charge_efficiency: float = 0.90
    ev_discharge_efficiency: float = 0.90
    ev_soc_min: float = 0.10
    ev_soc_max: float = 0.90

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """
        helper function to validate config file
        """
        # @TODO
        pass


@dataclass
class PVConfig:
    pv_capacity: float = 8.0  # kW

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """
        helper function to validate config file
        """
        # @TODO
        pass


@dataclass
class OtherConfig:
    # @TODO
    pass

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """
        helper function to validate config file
        """
        # @TODO
        pass


@dataclass
class Configuration:
    env: EnvConfig = field(default_factory=EnvConfig)
    building: BuildingConfig = field(default_factory=BuildingConfig)
    hvac: HVACConfig = field(default_factory=HVACConfig)  # feel free to add sub-classes later
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    ev: EVConfig = field(default_factory=EVConfig)
    pv: PVConfig = field(default_factory=PVConfig)


@dataclass
class PumpState(ComponentState):
    """Standalone; not aggregated into ThermalDomainState for now."""
    waterflow_m3s: float = 0.0
    power_W: float = 0.0
    energy_J_cum: float = 0.0  # accumulated electrical energy [kWh]

    def __post_init__(self):
        # identify this component; keep it consistent with your taxonomy
        self.domain = "thermal"
        self.component_type = "pump"

@dataclass
class FanState(ComponentState):
    """Standalone; not aggregated into ThermalDomainState for now."""
    airflow_m3s: float = 0.0
    power_W: float = 0.0
    energy_J_cum: float = 0.0  # accumulated electrical energy [kWh]

    def __post_init__(self):
        # identify this component; keep it consistent with your taxonomy
        self.domain = "thermal"
        self.component_type = "fan"



@dataclass
class CoilState(ComponentState):
    """Standalone; not aggregated into ThermalDomainState for now."""
    airflow_m3s: float = 0.0
    waterflow_m3s: float = 0.0
    air_inlet_temp_C: float = 0.0
    air_outlet_temp_C: float = 0.0
    water_inlet_temp_C: float = 0.0
    water_outlet_temp_C: float = 0.0

    def __post_init__(self):
        # identify this component; keep it consistent with your taxonomy
        self.domain = "thermal"
        self.component_type = "coil"