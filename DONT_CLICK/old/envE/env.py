"""
BEST_OPT - Building-to-Grid Emulator
Copyright (c) 2025 Zixin Jiang, BEST Lab, Syracuse University

This module implements a forward simulation environment for building energy systems
with PV, battery, and EV components.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import copy
from collections import defaultdict
from DONT_CLICK.old import Config
from DONT_CLICK.old import BuildingConfig
from DONT_CLICK.old import GlobalConfig

@dataclass
class Action:
    """
    Energy flows between each module.
    All power values in kW (non-negative values only).
    """
    building_id: int
    timestep: int
    # PV flows
    pv_to_building: float = 0.0
    pv_to_battery: float = 0.0
    pv_to_ev: float = 0.0
    pv_to_grid: float = 0.0
    pv_dumped: float = 0.0
    # Battery flows
    battery_to_building: float = 0.0
    battery_to_ev: float = 0.0
    battery_to_grid: float = 0.0
    grid_to_battery: float = 0.0
    ev_to_battery: float = 0.0
    # EV flows
    ev_to_building: float = 0.0
    ev_to_grid: float = 0.0
    grid_to_ev: float = 0.0
    # Grid flows
    grid_to_building: float = 0.0

@dataclass
class State:
    """
    Current state of a building's energy system.
    """
    building_id: int
    timestep: int
    load_demand: float  # Current load demand (kW)
    pv_generation: float  # Current PV generation (kW)
    battery_soc: float  # Battery state of charge [0-1]
    battery_energy: float  # Battery energy content (kWh)
    ev_present: bool  # Whether EV is present
    ev_soc: float  # EV state of charge [0-1]
    ev_energy: float  # EV energy content (kWh)

    # System constraint
    battery_max_charge_power: float = 0.0
    battery_max_discharge_power: float = 0.0
    ev_max_charge_power: float = 0.0
    ev_max_discharge_power: float = 0.0

@dataclass
class Observation:
    """
    Observation of the system state.
    """
    current_timestep: int
    building_states: List[State]
    electricity_price: float  # Current TOU price

    def get_building_state(self, building_id: int) -> Optional[State]:
        """Get state for a specific building."""
        for state in self.building_states:
            if state.building_id == building_id:
                return state
        return None

@dataclass
class Record:
    """
    Record of energy flows.
    """
    action: Action
    actual_flows: Action
    clipping_occurred: bool
    clipping_details: Dict[str, Tuple[float, float]]  # field_name: (requested, actual)

class BuildingToGridEmulator:
    """
    Emulator for building-to-grid simulation.
    """

    def __init__(self, config: 'Config', buildings: List['BuildingConfig']):
        """
        Initialize the emulator.

        Args:
            config: Global configuration
            buildings: List of building configurations
        """
        self.config = config
        self.buildings = buildings
        self.current_timestep = 0
        self.max_timesteps = len(config.global_config.T_amb)

        # Initialize building states
        self.building_states = self._initialize_building_states()

        # Energy flow history
        self.energy_flow_history: List[Record] = []

        # Performance metrics
        self.metrics = defaultdict(list)

    def _initialize_building_states(self) -> List[State]:
        """Initialize the state for each building."""
        states = []

        for i, building in enumerate(self.buildings):
            # Calculate initial battery state
            battery_soc = building.battery.initial_soc if building.battery else 0.0
            battery_energy = battery_soc * building.battery.capacity_kwh if building.battery else 0.0

            # Calculate initial EV state
            E = False
            ev_soc = 0.0
            ev_energy = 0.0
            if building.ev and (self.current_timestep >= building.ev.arrival_time or self.current_timestep < building.ev.departure_time):
                ev_present = True
                ev_soc = building.ev.arrival_soc
                ev_energy = ev_soc * building.ev.capacity_kwh

            # Calculate power limits
            battery_max_charge = building.battery.capacity_kwh * building.battery.c_rate if building.battery else 0.0
            battery_max_discharge = building.battery.capacity_kwh * building.battery.c_rate if building.battery else 0.0
            ev_max_charge = building.ev.capacity_kwh * building.ev.c_rate if (building.ev and ev_present) else 0.0
            ev_max_discharge = building.ev.capacity_kwh * building.ev.c_rate if (building.ev and ev_present) else 0.0

            state = State(
                building_id=i,
                timestep=self.current_timestep,
                load_demand=building.load[self.current_timestep],
                pv_generation=building.pv.pv_gen[self.current_timestep] if building.pv else 0.0,
                battery_soc=battery_soc,
                battery_energy=battery_energy,
                ev_present=ev_present,
                ev_soc=ev_soc,
                ev_energy=ev_energy,
                battery_max_charge_power=battery_max_charge,
                battery_max_discharge_power=battery_max_discharge,
                ev_max_charge_power=ev_max_charge,
                ev_max_discharge_power=ev_max_discharge
            )
            states.append(state)

        return states

    def get_observation(self) -> Observation:
        """
        Get current system observation.

        Returns:
            Observation: Current state of all buildings
        """
        return Observation(
            current_timestep=self.current_timestep,
            building_states=copy.deepcopy(self.building_states),
            electricity_price=self.config.global_config.TOU[self.current_timestep]
        )

    def step(self, actions: List[Action]) -> Tuple[Observation, List[Record], bool]:
        """
        Execute one simulation step.

        Args:
            actions: List of actions for each building

        Returns:
            Tuple of (next_observation, energy_flow_records, done)
        """
        if self.current_timestep >= self.max_timesteps:
            raise ValueError("Simulation has already ended")

        # Validate actions to avoid exceed constraints
        flow_records = []
        for action in actions:
            record = self._validate_action(action)
            flow_records.append(record)
            self.energy_flow_history.append(record)

        # Update metrics
        self._update_metrics(flow_records)

        # Advance timestep
        self.current_timestep += 1
        done = self.current_timestep >= self.max_timesteps

        if not done:
            # Update building states for next timestep
            self._update_building_states()

        # Return next observation
        next_obs = self.get_observation() if not done else None

        return next_obs, flow_records, done

    def ___validate_action(self, action: Action) -> Record:
        building_state = self.building_states[action.building_id]
        building_config = self.buildings[action.building_id]
        actual_action = copy.deepcopy(action)
        clipping_details = {}
        clipping_occurred = False

        def clip_by_priority(sources: List[str], available: float):
            result = {}
            for src in sources:
                val = getattr(actual_action, src)
                if val <= available:
                    result[src] = val
                    available -= val
                else:
                    result[src] = available
                    clipping_details[src] = (val, available)
                    available = 0
                    nonlocal clipping_occurred
                    clipping_occurred = True
            return result

        # PV validation
        available_pv = building_state.pv_generation
        # The priority logic is based on money, use "free" energy at the current step first
        pv_priority = ['pv_to_building', 'pv_to_battery', 'pv_to_ev', 'pv_to_grid']
        pv_result = clip_by_priority(pv_priority, available_pv)
        for k, v in pv_result.items():
            setattr(actual_action, k, v)
        actual_action.pv_dumped = max(0.0, available_pv - sum(pv_result.values()))

        building_demand = building_state.load_demand
        building_supply_fields = ['pv_to_building', 'battery_to_building', 'ev_to_building', 'grid_to_building']
        current_supply = sum(getattr(actual_action, f) for f in building_supply_fields)

        if current_supply > building_demand:
            remaining = building_demand
            for f in building_supply_fields:
                val = getattr(actual_action, f)
                if val <= remaining:
                    remaining -= val
                else:
                    clipping_details[f] = (val, remaining)
                    setattr(actual_action, f, remaining)
                    clipping_occurred = True
                    remaining = 0

        if building_config.battery:
            max_energy = (
                        building_config.battery.max_soc * building_config.battery.capacity_kwh - building_state.battery_energy)
            max_power = min(building_state.battery_max_charge_power, max_energy * (60 / self.config.global_config.Res))
            battery_charge_priority = ['pv_to_battery', 'ev_to_battery', 'grid_to_battery']
            battery_result = clip_by_priority(battery_charge_priority, max_power)
            for k, v in battery_result.items():
                setattr(actual_action, k, v)

        # Battery Charging
        if not building_config.battery:
            for field in ['pv_to_battery', 'ev_to_battery', 'grid_to_battery',
                          'battery_to_building', 'battery_to_ev', 'battery_to_grid']:
                original = getattr(actual_action, field)
                if original > 0.0:
                    clipping_details[field] = (original, 0.0)
                    clipping_occurred = True
                setattr(actual_action, field, 0.0)

        # Battery Discharging
        if building_config.battery:
            available_energy = (
                        building_state.battery_energy - building_config.battery.min_soc * building_config.battery.capacity_kwh)
            max_power = min(building_state.battery_max_discharge_power,
                            available_energy * (60 / self.config.global_config.Res))
            battery_discharge_priority = ['battery_to_building', 'battery_to_ev', 'battery_to_grid']
            battery_result = clip_by_priority(battery_discharge_priority, max_power)
            for k, v in battery_result.items():
                setattr(actual_action, k, v)

        # EV Charging & Discharging
        if not building_config.ev or not building_state.ev_present:
            for field in ['pv_to_ev', 'battery_to_ev', 'grid_to_ev',
                          'ev_to_building', 'ev_to_grid', 'ev_to_battery']:
                original = getattr(actual_action, field)
                if original > 0.0:
                    clipping_details[field] = (original, 0.0)
                    clipping_occurred = True
                setattr(actual_action, field, 0.0)

        if building_config.ev and building_state.ev_present:
            # EV Charging
            max_energy = (building_config.ev.max_soc * building_config.ev.capacity_kwh - building_state.ev_energy)
            max_power = min(building_state.ev_max_charge_power, max_energy * (60 / self.config.global_config.Res))
            ev_charge_priority = ['pv_to_ev', 'battery_to_ev', 'grid_to_ev']
            ev_charge_result = clip_by_priority(ev_charge_priority, max_power)
            for k, v in ev_charge_result.items():
                setattr(actual_action, k, v)

            # EV Discharging
            available_energy = (building_state.ev_energy - building_config.ev.min_soc * building_config.ev.capacity_kwh)
            max_power = min(building_state.ev_max_discharge_power,
                            available_energy * (60 / self.config.global_config.Res))
            ev_discharge_priority = ['ev_to_building', 'ev_to_grid', 'ev_to_battery']
            ev_discharge_result = clip_by_priority(ev_discharge_priority, max_power)
            for k, v in ev_discharge_result.items():
                setattr(actual_action, k, v)

        used_pv = actual_action.pv_to_building + actual_action.pv_to_battery + actual_action.pv_to_ev + actual_action.pv_to_grid
        actual_action.pv_dumped = max(0.0, building_state.pv_generation - used_pv)

        self._apply_action_to_state(building_state, actual_action, building_config)

        return Record(
            action=action,
            actual_flows=actual_action,
            clipping_occurred=clipping_occurred,
            clipping_details=clipping_details
        )

    def _validate_action(self, action: Action) -> Record:
        building_state = self.building_states[action.building_id]
        building_config = self.buildings[action.building_id]
        actual_action = copy.deepcopy(action)
        clipping_details = {}
        clipping_occurred = False

        def clip_by_priority(sources: List[str], available: float):
            result = {}
            for src in sources:
                val = getattr(actual_action, src)
                if val <= available:
                    result[src] = val
                    available -= val
                else:
                    result[src] = available
                    clipping_details[src] = (val, available)
                    available = 0
                    nonlocal clipping_occurred
                    clipping_occurred = True
            return result

        # ===== Clip to zero if no battery =====
        if not building_config.battery:
            for field in ['pv_to_battery', 'ev_to_battery', 'grid_to_battery',
                          'battery_to_building', 'battery_to_ev', 'battery_to_grid']:
                original = getattr(actual_action, field)
                if original > 0.0:
                    clipping_details[field] = (original, 0.0)
                    clipping_occurred = True
                setattr(actual_action, field, 0.0)

        # ===== Clip to zero if no EV or EV not present =====
        if not building_config.ev or not building_state.ev_present:
            for field in ['pv_to_ev', 'battery_to_ev', 'grid_to_ev',
                          'ev_to_building', 'ev_to_grid', 'ev_to_battery']:
                original = getattr(actual_action, field)
                if original > 0.0:
                    clipping_details[field] = (original, 0.0)
                    clipping_occurred = True
                setattr(actual_action, field, 0.0)

        # ===== PV clipping =====
        available_pv = building_state.pv_generation
        pv_priority = ['pv_to_building', 'pv_to_battery', 'pv_to_ev', 'pv_to_grid']
        pv_result = clip_by_priority(pv_priority, available_pv)
        for k, v in pv_result.items():
            setattr(actual_action, k, v)
        actual_action.pv_dumped = max(0.0, available_pv - sum(pv_result.values()))

        # ===== Battery charging =====
        if building_config.battery:
            max_energy = building_config.battery.max_soc * building_config.battery.capacity_kwh - building_state.battery_energy
            max_power = min(building_state.battery_max_charge_power, max_energy * (60 / self.config.global_config.Res))
            battery_charge_priority = ['pv_to_battery', 'ev_to_battery', 'grid_to_battery']
            battery_result = clip_by_priority(battery_charge_priority, max_power)
            for k, v in battery_result.items():
                setattr(actual_action, k, v)

        # ===== Battery discharging =====
        if building_config.battery:
            available_energy = building_state.battery_energy - building_config.battery.min_soc * building_config.battery.capacity_kwh
            max_power = min(building_state.battery_max_discharge_power,
                            available_energy * (60 / self.config.global_config.Res))
            battery_discharge_priority = ['battery_to_building', 'battery_to_ev', 'battery_to_grid']
            battery_result = clip_by_priority(battery_discharge_priority, max_power)
            for k, v in battery_result.items():
                setattr(actual_action, k, v)

        # ===== EV charging/discharging =====
        if building_config.ev and building_state.ev_present:
            # Charging
            max_energy = building_config.ev.max_soc * building_config.ev.capacity_kwh - building_state.ev_energy
            max_power = min(building_state.ev_max_charge_power, max_energy * (60 / self.config.global_config.Res))
            ev_charge_priority = ['pv_to_ev', 'battery_to_ev', 'grid_to_ev']
            ev_charge_result = clip_by_priority(ev_charge_priority, max_power)
            for k, v in ev_charge_result.items():
                setattr(actual_action, k, v)

            # Discharging
            available_energy = building_state.ev_energy - building_config.ev.min_soc * building_config.ev.capacity_kwh
            max_power = min(building_state.ev_max_discharge_power,
                            available_energy * (60 / self.config.global_config.Res))
            ev_discharge_priority = ['ev_to_building', 'ev_to_grid', 'ev_to_battery']
            ev_discharge_result = clip_by_priority(ev_discharge_priority, max_power)
            for k, v in ev_discharge_result.items():
                setattr(actual_action, k, v)

        # ===== Demand-based clipping (AFTER all *_to_building are clipped) =====
        building_demand = building_state.load_demand
        building_supply_fields = ['pv_to_building', 'battery_to_building', 'ev_to_building', 'grid_to_building']
        current_supply = sum(getattr(actual_action, f) for f in building_supply_fields)
        if current_supply > building_demand:
            remaining = building_demand
            for f in building_supply_fields:
                val = getattr(actual_action, f)
                if val <= remaining:
                    remaining -= val
                else:
                    clipping_details[f] = (val, remaining)
                    setattr(actual_action, f, remaining)
                    clipping_occurred = True
                    remaining = 0

        # ===== Recalculate PV dumped =====
        used_pv = actual_action.pv_to_building + actual_action.pv_to_battery + actual_action.pv_to_ev + actual_action.pv_to_grid
        actual_action.pv_dumped = max(0.0, building_state.pv_generation - used_pv)

        self._apply_action_to_state(building_state, actual_action, building_config)

        return Record(
            action=action,
            actual_flows=actual_action,
            clipping_occurred=clipping_occurred,
            clipping_details=clipping_details
        )


    def _apply_action_to_state(self, state: State, action: Action, config: 'BuildingConfig'):
        """Apply validated action to update building state."""
        dt_hours = self.config.global_config.Res / 60.0  # Convert minutes to hours

        # Update battery energy
        if config.battery:
            battery_charge_energy = (action.pv_to_battery * config.battery.charge_efficiency +
                                   action.grid_to_battery * config.battery.charge_efficiency +
                                   action.ev_to_battery * config.battery.charge_efficiency) * dt_hours

            battery_discharge_energy = ((action.battery_to_building + action.battery_to_ev +
                                       action.battery_to_grid) / config.battery.discharge_efficiency) * dt_hours

            state.battery_energy += battery_charge_energy - battery_discharge_energy
            state.battery_energy = np.clip(state.battery_energy,
                                         config.battery.min_soc * config.battery.capacity_kwh,
                                         config.battery.max_soc * config.battery.capacity_kwh)
            state.battery_soc = state.battery_energy / config.battery.capacity_kwh

        # Update EV energy
        if config.ev and state.ev_present:
            ev_charge_energy = (action.pv_to_ev * config.ev.charge_efficiency +
                              action.battery_to_ev * config.ev.charge_efficiency +
                              action.grid_to_ev * config.ev.charge_efficiency) * dt_hours

            ev_discharge_energy = ((action.ev_to_building + action.ev_to_grid +
                                  action.ev_to_battery) / config.ev.discharge_efficiency) * dt_hours

            state.ev_energy += ev_charge_energy - ev_discharge_energy
            state.ev_energy = np.clip(state.ev_energy,
                                    config.ev.min_soc * config.ev.capacity_kwh,
                                    config.ev.max_soc * config.ev.capacity_kwh)
            state.ev_soc = state.ev_energy / config.ev.capacity_kwh

    def _update_building_states(self):
        """Update building states for the new timestep."""
        for i, state in enumerate(self.building_states):
            building = self.buildings[i]

            # Update basic state
            state.timestep = self.current_timestep
            state.load_demand = building.load[self.current_timestep]
            state.pv_generation = building.pv.pv_gen[self.current_timestep] if building.pv else 0.0

            # Update EV presence
            if building.ev:
                state.ev_present = (self.current_timestep >= building.ev.arrival_time and
                                  self.current_timestep < building.ev.departure_time)
                if not state.ev_present:
                    state.ev_soc = 0.0
                    state.ev_energy = 0.0

            # Update power limits
            state.battery_max_charge_power = (building.battery.capacity_kwh * building.battery.c_rate
                                            if building.battery else 0.0)
            state.battery_max_discharge_power = (building.battery.capacity_kwh * building.battery.c_rate
                                               if building.battery else 0.0)
            state.ev_max_charge_power = (building.ev.capacity_kwh * building.ev.c_rate
                                       if building.ev and state.ev_present else 0.0)
            state.ev_max_discharge_power = (building.ev.capacity_kwh * building.ev.c_rate
                                          if building.ev and state.ev_present else 0.0)

    def _update_metrics(self, flow_records: List[Record]):
        """Update performance metrics."""
        total_grid_import = 0.0
        total_grid_export = 0.0
        total_load_served = 0.0
        total_pv_generation = 0.0

        for record in flow_records:
            action = record.actual_flows
            building_state = self.building_states[action.building_id]

            # Grid flows
            grid_import = action.grid_to_building + action.grid_to_battery + action.grid_to_ev
            grid_export = action.pv_to_grid + action.battery_to_grid + action.ev_to_grid

            total_grid_import += grid_import
            total_grid_export += grid_export
            total_load_served += building_state.load_demand
            total_pv_generation += building_state.pv_generation

        self.metrics['grid_import'].append(total_grid_import)
        self.metrics['grid_export'].append(total_grid_export)
        self.metrics['load_served'].append(total_load_served)
        self.metrics['pv_generation'].append(total_pv_generation)
        self.metrics['net_grid_flow'].append(total_grid_import - total_grid_export)

    def reset(self):
        """Reset the simulation to initial state."""
        self.current_timestep = 0
        self.building_states = self._initialize_building_states()
        self.energy_flow_history.clear()
        self.metrics.clear()

    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary metrics for the simulation."""
        if not self.metrics:
            return {}

        return {
            'total_grid_import_kwh': sum(self.metrics['grid_import']) * self.config.global_config.Res / 60,
            'total_grid_export_kwh': sum(self.metrics['grid_export']) * self.config.global_config.Res / 60,
            'total_load_kwh': sum(self.metrics['load_served']) * self.config.global_config.Res / 60,
            'total_pv_generation_kwh': sum(self.metrics['pv_generation']) * self.config.global_config.Res / 60,
            'peak_grid_import_kw': max(self.metrics['grid_import']) if self.metrics['grid_import'] else 0,
            'peak_grid_export_kw': max(self.metrics['grid_export']) if self.metrics['grid_export'] else 0,
            'self_consumption_ratio': 1 - sum(self.metrics['grid_export']) / sum(self.metrics['pv_generation'])
                                    if sum(self.metrics['pv_generation']) > 0 else 0,
        }


# Example usage and helper functions
def create_simple_action(building_id: int, timestep: int, **kwargs) -> Action:
    """Helper function to create actions with defaults."""
    return Action(building_id=building_id, timestep=timestep, **kwargs)

# Create an emulator

# Create a dummy global config (you must replace with real data)
global_config = GlobalConfig(
    T_amb=np.array([25]*24),     # ambient temperature for 24 hours
    Sol=np.array([500]*24),      # solar radiation
    TOU=np.array([0.1]*24),      # time-of-use price
    Res=60,                       # time resolution (minutes)
    cpus=4,
)

# Create Config
my_config = Config(global_config=global_config)
pv_params = {
    'capacity_kw': 20  # PV Capacity: 4 kW
}
battery_params = {
    'capacity_kwh': 10.0,          # Battery Capacity: 5 kWh
    'c_rate': 0.25,               # 0.25C charge/discharge rate (4 hours)
    'charge_efficiency': 0.95,    # 95% charging efficiency
    'discharge_efficiency': 0.95, # 95% discharging efficiency
    'min_soc': 0.1,               # 10% minimum SOC
    'max_soc': 0.9,               # 90% maximum SOC
    'initial_soc': 0.5            # Start at 50% SOC
}

ev_params = {
    'capacity_kwh': 50.0,          # Battery Capacity: 5 kWh
    'c_rate': 0.2,               # 0.25C charge/discharge rate (4 hours)
    'charge_efficiency': 0.95,    # 95% charging efficiency
    'discharge_efficiency': 0.95, # 95% discharging efficiency
    'min_soc': 0.1,               # 10% minimum SOC
    'max_soc': 1.0,               # 90% maximum SOC
    'departure_time': 8*4,
    'arrival_time': 18*4,
    'arrival_soc': 0.6,
    'required_departure_soc': 0.95
}
building = my_config.create_building(
    load=np.array([50.0]*24),      # Assign Building load (Need to be kW)
    pv_params=pv_params,           # Assign PV parameters
    battery_params=battery_params,  # Assign Battery parameters
    ev_params=ev_params

)


# Make a list of buildings
my_buildings = [building]

# Now you can initialize the emulator
emulator = BuildingToGridEmulator(config=my_config, buildings=my_buildings)

# Run simulation for one step
obs = emulator.get_observation()

# Create an action with excessive requests
example_action = create_simple_action(
    building_id=0,
    timestep=obs.current_timestep,
    pv_to_building=5.0,
    pv_to_battery=5.0,
    pv_to_grid=5.0,
    battery_to_building=10.0,
    battery_to_ev=5.0,
    ev_to_grid=2.0,
    ev_to_building=10.0,
)

next_obs, records, done = emulator.step([example_action])

# Check the result
print("Actual Flows:", records[0].actual_flows)
print("Clipping Details:", records[0].clipping_details)
