"""
Revised Electrical Supervisory Controller with improved structure and bug fixes
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from enum import Enum
from dataclasses import dataclass, field

from ..core.base import BaseModule
from ..core.data_structure import (
    State, Disturbance, Observation,
    ElectricalDomainState, DERSystemState,
    BatteryState, PVState, EVState
)


@dataclass
class ElectricalAction:
    """Enhanced Electrical Action with all power flow attributes."""
    # PV flows
    pv2building: float = 0.0
    pv2battery: float = 0.0
    pv2ev: float = 0.0
    pv2grid: float = 0.0

    # Battery flows
    battery2building: float = 0.0
    battery2ev: float = 0.0
    battery2grid: float = 0.0

    # EV flows
    ev2building: float = 0.0
    ev2battery: float = 0.0
    ev2grid: float = 0.0

    # Grid flows
    grid2building: float = 0.0
    grid2battery: float = 0.0
    grid2ev: float = 0.0

    # Aggregated commands per component
    battery_commands: Dict[str, float] = field(default_factory=dict)  # battery_id -> power command
    ev_commands: Dict[str, float] = field(default_factory=dict)  # ev_id -> power command
    pv_commands: Dict[str, float] = field(default_factory=dict)  # pv_id -> curtailment factor


class ControlMode(Enum):
    """Control modes for the electrical system."""
    SELF_CONSUMPTION = "self_consumption"
    TIME_OF_USE = "time_of_use"
    DEMAND_RESPONSE = "demand_response"
    ISLANDED = "islanded"


class SupervisoryController(BaseModule):
    """
    Revised rule-based electrical supervisory controller for DER power flow management.

    Improvements:
    - Handles multiple components of same type properly
    - Clear power flow allocation strategy
    - Better state management
    - Robust error handling
    """

    def __init__(self, config: Dict[str, Any], name: str = "SupervisoryController"):
        """Initialize Electrical Supervisory Controller."""
        super().__init__(config, name)

        # Domain configuration
        self.domain = config.get("domain", "electrical")
        self.control_mode = ControlMode(config.get("mode", "self_consumption"))

        # System configuration
        self.system_config = config.get("system_config", {})

        # SOC thresholds
        self.bat_soc_min = config.get("bat_soc_min", 0.1)
        self.bat_soc_max = config.get("bat_soc_max", 0.9)
        self.bat_soc_reserve = config.get("bat_soc_reserve", 0.2)

        self.ev_soc_min = config.get("ev_soc_min", 0.2)
        self.ev_soc_target = config.get("ev_soc_target", 0.8)
        self.ev_v2g_enabled = config.get("ev_v2g_enabled", True)

        # Power limits
        self.max_grid_import = config.get("max_grid_import", 10000)  # Watts
        self.max_grid_export = config.get("max_grid_export", 5000)  # Watts

        # Default charging/discharging rates (will be overridden by component configs)
        self.default_bat_charge_rate = config.get("bat_charge_rate", 2000)
        self.default_bat_discharge_rate = config.get("bat_discharge_rate", 2000)
        self.default_ev_charge_rate = config.get("ev_charge_rate", 3000)
        self.default_ev_discharge_rate = config.get("ev_discharge_rate", 2000)

        # Grid status
        self.grid_connected = True

        # Component detection and configuration
        self._detect_and_configure_components()

        self.logger = logging.getLogger(f"{__name__}.{name}")

    def initialize(self) -> None:
        """Initialize the controller."""
        self.logger.info(f"Initialized electrical controller: {self.name}")
        self.logger.info(f"Components detected - PV: {len(self.pv_configs)}, "
                        f"Batteries: {len(self.battery_configs)}, EVs: {len(self.ev_configs)}")

    def step(self,
             state: Any,
             observation: Observation,
             disturbance: Disturbance,
             timestep: float) -> ElectricalAction:
        """
        Determine electrical control action based on current conditions.

        Handles multiple components properly by aggregating states and
        distributing commands proportionally.
        """
        try:
            # Extract system state properly
            der_state = self._extract_der_state(state)
            if not der_state:
                self.logger.warning("No DER system state found")
                return ElectricalAction()

            # Get aggregated system status
            building_load = self._get_building_load(state)
            pv_generation_total = self._get_total_pv_generation(der_state, disturbance)
            battery_states = self._get_all_battery_states(der_state)
            ev_states = self._get_all_ev_states(der_state)

            # Get control signals
            is_peak = self._is_peak_period(disturbance)
            self.grid_connected = self._get_grid_status(disturbance)

            # Determine control strategy based on mode
            if not self.grid_connected or self.control_mode == ControlMode.ISLANDED:
                action = self._islanded_mode_control(
                    building_load, pv_generation_total,
                    battery_states, ev_states, disturbance
                )
            elif self.control_mode == ControlMode.TIME_OF_USE:
                action = self._time_of_use_control(
                    building_load, pv_generation_total,
                    battery_states, ev_states, is_peak, disturbance
                )
            else:  # SELF_CONSUMPTION mode (default)
                action = self._self_consumption_control(
                    building_load, pv_generation_total,
                    battery_states, ev_states, disturbance
                )

            # Log decision
            self._log_control_decision(action, building_load, pv_generation_total)

            return action

        except Exception as e:
            self.logger.error(f"Error in electrical controller step: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ElectricalAction()

    def _detect_and_configure_components(self) -> None:
        """Detect and configure all components in the system."""
        # Parse PV configurations
        self.pv_configs = self._parse_component_config('pv_systems', 'pv')
        self.has_pv = len(self.pv_configs) > 0

        # Parse battery configurations
        self.battery_configs = self._parse_component_config('batteries', 'bat')
        self.has_battery = len(self.battery_configs) > 0

        # Parse EV configurations
        self.ev_configs = self._parse_component_config('evs', 'ev')
        self.has_ev = len(self.ev_configs) > 0

    def _parse_component_config(self, plural_key: str, singular_key: str) -> Dict[str, Dict[str, Any]]:
        """Parse component configuration supporting both single and multiple formats."""
        configs = {}

        # Check for multiple components
        if plural_key in self.system_config:
            multi_config = self.system_config[plural_key]
            if isinstance(multi_config, list):
                for idx, comp_config in enumerate(multi_config):
                    comp_id = comp_config.get('id', f"{singular_key}_{idx + 1}")
                    configs[comp_id] = comp_config
            elif isinstance(multi_config, dict):
                configs = multi_config

        # Check for single component (backward compatibility)
        elif singular_key in self.system_config:
            single_config = self.system_config[singular_key]
            comp_id = single_config.get('id', f"{singular_key}_1")
            configs[comp_id] = single_config

        return configs

    def _extract_der_state(self, state: Any) -> Optional[DERSystemState]:
        """Extract DER system state from the provided state object."""
        # Handle different state types
        if isinstance(state, DERSystemState):
            return state
        elif isinstance(state, ElectricalDomainState):
            # Get first DER system
            if state.der_systems:
                return list(state.der_systems.values())[0]
        elif hasattr(state, 'electrical'):
            # Full building state
            if hasattr(state.electrical, 'der_systems') and state.electrical.der_systems:
                return list(state.electrical.der_systems.values())[0]

        return None

    def _get_building_load(self, state: Any) -> float:
        """Get total building electrical load in Watts."""
        # This should come from the building electrical state
        if hasattr(state, 'electrical') and hasattr(state.electrical, 'building_loads'):
            total_load = sum(load.base_load for load in state.electrical.building_loads.values())
            return total_load
        return 1000.0  # Default fallback

    def _get_total_pv_generation(self, der_state: DERSystemState, disturbance: Disturbance) -> float:
        """Get total PV generation from all PV systems."""
        if not self.has_pv or not der_state.pv_systems:
            return 0.0

        total_generation = 0.0
        for pv_id, pv_state in der_state.pv_systems.items():
            # Use actual generation if available, otherwise estimate
            if hasattr(pv_state, 'pv_generation'):
                total_generation += pv_state.pv_generation
            else:
                # Estimate based on solar radiation
                pv_config = self.pv_configs.get(pv_id, {})
                capacity_kw = pv_config.get('rated_capacity_kW', 2.0)
                solar_radiation = disturbance.weather.solar_radiation if disturbance.weather else 0
                efficiency = 0.001  # W/m² to kW conversion with panel efficiency
                total_generation += solar_radiation * capacity_kw * efficiency

        return total_generation

    def _get_all_battery_states(self, der_state: DERSystemState) -> Dict[str, Dict[str, Any]]:
        """Get states of all batteries."""
        battery_states = {}

        if not self.has_battery or not der_state.batteries:
            return battery_states

        for bat_id, bat_state in der_state.batteries.items():
            bat_config = self.battery_configs.get(bat_id, {})
            battery_states[bat_id] = {
                'soc': bat_state.battery_soc,
                'capacity_kwh': bat_config.get('rated_capacity_kWh', 5.0),
                'max_charge_rate': bat_config.get('max_charge_rate', self.default_bat_charge_rate),
                'max_discharge_rate': bat_config.get('max_discharge_rate', self.default_bat_discharge_rate),
                'state': bat_state
            }

        return battery_states

    def _get_all_ev_states(self, der_state: DERSystemState) -> Dict[str, Dict[str, Any]]:
        """Get states of all EVs."""
        ev_states = {}

        if not self.has_ev or not der_state.evs:
            return ev_states

        for ev_id, ev_state in der_state.evs.items():
            ev_config = self.ev_configs.get(ev_id, {})
            ev_states[ev_id] = {
                'soc': ev_state.ev_soc,
                'connected': ev_state.is_connected,
                'capacity_kwh': ev_config.get('rated_capacity_kWh', 40.0),
                'max_charge_rate': ev_config.get('max_charge_rate', self.default_ev_charge_rate),
                'max_discharge_rate': ev_config.get('max_discharge_rate', self.default_ev_discharge_rate),
                'state': ev_state
            }

        return ev_states

    def _self_consumption_control(self, building_load: float, pv_generation: float,
                                  battery_states: Dict, ev_states: Dict,
                                  disturbance: Disturbance) -> ElectricalAction:
        """
        Self-consumption mode: Maximize use of local generation.
        Priority: PV → Building → Battery → EV → Grid Export
        """
        action = ElectricalAction()

        # First use PV for building
        pv_to_building = min(pv_generation, building_load)
        action.pv2building = pv_to_building
        remaining_pv = pv_generation - pv_to_building
        remaining_load = building_load - pv_to_building

        # If deficit, use batteries proportionally
        if remaining_load > 0 and battery_states:
            remaining_load = self._discharge_batteries_proportionally(
                action, battery_states, remaining_load
            )

        # If still deficit, use EVs if available
        if remaining_load > 0 and ev_states:
            remaining_load = self._discharge_evs_proportionally(
                action, ev_states, remaining_load
            )

        # If still deficit, import from grid
        if remaining_load > 0:
            action.grid2building = min(remaining_load, self.max_grid_import)

        # If excess PV, charge batteries proportionally
        if remaining_pv > 0 and battery_states:
            remaining_pv = self._charge_batteries_proportionally(
                action, battery_states, remaining_pv
            )

        # If still excess, charge EVs
        if remaining_pv > 0 and ev_states:
            remaining_pv = self._charge_evs_proportionally(
                action, ev_states, remaining_pv
            )

        # Export remaining to grid
        if remaining_pv > 0:
            action.pv2grid = min(remaining_pv, self.max_grid_export)

        return action

    def _time_of_use_control(self, building_load: float, pv_generation: float,
                             battery_states: Dict, ev_states: Dict,
                             is_peak: bool, disturbance: Disturbance) -> ElectricalAction:
        """
        Time-of-use mode: Optimize based on peak/off-peak periods.
        Peak: Minimize grid import, use storage
        Off-peak: Charge storage from grid
        """
        action = ElectricalAction()

        # Always use PV for building first
        pv_to_building = min(pv_generation, building_load)
        action.pv2building = pv_to_building
        remaining_pv = pv_generation - pv_to_building
        remaining_load = building_load - pv_to_building

        if is_peak:
            # PEAK: Discharge storage to minimize grid import
            if remaining_load > 0 and battery_states:
                remaining_load = self._discharge_batteries_proportionally(
                    action, battery_states, remaining_load, reserve_soc=self.bat_soc_reserve
                )

            if remaining_load > 0 and ev_states and self.ev_v2g_enabled:
                remaining_load = self._discharge_evs_proportionally(
                    action, ev_states, remaining_load
                )

            # Use excess PV to charge storage
            if remaining_pv > 0:
                if battery_states:
                    remaining_pv = self._charge_batteries_proportionally(
                        action, battery_states, remaining_pv
                    )
                if remaining_pv > 0:
                    action.pv2grid = min(remaining_pv, self.max_grid_export)

            # Import from grid only if necessary
            if remaining_load > 0:
                action.grid2building = min(remaining_load, self.max_grid_import)

        else:
            # OFF-PEAK: Charge storage from grid
            # Supply building from grid
            if remaining_load > 0:
                action.grid2building = remaining_load

            # Charge batteries to max
            if battery_states:
                total_charge_needed = self._calculate_total_charge_needed(battery_states, self.bat_soc_max)
                if total_charge_needed > 0:
                    # Use PV first
                    if remaining_pv > 0:
                        pv_charge = min(remaining_pv, total_charge_needed)
                        self._charge_batteries_proportionally(action, battery_states, pv_charge)
                        remaining_pv -= pv_charge
                        total_charge_needed -= pv_charge

                    # Then use grid
                    if total_charge_needed > 0:
                        grid_charge = min(total_charge_needed, self.max_grid_import - action.grid2building)
                        self._charge_batteries_from_grid(action, battery_states, grid_charge)

            # Charge EVs
            if ev_states:
                total_ev_charge_needed = self._calculate_total_charge_needed(ev_states, self.ev_soc_target)
                if total_ev_charge_needed > 0:
                    # Use remaining PV
                    if remaining_pv > 0:
                        pv_charge = min(remaining_pv, total_ev_charge_needed)
                        self._charge_evs_proportionally(action, ev_states, pv_charge)
                        remaining_pv -= pv_charge
                        total_ev_charge_needed -= pv_charge

                    # Then use grid
                    if total_ev_charge_needed > 0:
                        current_grid_usage = action.grid2building + action.grid2battery
                        grid_charge = min(total_ev_charge_needed, self.max_grid_import - current_grid_usage)
                        self._charge_evs_from_grid(action, ev_states, grid_charge)

            # Export any remaining PV
            if remaining_pv > 0:
                action.pv2grid = min(remaining_pv, self.max_grid_export)

        return action

    def _islanded_mode_control(self, building_load: float, pv_generation: float,
                               battery_states: Dict, ev_states: Dict,
                               disturbance: Disturbance) -> ElectricalAction:
        """
        Islanded mode: No grid connection, careful resource management.
        Priority: Critical loads only, preserve battery reserve
        """
        action = ElectricalAction()

        # No grid flows in islanded mode
        action.grid2building = 0
        action.grid2battery = 0
        action.grid2ev = 0
        action.pv2grid = 0

        # Use PV for building
        pv_to_building = min(pv_generation, building_load)
        action.pv2building = pv_to_building
        remaining_pv = pv_generation - pv_to_building
        remaining_load = building_load - pv_to_building

        # Use batteries carefully (maintain reserve)
        if remaining_load > 0 and battery_states:
            remaining_load = self._discharge_batteries_proportionally(
                action, battery_states, remaining_load,
                reserve_soc=self.bat_soc_reserve * 1.5  # Higher reserve in islanded mode
            )

        # Use EV only if critical
        if remaining_load > 0 and ev_states and self.ev_v2g_enabled:
            # Only use if batteries are depleted
            avg_bat_soc = self._get_average_soc(battery_states)
            if avg_bat_soc < self.bat_soc_reserve * 2:
                remaining_load = self._discharge_evs_proportionally(
                    action, ev_states, remaining_load,
                    reserve_soc=self.ev_soc_min + 0.1
                )

        # Store excess PV
        if remaining_pv > 0:
            if battery_states:
                remaining_pv = self._charge_batteries_proportionally(
                    action, battery_states, remaining_pv
                )
            if remaining_pv > 0 and ev_states:
                remaining_pv = self._charge_evs_proportionally(
                    action, ev_states, remaining_pv
                )

        # Log if load cannot be met
        if remaining_load > 0:
            self.logger.warning(f"Cannot meet {remaining_load:.1f}W of load in islanded mode")

        # Log if PV is curtailed
        if remaining_pv > 0:
            self.logger.info(f"Curtailing {remaining_pv:.1f}W of PV in islanded mode")

        return action

    # Helper methods for proportional control

    def _discharge_batteries_proportionally(self, action: ElectricalAction,
                                           battery_states: Dict, power_needed: float,
                                           reserve_soc: float = None) -> float:
        """Discharge batteries proportionally based on available capacity."""
        if reserve_soc is None:
            reserve_soc = self.bat_soc_min

        total_available = 0
        available_per_battery = {}

        # Calculate available discharge from each battery
        for bat_id, bat_info in battery_states.items():
            soc = bat_info['soc']
            if soc > reserve_soc:
                available_energy = (soc - reserve_soc) * bat_info['capacity_kwh'] * 1000  # Wh to W
                available_power = min(available_energy, bat_info['max_discharge_rate'])
                available_per_battery[bat_id] = available_power
                total_available += available_power

        if total_available == 0:
            return power_needed

        # Distribute discharge proportionally
        power_allocated = min(power_needed, total_available)
        remaining_power = power_needed - power_allocated

        for bat_id, available_power in available_per_battery.items():
            proportion = available_power / total_available
            discharge_power = power_allocated * proportion
            action.battery_commands[bat_id] = -discharge_power  # Negative for discharge

        action.battery2building = power_allocated

        return remaining_power

    def _charge_batteries_proportionally(self, action: ElectricalAction,
                                        battery_states: Dict, power_available: float) -> float:
        """Charge batteries proportionally based on capacity headroom."""
        total_headroom = 0
        headroom_per_battery = {}

        # Calculate charging headroom for each battery
        for bat_id, bat_info in battery_states.items():
            soc = bat_info['soc']
            if soc < self.bat_soc_max:
                headroom_energy = (self.bat_soc_max - soc) * bat_info['capacity_kwh'] * 1000
                headroom_power = min(headroom_energy, bat_info['max_charge_rate'])
                headroom_per_battery[bat_id] = headroom_power
                total_headroom += headroom_power

        if total_headroom == 0:
            return power_available

        # Distribute charging proportionally
        power_allocated = min(power_available, total_headroom)
        remaining_power = power_available - power_allocated

        for bat_id, headroom_power in headroom_per_battery.items():
            proportion = headroom_power / total_headroom
            charge_power = power_allocated * proportion
            if bat_id not in action.battery_commands:
                action.battery_commands[bat_id] = 0
            action.battery_commands[bat_id] += charge_power  # Positive for charge

        action.pv2battery += power_allocated

        return remaining_power

    def _discharge_evs_proportionally(self, action: ElectricalAction,
                                     ev_states: Dict, power_needed: float,
                                     reserve_soc: float = None) -> float:
        """Discharge connected EVs proportionally."""
        if reserve_soc is None:
            reserve_soc = self.ev_soc_min

        total_available = 0
        available_per_ev = {}

        for ev_id, ev_info in ev_states.items():
            if ev_info['connected'] and ev_info['soc'] > reserve_soc:
                available_energy = (ev_info['soc'] - reserve_soc) * ev_info['capacity_kwh'] * 1000
                available_power = min(available_energy, ev_info['max_discharge_rate'])
                available_per_ev[ev_id] = available_power
                total_available += available_power

        if total_available == 0:
            return power_needed

        power_allocated = min(power_needed, total_available)
        remaining_power = power_needed - power_allocated

        for ev_id, available_power in available_per_ev.items():
            proportion = available_power / total_available
            discharge_power = power_allocated * proportion
            action.ev_commands[ev_id] = -discharge_power

        action.ev2building = power_allocated

        return remaining_power

    def _charge_evs_proportionally(self, action: ElectricalAction,
                                  ev_states: Dict, power_available: float) -> float:
        """Charge connected EVs proportionally."""
        total_headroom = 0
        headroom_per_ev = {}

        for ev_id, ev_info in ev_states.items():
            if ev_info['connected'] and ev_info['soc'] < self.ev_soc_target:
                headroom_energy = (self.ev_soc_target - ev_info['soc']) * ev_info['capacity_kwh'] * 1000
                headroom_power = min(headroom_energy, ev_info['max_charge_rate'])
                headroom_per_ev[ev_id] = headroom_power
                total_headroom += headroom_power

        if total_headroom == 0:
            return power_available

        power_allocated = min(power_available, total_headroom)
        remaining_power = power_available - power_allocated

        for ev_id, headroom_power in headroom_per_ev.items():
            proportion = headroom_power / total_headroom
            charge_power = power_allocated * proportion
            if ev_id not in action.ev_commands:
                action.ev_commands[ev_id] = 0
            action.ev_commands[ev_id] += charge_power

        action.pv2ev += power_allocated

        return remaining_power

    def _charge_batteries_from_grid(self, action: ElectricalAction,
                                   battery_states: Dict, power_available: float) -> float:
        """Charge batteries from grid."""
        remaining = self._charge_batteries_proportionally(action, battery_states, power_available)
        action.grid2battery = power_available - remaining
        return remaining

    def _charge_evs_from_grid(self, action: ElectricalAction,
                             ev_states: Dict, power_available: float) -> float:
        """Charge EVs from grid."""
        remaining = self._charge_evs_proportionally(action, ev_states, power_available)
        action.grid2ev = power_available - remaining
        return remaining

    def _calculate_total_charge_needed(self, component_states: Dict, target_soc: float) -> float:
        """Calculate total charging power needed to reach target SOC."""
        total_needed = 0
        for comp_id, comp_info in component_states.items():
            if comp_info.get('connected', True):  # Default to True for batteries
                soc_deficit = max(0, target_soc - comp_info['soc'])
                energy_needed = soc_deficit * comp_info['capacity_kwh'] * 1000
                power_needed = min(energy_needed, comp_info['max_charge_rate'])
                total_needed += power_needed
        return total_needed

    def _get_average_soc(self, component_states: Dict) -> float:
        """Get average SOC of components."""
        if not component_states:
            return 0.0
        total_soc = sum(comp['soc'] for comp in component_states.values())
        return total_soc / len(component_states)

    def _is_peak_period(self, disturbance: Disturbance) -> bool:
        """Check if current time is peak period."""
        if hasattr(disturbance, 'prices') and hasattr(disturbance.prices, 'peaksignal'):
            return disturbance.prices.peaksignal
        return False

    def _get_grid_status(self, disturbance: Disturbance) -> bool:
        """Check grid connection status."""
        if hasattr(disturbance, 'grid_connected'):
            return disturbance.grid_connected
        return True  # Default to connected

    def _log_control_decision(self, action: ElectricalAction,
                            building_load: float, pv_generation: float) -> None:
        """Log control decision for debugging."""
        self.logger.debug(f"Control decision - Load: {building_load:.1f}W, "
                         f"PV: {pv_generation:.1f}W")
        self.logger.debug(f"Power flows - PV2Building: {action.pv2building:.1f}W, "
                         f"Battery2Building: {action.battery2building:.1f}W, "
                         f"Grid2Building: {action.grid2building:.1f}W")

    def reset(self) -> None:
        """Reset controller to initial state."""
        self.grid_connected = True
        self.logger.debug(f"Reset electrical controller: {self.name}")

    def get_state(self) -> Dict[str, Any]:
        """Get current controller state."""
        return {
            'control_mode': self.control_mode.value,
            'grid_connected': self.grid_connected,
            'num_batteries': len(self.battery_configs),
            'num_evs': len(self.ev_configs),
            'num_pvs': len(self.pv_configs)
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set controller state."""
        if 'control_mode' in state:
            self.control_mode = ControlMode(state['control_mode'])
        if 'grid_connected' in state:
            self.grid_connected = state['grid_connected']