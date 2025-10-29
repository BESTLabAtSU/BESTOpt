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
    DERSystemState, Disturbance, ClusterObservation, DERSystemAction, DomainState,SystemType,
    PVComponentState, BatteryComponentState, EVComponentState, DERMode, ComponentType
)


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
        self.control_mode = DERMode(config.get("mode", "self-consumption"))

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
             state: DomainState,
             # observation: ClusterObservation,
             observation: Any, #@todo the building and system need to be properly managed in observation follow standard format, just use simplfied format for now
             disturbance: Disturbance,
             timestep: float) -> DERSystemAction:
        """
        Determine electrical control action based on current conditions.

        Handles multiple components properly by aggregating states and
        distributing commands proportionally.
        """
        try:
            system_state, building_state = observation
            building_load = self._get_building_load(building_state)/1000 #kw
            pv_generation = self._get_pv_generation(system_state)/1000 #kw
            DER_state = system_state.components

            # Get control signals
            is_peak = self._is_peak_period(disturbance)
            self.grid_connected = self._get_grid_status(disturbance)
            action = DERSystemAction(
                system_id=system_state.system_id,
                system_type=SystemType.DER.value)

            action = self.tou_control(
                building_load, pv_generation, DER_state, is_peak, action
            )

            # # Determine control strategy based on mode
            # if not self.grid_connected or self.control_mode == DERMode.ISLANDED:
            #     action = self._islanded_mode_control(
            #         building_load, pv_generation_total,
            #         battery_states, ev_states, disturbance
            #     )
            # elif self.control_mode == DERMode.TIME_OF_USE:
            #     action = self._time_of_use_control(
            #         building_load, pv_generation_total,
            #         battery_states, ev_states, is_peak, disturbance
            #     )
            # else:  # SELF_CONSUMPTION mode (default)
            #     action = self._self_consumption_control(
            #         building_load, pv_generation_total,
            #         battery_states, ev_states, disturbance
            #     )

            # Log decision
            # self._log_control_decision(action, building_load, pv_generation_total)

            return action

        except Exception as e:
            self.logger.error(f"Error in electrical controller step: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return DERSystemAction(
            system_id=state['der_system_1'].system_id,
            system_type=SystemType.DER.value
        )

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

    def _get_building_load(self, state: Any) -> float:
        building_load = state.components['electrical'].building_power_w
        return building_load  # Default fallback

    def _get_pv_generation(self, state: Any) -> float:
        pv_generation = state.components['pv_1'].generation_w
        return pv_generation


    def _get_all_battery_states(self, state: DERSystemState) -> Dict[str, Dict[str, Any]]:
        """Get states of all batteries."""
        battery_states = {}
        for bat_id, bat_state in state['der_system_1'].components.items():
            if bat_state.component_type == ComponentType.BATTERY:
                bat_config = self.battery_configs.get(bat_id, {})
                battery_states[bat_id] = {
                    'soc': bat_state.soc,
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
                                  disturbance: Disturbance) -> DERSystemAction:
        """
        Self-consumption mode: Maximize use of local generation.
        Priority: PV → Building → Battery → EV → Grid Export
        """
        action = DERSystemAction()

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

    def tou_control(self, building_load, pv_generation, DER_state, is_peak, action):
        """
            DERSystemAction with all power flow decisions
        """
        # Organize components by type
        batteries = {k: v for k, v in DER_state.items() if 'bat' in k.lower()}
        evs = {k: v for k, v in DER_state.items() if 'ev' in k.lower()}

        # Sort components by ID for consistent priority (use first, then another)
        battery_ids = sorted(batteries.keys())
        ev_ids = sorted(evs.keys())

        # Helper function to get available capacity
        def get_capacity(component):
            """
            Calculate available charge/discharge capacity
            C-rate: 0.25 means 25% of capacity per hour (4 hours to full charge)
            """
            capacity_kwh = component.capacity_kwh
            current_kwh = component.soc * capacity_kwh

            # C-rate to kW: C-rate * capacity_kwh = kW
            # e.g., 0.25C * 10kWh = 2.5kW charging power
            max_charge_rate = component.charge_speed * capacity_kwh  # C-rate * capacity
            max_discharge_rate = component.discharge_speed * capacity_kwh  # C-rate * capacity

            # Available energy to charge/discharge
            energy_to_full = capacity_kwh - current_kwh
            energy_available = current_kwh

            # Actual capacity limited by both rate and available energy
            max_charge = min(max_charge_rate, energy_to_full)
            max_discharge = min(max_discharge_rate, energy_available)

            return max_charge, max_discharge

        # Step 1: PV to building first (always prioritize self-consumption)
        remaining_pv = pv_generation
        remaining_load = building_load

        if remaining_pv > 0 and remaining_load > 0:
            pv_to_building = min(remaining_pv, remaining_load)
            action.pv2building = pv_to_building
            remaining_pv -= pv_to_building
            remaining_load -= pv_to_building

        # Step 2: Handle based on price period
        if not is_peak:
            # OFF-PEAK: Charge batteries and EVs

            # First use excess PV for charging (batteries first)
            for bat_id in battery_ids:
                if remaining_pv <= 0:
                    break
                bat = batteries[bat_id]
                max_charge, _ = get_capacity(bat)
                if max_charge > 0:
                    charge_power = min(max_charge, remaining_pv)
                    action.pv2battery[bat_id] = charge_power
                    remaining_pv -= charge_power

            # Then EVs with excess PV
            for ev_id in ev_ids:
                if remaining_pv <= 0:
                    break
                ev = evs[ev_id]
                if ev.is_active:
                    max_charge, _ = get_capacity(ev)
                    if max_charge > 0:
                        charge_power = min(max_charge, remaining_pv)
                        action.pv2ev[ev_id] = charge_power
                        remaining_pv -= charge_power

            # Charge from grid at C-rate speed (batteries first)
            for bat_id in battery_ids:
                bat = batteries[bat_id]
                max_charge, _ = get_capacity(bat)
                if max_charge > 0:
                    # Use C-rate charging speed from grid
                    charge_power = min(max_charge, bat.charge_speed * bat.capacity_kwh)
                    action.grid2battery[bat_id] = charge_power

            # Then charge EVs from grid
            for ev_id in ev_ids:
                ev = evs[ev_id]
                if ev.is_active:
                    max_charge, _ = get_capacity(ev)
                    if max_charge > 0:
                        charge_power = min(max_charge, ev.charge_speed * ev.capacity_kwh)
                        action.grid2ev[ev_id] = charge_power

            # Send excess PV to grid if any
            if remaining_pv > 0:
                action.pv2grid = remaining_pv

        else:
            # PEAK: Discharge batteries and EVs to support building

            # Discharge batteries first
            for bat_id in battery_ids:
                if remaining_load <= 0:
                    break
                bat = batteries[bat_id]
                _, max_discharge = get_capacity(bat)
                if max_discharge > 0:
                    discharge_power = min(max_discharge, remaining_load)
                    action.battery2building[bat_id] = discharge_power
                    remaining_load -= discharge_power

            # Then discharge EVs if needed
            for ev_id in ev_ids:
                if remaining_load <= 0:
                    break
                ev = evs[ev_id]
                if ev.is_active:
                    _, max_discharge = get_capacity(ev)
                    if max_discharge > 0:
                        discharge_power = min(max_discharge, remaining_load)
                        action.ev2building[ev_id] = discharge_power
                        remaining_load -= discharge_power

            # Send excess PV to grid (don't charge during peak)
            if remaining_pv > 0:
                action.pv2grid = remaining_pv

        # Step 3: Cover remaining building load from grid
        if remaining_load > 0:
            action.grid2building = remaining_load

        return action


    def _time_of_use_control(self, building_load: float, pv_generation: float,
                             battery_states: Dict, ev_states: Dict,
                             is_peak: bool, disturbance: Disturbance) -> DERSystemAction:
        """
        Time-of-use mode: Optimize based on peak/off-peak periods.
        Peak: Minimize grid import, use storage
        Off-peak: Charge storage from grid
        """
        action = DERSystemAction()

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
                               disturbance: Disturbance) -> DERSystemAction:
        """
        Islanded mode: No grid connection, careful resource management.
        Priority: Critical loads only, preserve battery reserve
        """
        action = DERSystemAction()

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

    def _discharge_batteries_proportionally(self, action: DERSystemAction,
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

    def _charge_batteries_proportionally(self, action: DERSystemAction,
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

    def _discharge_evs_proportionally(self, action: DERSystemAction,
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

    def _charge_evs_proportionally(self, action: DERSystemAction,
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

    def _charge_batteries_from_grid(self, action: DERSystemAction,
                                   battery_states: Dict, power_available: float) -> float:
        """Charge batteries from grid."""
        remaining = self._charge_batteries_proportionally(action, battery_states, power_available)
        action.grid2battery = power_available - remaining
        return remaining

    def _charge_evs_from_grid(self, action: DERSystemAction,
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

    def _log_control_decision(self, action: DERSystemAction,
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
            self.control_mode = DERMode(state['control_mode'])
        if 'grid_connected' in state:
            self.grid_connected = state['grid_connected']