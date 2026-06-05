"""
Simplified Electrical Supervisory Controller with realistic power flow model.

Key changes:
- Power is treated as net balance (generation vs demand)
- Only battery charge/discharge is controllable
- EV charging is automatic when connected (not controllable)
- No distinction of power source (grid vs PV) - just net flow
"""

import numpy as np
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, field

from ..core.base import BaseModule
from ..core.data_structure import (
    DERSystemState, Disturbance, DERSystemAction, DomainState, SystemType,
    DERMode, ComponentType
)


@dataclass
class PowerBalance:
    """Represents the power balance at each timestep."""
    total_demand: float = 0.0      # Total load (building + EV charging) [kW]
    building_load: float = 0.0     # Building electrical load [kW]
    ev_charging: float = 0.0       # EV charging power [kW]
    pv_generation: float = 0.0     # PV generation [kW]
    battery_power: float = 0.0     # Battery power (+charge, -discharge) [kW]
    net_grid: float = 0.0          # Net grid power (+import, -export/curtail) [kW]
    curtailment: float = 0.0       # Curtailed PV power [kW]


class SupervisoryController(BaseModule):
    """
    Simplified rule-based electrical supervisory controller.

    Power Flow Model:
    - Demand side: building_load + ev_charging
    - Generation side: pv_generation
    - Controllable: battery_power (charge/discharge)
    - Result: net_grid = demand + battery_power - generation

    Control Strategy (TOU-based):
    - Off-peak: Charge battery from grid at max rate
    - Peak: Discharge battery to reduce grid import
    - Excess PV: Curtailed (no grid export allowed)
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
        self.ev_soc_max = config.get("ev_soc_max", 0.9)

        # Grid constraint (no export allowed)
        self.max_grid_import = config.get("max_grid_import", 20000) / 1000  # kW
        self.allow_grid_export = config.get("allow_grid_export", False)

        # Default rates (will be overridden by component configs)
        self.default_bat_charge_rate = config.get("bat_charge_rate", 2000) / 1000  # kW
        self.default_bat_discharge_rate = config.get("bat_discharge_rate", 2000) / 1000  # kW
        self.default_ev_charge_rate = config.get("ev_charge_rate", 7000) / 1000  # kW

        # Timestep duration
        self.timestep_hours = config.get("timestep_hours", 0.25)  # 15 minutes

        # Grid status
        self.grid_connected = True

        # Component detection
        self._detect_and_configure_components()

        self.logger = logging.getLogger(f"{__name__}.{name}")

    def initialize(self) -> None:
        """Initialize the controller."""
        self.logger.info(f"Initialized electrical controller: {self.name}")
        self.logger.info(f"Components - Batteries: {len(self.battery_configs)}, "
                        f"EVs: {len(self.ev_configs)}, PVs: {len(self.pv_configs)}")

    def step(self,
             state,
             observation,
             disturbance,
             timestep,
             external_action=None):
        """
        Determine electrical control action based on current conditions.

        If external_action is provided (e.g., from RL agent) and contains
        battery_power commands, those override the rule-based battery logic.
        EV charging remains automatic.

        CHANGES:
        - Added external_action parameter (was in signature but unused)
        - External battery_power dict is forwarded to _calculate_power_balance
        """
        try:
            system_state, building_state = observation

            building_load = self._get_building_load(building_state) / 1000
            pv_generation = self._get_pv_generation(system_state) / 1000
            der_components = system_state.components

            is_peak = self._is_peak_period(disturbance)
            self.grid_connected = self._get_grid_status(disturbance)

            action = DERSystemAction(
                system_id=system_state.system_id,
                system_type=SystemType.DER.value
            )

            # >>> NEW: extract external battery command if present <<<
            external_battery_power = None
            if external_action is not None:
                ext_bat = getattr(external_action, 'battery_power', None)
                if ext_bat and isinstance(ext_bat, dict) and len(ext_bat) > 0:
                    external_battery_power = ext_bat
                    self.logger.debug(f"Using external battery commands: {ext_bat}")

            power_balance = self._calculate_power_balance(
                building_load=building_load,
                pv_generation=pv_generation,
                der_components=der_components,
                is_peak=is_peak,
                action=action,
                external_battery_power=external_battery_power,  # <<< NEW
            )

            action.power_balance = power_balance
            return action

        except Exception as e:
            self.logger.error(f"Error in electrical controller step: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return DERSystemAction(
                system_id=state.system_id if hasattr(state, 'system_id') else 'der_system_1',
                system_type=SystemType.DER.value
            )

    def _calculate_power_balance(self,
                                 building_load,
                                 pv_generation,
                                 der_components,
                                 is_peak,
                                 action,
                                 external_battery_power=None):
        """
        Calculate power balance and determine battery control.

        CHANGES:
        - New parameter external_battery_power: Dict[str, float] or None
          If provided, battery commands come from the RL agent instead of
          the rule-based strategy.
        """
        balance = PowerBalance()
        balance.building_load = building_load
        balance.pv_generation = pv_generation

        batteries = {k: v for k, v in der_components.items() if 'bat' in k.lower()}
        evs = {k: v for k, v in der_components.items() if 'ev' in k.lower()}

        # Step 1: EV charging (automatic, uncontrollable)
        total_ev_charging = 0.0
        for ev_id, ev in evs.items():
            if getattr(ev, "is_connected", False) and getattr(ev, "is_active", True):
                ev_charge_power = self._get_ev_charge_power(ev)
                action.ev_charging[ev_id] = ev_charge_power
                total_ev_charging += ev_charge_power

        balance.ev_charging = total_ev_charging
        balance.total_demand = building_load + total_ev_charging

        # Step 2: net power before battery
        net_power = balance.total_demand - pv_generation

        # Step 3: Battery control
        if external_battery_power is not None:
            # >>> RL AGENT MODE: use external commands <<<
            battery_power = self._apply_external_battery(
                batteries=batteries,
                external_commands=external_battery_power,
                action=action
            )
        else:
            # >>> RULE-BASED MODE: original TOU logic <<<
            battery_power = self._apply_internal_battery(
                batteries=batteries,
                net_power=net_power,
                is_peak=is_peak,
                action=action
            )

        balance.battery_power = battery_power

        # Step 4: grid power
        final_net = balance.total_demand + battery_power - pv_generation

        if final_net > self.max_grid_import:
            excess = final_net - self.max_grid_import

            # First: curtail battery charging
            if battery_power > 0:
                reduction = min(battery_power, excess)
                battery_power -= reduction
                excess -= reduction
                # Update action proportionally across batteries
                if reduction > 0:
                    for bat_id in action.battery_power:
                        if action.battery_power[bat_id] > 0:
                            bat_reduction = min(action.battery_power[bat_id], reduction)
                            action.battery_power[bat_id] -= bat_reduction
                            reduction -= bat_reduction
                            if reduction <= 0:
                                break

            # Second: curtail EV charging if still over limit
            if excess > 0 and total_ev_charging > 0:
                ev_reduction = min(total_ev_charging, excess)
                excess -= ev_reduction
                # Update EV actions proportionally
                remaining_reduction = ev_reduction
                for ev_id in action.ev_charging:
                    if action.ev_charging[ev_id] > 0 and remaining_reduction > 0:
                        cut = min(action.ev_charging[ev_id], remaining_reduction)
                        action.ev_charging[ev_id] -= cut
                        remaining_reduction -= cut
                # Recalculate EV total
                total_ev_charging -= ev_reduction
                balance.ev_charging = total_ev_charging
                balance.total_demand = building_load + total_ev_charging

            balance.battery_power = battery_power

        # Recalculate final net after all curtailments
        final_net = balance.total_demand + balance.battery_power - pv_generation

        if final_net >= 0:
            balance.net_grid = min(final_net, self.max_grid_import)
            balance.curtailment = 0.0
        else:
            if self.allow_grid_export:
                balance.net_grid = final_net
                balance.curtailment = 0.0
            else:
                balance.net_grid = 0.0
                balance.curtailment = -final_net

        action.grid_import = balance.net_grid
        action.curtailment = balance.curtailment

        return balance

    def _get_ev_charge_power(self, ev) -> float:
        """
        Calculate EV charging power (automatic charging when connected).
        Charges at max rate until SOC limit, within constraints.
        """
        soc = float(getattr(ev, "soc", 0.5))
        capacity_kwh = float(getattr(ev, "capacity_kwh", 40.0))
        charge_c = float(getattr(ev, "charge_speed", 0.5))

        # Check if can charge
        if soc >= self.ev_soc_max:
            return 0.0

        # Max charge rate from C-rate
        max_charge_kw = charge_c * capacity_kwh

        # Energy headroom to max SOC
        energy_to_max = (self.ev_soc_max - soc) * capacity_kwh
        max_from_energy = energy_to_max / self.timestep_hours

        # Use minimum of rate limit and energy headroom
        charge_power = min(max_charge_kw, max_from_energy, self.default_ev_charge_rate)

        return max(0.0, charge_power)

    def _apply_external_battery(self, batteries, external_commands, action):
        """
        Apply external (RL) battery power commands with safety clamping.

        Args:
            batteries: Dict of battery component states
            external_commands: Dict[battery_id, power_kw]  (+charge, -discharge)
            action: DERSystemAction to fill

        Returns:
            total battery power (kW)
        """
        total_battery_power = 0.0

        for bat_id in sorted(batteries.keys()):
            bat = batteries[bat_id]
            max_charge_kw, max_discharge_kw = self._get_battery_capacity(bat)

            # Get external command (default 0 if not specified)
            commanded_power = external_commands.get(bat_id, 0.0)

            # Clamp to physical limits
            if commanded_power >= 0:
                # Charging
                actual_power = min(commanded_power, max_charge_kw)
            else:
                # Discharging
                actual_power = max(commanded_power, -max_discharge_kw)

            action.battery_power[bat_id] = actual_power
            total_battery_power += actual_power

        return total_battery_power

    def _apply_internal_battery(self,
                         batteries: Dict,
                         net_power: float,
                         is_peak: bool,
                         action: DERSystemAction) -> float:
        """
        Control battery based on TOU strategy.

        Returns: total battery power (positive=charging, negative=discharging)

        Strategy:
        - Off-peak: Charge at max rate (to prepare for peak)
        - Peak: Discharge to offset demand (reduce grid import)
        - Always: Absorb excess PV when possible
        """
        total_battery_power = 0.0

        for bat_id in sorted(batteries.keys()):
            bat = batteries[bat_id]
            max_charge_kw, max_discharge_kw = self._get_battery_capacity(bat)

            if is_peak:
                # Peak: Discharge to reduce grid import
                if net_power > 0:  # There's demand to meet
                    discharge = min(max_discharge_kw, net_power)
                    action.battery_power[bat_id] = -discharge  # negative = discharge
                    total_battery_power -= discharge
                    net_power -= discharge  # Update remaining demand
                else:
                    # Net negative (excess PV) - charge with excess
                    charge = min(max_charge_kw, -net_power)
                    action.battery_power[bat_id] = charge
                    total_battery_power += charge
                    net_power += charge
            else:
                # Off-peak: Charge at max rate
                if net_power < 0:
                    # Excess PV available - use it first
                    charge = min(max_charge_kw, -net_power)
                    action.battery_power[bat_id] = charge
                    total_battery_power += charge
                else:
                    # No excess PV - charge from grid at max rate
                    action.battery_power[bat_id] = max_charge_kw
                    total_battery_power += max_charge_kw

        return total_battery_power

    def _get_battery_capacity(self, battery) -> tuple:
        """
        Get battery charge/discharge capacity for this timestep.
        Returns: (max_charge_kw, max_discharge_kw)
        """
        capacity_kwh = float(getattr(battery, "capacity_kwh", 5.0))
        soc = float(getattr(battery, "soc", 0.5))
        charge_c = float(getattr(battery, "charge_speed", 0.5))
        discharge_c = float(getattr(battery, "discharge_speed", 0.5))

        # Rate limits from C-rate
        rate_charge_kw = charge_c * capacity_kwh
        rate_discharge_kw = discharge_c * capacity_kwh

        # Energy limits based on SOC bounds
        energy_to_max = max(0.0, (self.bat_soc_max - soc) * capacity_kwh)
        energy_above_min = max(0.0, (soc - self.bat_soc_min) * capacity_kwh)

        # Convert to power (kW) for this timestep
        max_charge_kw = min(rate_charge_kw, energy_to_max / self.timestep_hours)
        max_discharge_kw = min(rate_discharge_kw, energy_above_min / self.timestep_hours)

        return max_charge_kw, max_discharge_kw

    def _detect_and_configure_components(self) -> None:
        """Detect and configure all components in the system."""
        self.pv_configs = self._parse_component_config('pv_systems', 'pv')
        self.has_pv = len(self.pv_configs) > 0

        self.battery_configs = self._parse_component_config('batteries', 'bat')
        self.has_battery = len(self.battery_configs) > 0

        self.ev_configs = self._parse_component_config('evs', 'ev')
        self.has_ev = len(self.ev_configs) > 0

    def _parse_component_config(self, plural_key: str, singular_key: str) -> Dict:
        """Parse component configuration."""
        configs = {}
        if plural_key in self.system_config:
            multi_config = self.system_config[plural_key]
            if isinstance(multi_config, list):
                for idx, comp_config in enumerate(multi_config):
                    comp_id = comp_config.get('id', f"{singular_key}_{idx + 1}")
                    configs[comp_id] = comp_config
            elif isinstance(multi_config, dict):
                configs = multi_config
        elif singular_key in self.system_config:
            single_config = self.system_config[singular_key]
            comp_id = single_config.get('id', f"{singular_key}_1")
            configs[comp_id] = single_config
        return configs

    def _get_building_load(self, state: Any) -> float:
        return state.components['electrical'].total_load_w

    def _get_pv_generation(self, state: Any) -> float:
        """Get total PV generation, returning 0 if no PV is installed."""
        total_pv = 0.0
        for comp_id, comp in state.components.items():
            if 'pv' in comp_id.lower() and hasattr(comp, 'generation_w'):
                total_pv += comp.generation_w
        return total_pv

    def _is_peak_period(self, disturbance: Disturbance) -> bool:
        if hasattr(disturbance, 'prices') and hasattr(disturbance.prices, 'peaksignal'):
            return disturbance.prices.peaksignal
        return False

    def _get_grid_status(self, disturbance: Disturbance) -> bool:
        if hasattr(disturbance, 'grid_connected'):
            return disturbance.grid_connected
        return True

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