"""
Electrical Supervisory Controller
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from enum import Enum

from ..core.base import BaseModule
from ..core.data_structure import (
    State, ElectricalAction, Disturbance, Observation,
    ElectricalDomainState, BatteryState, PVState, EVState
)

class SupervisoryController(BaseModule):
    """
    Rule-based electrical supervisory controller for DER power flow management.

    Features:
    - Flexibility Mode: Time-of-use based charging/discharging strategy
    - Resilience Mode: Grid-connected or islanded operation
    - Manages power flow between PV, Battery, EV, Building, and Grid
    - Handles peak/off-peak periods and grid disconnection events
    """

    def __init__(self, config: Dict[str, Any], name: str = "SupervisoryController"):
        """
        Initialize Electrical Supervisory Controller.

        Args:
            config: Controller configuration parameters
            name: Module name
        """
        super().__init__(config, name)

        # Domain configuration
        self.domain = config.get("domain", "electrical")
        self.system_config = config.get("system_config")

        # Battery SOC thresholds
        self.bat_soc_min = config.get("bat_soc_min", 0.1)  # Minimum SOC
        self.bat_soc_max = config.get("bat_soc_max", 0.9)  # Maximum SOC
        self.bat_soc_reserve = config.get("bat_soc_reserve", 0.2)  # Reserve for resilience

        # EV SOC thresholds
        self.ev_soc_min = config.get("ev_soc_min", 0.2)  # Minimum SOC for EV
        self.ev_soc_target = config.get("ev_soc_target", 0.8)  # Target SOC when plugged in
        self.ev_v2g_enabled = config.get("ev_v2g_enabled", True)  # Vehicle-to-grid enabled

        # Power limits
        self.max_grid_import = config.get("max_grid_import", 10000)  # Watts
        self.max_grid_export = config.get("max_grid_export", 5000)  # Watts

        # Charging/discharging rates
        self.bat_charge_rate = config.get("bat_charge_rate", 2000)  # Watts (fixed speed, @TODo replaced by X% C later)
        self.bat_discharge_rate = config.get("bat_discharge_rate", 2000)  # Watts
        self.ev_charge_rate = config.get("ev_charge_rate", 3000)  # Watts
        self.ev_discharge_rate = config.get("ev_discharge_rate", 2000)  # Watts

        # Grid status (for resilience mode)
        self.grid_connected = True  # Assume grid is connected initially

        # Detect system components
        self._detect_components()

        self.logger = logging.getLogger(f"{__name__}.{name}")

    def initialize(self) -> None:
        """Initialize the controller."""
        self.logger.info(f"Initialized electrical controller: {self.name}")

    def step(self,
             state: Any,
             observation: Observation,
             disturbance: Disturbance,
             timestep: float) -> ElectricalAction:
        """
        Determine electrical control action based on current conditions.

        Args:
            state: Current electrical domain state
            observation: Current observations
            disturbance: Current disturbances
            timestep: Current simulation timestep

        Returns:
            ElectricalAction with power flow commands
        """
        try:
            # Extract current system state
            building_load = 10 # self._get_building_load(state)
            pv_generation = self._get_pv_generation(state, disturbance)
            bat_soc = self._get_battery_soc(state)
            ev_soc = self._get_ev_soc(state)
            ev_connected = self._is_ev_connected(disturbance)
            is_peak = self._is_peak_period(disturbance)
            self.grid_connected = self._get_grid_status(disturbance)

            if not self.grid_connected:
                # Islanded mode - manage local resources only
                power_flows = self._resilience_islanded_logic(
                    building_load, pv_generation,
                    bat_soc, ev_soc, ev_connected
                )
            else:
                # Grid connected - use flexibility mode logic
                power_flows = self._flexibility_mode_logic(
                    building_load, pv_generation,
                    bat_soc, ev_soc, ev_connected,
                    is_peak
                )

            # Create action from power flows
            action = self._create_action(power_flows)

            # Log decision
            self._log_decision(power_flows, building_load, pv_generation, is_peak)

            return action

        except Exception as e:
            self.logger.error(f"Error in electrical controller step: {e}")
            # Return safe default action
            return ElectricalAction()

    def _flexibility_mode_logic(self, building_load: float, pv_generation: float,
                               bat_soc: float, ev_soc: float, ev_connected: bool,
                               is_peak: bool) -> Dict[str, float]:
        """
        Flexibility Mode: Time-of-use based strategy.

        Off-peak: Fixed speed charging (EV priority)
        Peak: Discharge storage (Battery first)
        """
        flows = self._initialize_power_flows()

        # First, always use PV for building load
        pv_to_building = min(pv_generation, building_load)
        flows['pv2building'] = pv_to_building
        remaining_pv = pv_generation - pv_to_building
        remaining_load = building_load - pv_to_building

        if is_peak:
            # PEAK PERIOD: Minimize grid import, use storage

            # Step 1: Use battery first for remaining load (battery priority during discharge)
            if self.has_battery and bat_soc > self.bat_soc_min and remaining_load > 0:
                bat_discharge = min(
                    remaining_load,
                    self.bat_discharge_rate,
                    self._calculate_max_discharge(bat_soc, self.bat_soc_min)
                )
                flows['battery2building'] = bat_discharge
                remaining_load -= bat_discharge

            # Step 2: Use EV V2G if battery insufficient
            if (self.has_ev and ev_connected and self.ev_v2g_enabled and
                ev_soc > self.ev_soc_min and remaining_load > 0):
                ev_discharge = min(
                    remaining_load,
                    self.ev_discharge_rate,
                    self._calculate_max_discharge(ev_soc, self.ev_soc_min)
                )
                flows['ev2building'] = ev_discharge
                remaining_load -= ev_discharge

            # Step 3: Use remaining PV for charging if any excess
            if remaining_pv > 0:
                # Charge battery with excess PV
                if self.has_battery and bat_soc < self.bat_soc_max:
                    bat_charge_from_pv = min(
                        remaining_pv,
                        self.bat_charge_rate,
                        self._calculate_max_charge(bat_soc, self.bat_soc_max)
                    )
                    flows['pv2battery'] = bat_charge_from_pv
                    remaining_pv -= bat_charge_from_pv

                # Charge EV with remaining PV
                if self.has_ev and ev_connected and ev_soc < self.ev_soc_target and remaining_pv > 0:
                    ev_charge_from_pv = min(
                        remaining_pv,
                        self.ev_charge_rate,
                        self._calculate_max_charge(ev_soc, self.ev_soc_target)
                    )
                    flows['pv2ev'] = ev_charge_from_pv
                    remaining_pv -= ev_charge_from_pv

                # Export excess PV to grid
                if remaining_pv > 0:
                    flows['pv2grid'] = min(remaining_pv, self.max_grid_export)

            # Step 4: Import from grid only if absolutely necessary
            if remaining_load > 0:
                flows['grid2building'] = min(remaining_load, self.max_grid_import)

        else:
            # OFF-PEAK PERIOD: Fixed speed charging, EV priority

            # Step 1: Supply remaining building load from grid
            if remaining_load > 0:
                flows['grid2building'] = remaining_load

            # Step 2: Charge EV first (higher priority during off-peak)
            if self.has_ev and ev_connected and ev_soc < self.ev_soc_target:
                ev_charge_needed = self._calculate_max_charge(ev_soc, self.ev_soc_target)

                # Use PV first for EV charging
                if remaining_pv > 0:
                    ev_charge_from_pv = min(remaining_pv, self.ev_charge_rate, ev_charge_needed)
                    flows['pv2ev'] = ev_charge_from_pv
                    remaining_pv -= ev_charge_from_pv
                    ev_charge_needed -= ev_charge_from_pv

                # Use grid for remaining EV charging need (fixed speed)
                if ev_charge_needed > 0:
                    ev_charge_from_grid = min(self.ev_charge_rate, ev_charge_needed)
                    flows['grid2ev'] = ev_charge_from_grid

            # Step 3: Charge battery second
            if self.has_battery and bat_soc < self.bat_soc_max:
                bat_charge_needed = self._calculate_max_charge(bat_soc, self.bat_soc_max)

                # Use remaining PV for battery charging
                if remaining_pv > 0:
                    bat_charge_from_pv = min(remaining_pv, self.bat_charge_rate, bat_charge_needed)
                    flows['pv2battery'] = bat_charge_from_pv
                    remaining_pv -= bat_charge_from_pv
                    bat_charge_needed -= bat_charge_from_pv

                # Use grid for remaining battery charging need (fixed speed)
                if bat_charge_needed > 0:
                    bat_charge_from_grid = min(self.bat_charge_rate, bat_charge_needed)
                    flows['grid2battery'] = bat_charge_from_grid

            # Step 4: Export any remaining PV to grid
            if remaining_pv > 0:
                flows['pv2grid'] = min(remaining_pv, self.max_grid_export)

        return flows

    def _resilience_islanded_logic(self, building_load: float, pv_generation: float,
                                   bat_soc: float, ev_soc: float,
                                   ev_connected: bool) -> Dict[str, float]:
        """
        Resilience Mode - Islanded: No grid connection, manage local resources.

        Priority:
        1. Use PV for building
        2. Use battery for deficit
        3. Use EV V2G if critical
        4. Store excess PV in battery/EV
        5. Curtail if necessary
        """
        flows = self._initialize_power_flows()

        # No grid flows in islanded mode
        flows['grid2building'] = 0
        flows['grid2battery'] = 0
        flows['grid2ev'] = 0
        flows['pv2grid'] = 0
        flows['battery2grid'] = 0
        flows['ev2grid'] = 0

        # Step 1: Use PV for building load
        pv_to_building = min(pv_generation, building_load)
        flows['pv2building'] = pv_to_building
        remaining_pv = pv_generation - pv_to_building
        remaining_load = building_load - pv_to_building

        # Step 2: Use battery for remaining load
        if self.has_battery and bat_soc > self.bat_soc_reserve and remaining_load > 0:
            # Keep some reserve for critical loads
            usable_soc = bat_soc - self.bat_soc_reserve
            if usable_soc > 0:
                bat_discharge = min(
                    remaining_load,
                    self.bat_discharge_rate,
                    self._calculate_max_discharge(bat_soc, self.bat_soc_reserve)
                )
                flows['battery2building'] = bat_discharge
                remaining_load -= bat_discharge

        # Step 3: Use EV V2G only if critical (battery depleted)
        if (self.has_ev and ev_connected and self.ev_v2g_enabled and
            ev_soc > self.ev_soc_min + 0.1 and remaining_load > 0):
            # Keep higher reserve for EV in islanded mode
            ev_discharge = min(
                remaining_load,
                self.ev_discharge_rate,
                self._calculate_max_discharge(ev_soc, self.ev_soc_min + 0.1)
            )
            flows['ev2building'] = ev_discharge
            remaining_load -= ev_discharge

        # Step 4: Store excess PV (if any)
        if remaining_pv > 0:
            # Charge battery first
            if self.has_battery and bat_soc < self.bat_soc_max:
                bat_charge = min(
                    remaining_pv,
                    self.bat_charge_rate,
                    self._calculate_max_charge(bat_soc, self.bat_soc_max)
                )
                flows['pv2battery'] = bat_charge
                remaining_pv -= bat_charge

            # Charge EV with remaining PV
            if self.has_ev and ev_connected and ev_soc < self.ev_soc_target and remaining_pv > 0:
                ev_charge = min(
                    remaining_pv,
                    self.ev_charge_rate,
                    self._calculate_max_charge(ev_soc, self.ev_soc_target)
                )
                flows['pv2ev'] = ev_charge
                remaining_pv -= ev_charge

            # Note: Any remaining PV would be curtailed in islanded mode
            if remaining_pv > 0:
                self.logger.debug(f"Curtailing {remaining_pv:.1f}W of PV generation in islanded mode")

        # Log if load cannot be met
        if remaining_load > 0:
            self.logger.warning(f"Cannot meet {remaining_load:.1f}W of load in islanded mode")

        return flows

    def _detect_components(self) -> None:
        """Detect which components are available in the system."""
        self.has_pv = self.system_config.get("pv") is not None
        self.has_battery = self.system_config.get("bat") is not None
        self.has_ev = self.system_config.get("ev") is not None

    def _get_building_load(self, state: ElectricalDomainState) -> float:
        """Get total building electrical load in Watts."""
        return state.bldg_e_loads

    def _get_pv_generation(self, state: ElectricalDomainState,
                          disturbance: Disturbance) -> float:
        """Get current PV generation in Watts."""
        if not self.has_pv:
            return 0.0
        return state.total_generation

    def _get_battery_soc(self, state: ElectricalDomainState) -> float:
        """Get battery state of charge (0-1)."""
        if not self.has_battery:
            return 0.0
        # Get first battery's SOC
        for bat_id, bat_state in state.batteries.items():
            return bat_state.battery_soc
        return 0.0

    def _get_ev_soc(self, state: ElectricalDomainState) -> float:
        """Get EV state of charge (0-1)."""
        if not self.has_ev:
            return 0.0
        for ev_id, ev_state in state.evs.items():
            return ev_state.ev_soc
        return 0.0

    def _is_ev_connected(self, disturbance: Disturbance) -> bool:
        """Check if EV is connected to charger."""
        if not self.has_ev:
            return False
        return disturbance.occupancy._is_ev_connected

    def _is_peak_period(self, disturbance: Disturbance) -> bool:
        """Check if current time is peak period."""
        return disturbance.prices.peaksignal

    def _get_grid_status(self, disturbance: Disturbance) -> bool:
        """
        Check grid connection status.
        """
        # @TODO add later from disturbance
        if hasattr(disturbance, 'grid_connected'):
            return disturbance.grid_connected
        # Default to connected if no signal
        return True

    def _calculate_max_charge(self, current_soc: float, target_soc: float) -> float:
        """
        Calculate maximum charging power based on SOC.
        Returns a large value for now (actual limits handled by component models).
        """
        if current_soc >= target_soc:
            return 0.0
        # Could implement SOC-based charging curves here
        return 10000.0  # Large value, actual limit enforced elsewhere

    def _calculate_max_discharge(self, current_soc: float, min_soc: float) -> float:
        """
        Calculate maximum discharging power based on SOC.
        Returns a large value for now (actual limits handled by component models).
        """
        if current_soc <= min_soc:
            return 0.0
        # Could implement SOC-based discharging curves here
        return 10000.0  # Large value, actual limit enforced elsewhere

    def _initialize_power_flows(self) -> Dict[str, float]:
        """Initialize all power flows to zero."""
        return {
            'pv2building': 0.0,
            'pv2battery': 0.0,
            'pv2ev': 0.0,
            'pv2grid': 0.0,
            'battery2building': 0.0,
            'battery2ev': 0.0,
            'battery2grid': 0.0,
            'ev2building': 0.0,
            'ev2grid': 0.0,
            'grid2building': 0.0,
            'grid2battery': 0.0,
            'grid2ev': 0.0
        }

    def _create_action(self, power_flows: Dict[str, float]) -> ElectricalAction:
        """Create ElectricalAction from power flow dictionary."""
        action = ElectricalAction()

        # Set all power flows
        for flow_name, power in power_flows.items():
            if hasattr(action, flow_name):
                setattr(action, flow_name, power)

        return action

    def _log_decision(self, power_flows: Dict[str, float],
                     building_load: float, pv_generation: float,
                     is_peak: bool) -> None:
        pass

    def set_grid_status(self, connected: bool) -> None:
        """
        Manually set grid connection status.

        Args:
            connected: True if grid is connected, False if islanded
        """
        self.grid_connected = connected
        self.logger.info(f"Grid status set to: {'CONNECTED' if connected else 'DISCONNECTED'}")

    def set_control_mode(self, mode: str) -> None:
        pass

    def reset(self) -> None:
        """Reset controller to initial state."""
        self.grid_connected = True
        self.logger.debug(f"Reset electrical controller: {self.name}")

    def get_state(self) -> Dict[str, Any]:
        """Get current controller state."""
        return {
            'control_mode': self.control_mode.value,
            'grid_connected': self.grid_connected
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        pass
