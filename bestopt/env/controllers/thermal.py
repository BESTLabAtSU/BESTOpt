"""
Rule-based thermal controller for HVAC systems.
"""

import numpy as np
from typing import Dict, Any, Optional
from ..core.base import BaseModule
from ..core.data_structure import State, Action, Disturbance, Observation, ThermalAction, HVACMode


class RuleBased(BaseModule):
    """
    Rule-based thermal controller.
    """

    def __init__(self, config: Dict[str, Any], name: str = "RuleBased"):
        """Initialize rule-based thermal controller."""
        super().__init__(config, name)

        # Control parameters
        self.domain = config.get("domain", "thermal")
        self.mode = HVACMode(config.get("mode", "auto"))

        # Comfort settings
        self.setpoint_cooling = config.get("setpoint_cooling", 24.0)  # °C
        self.setpoint_heating = config.get("setpoint_heating", 21.0)  # °C
        self.deadband = config.get("deadband", 0.5)  # °C

        # HVAC capacity settings
        self.cooling_power_max = config.get("cooling_power_max", 4000.0)  # W
        self.heating_power_max = config.get("heating_power_max", 4000.0)  # W
        self.stage1_power = config.get("stage1_power", 2000.0)  # W (50% capacity)
        self.stage2_power = config.get("stage2_power", 4000.0)  # W (100% capacity)

        # Control states
        self.current_hvac_power = 0.0
        self.last_mode = HVACMode.OFF

    def initialize(self) -> None:
        """Initialize the controller."""
        self.current_hvac_power = 0.0
        self.last_mode = HVACMode.OFF
        self.hvac_is_on = False
        self.logger.info(f"Initialized thermal controller: {self.name}")

    def step(self, state: Any, observation: Any, disturbance: Disturbance,
             timestep: float) -> ThermalAction:
        """
        Determine thermal control action based on current conditions.

        Args:
            state: Current thermal domain state (or full state)
            action: Not used
            disturbance: Current disturbances
            timestep: Current simulation timestep

        Returns:
            ThermalAction with control commands
        """
        try:
            # Extract temperature from state
            # Handle both domain state and building state
            if hasattr(state, 'thermal_zones') and state.thermal_zones:
                # Get temperature from first thermal zone
                zone_id = list(state.thermal_zones.keys())[0]
                current_temp = state.thermal_zones[zone_id].temperature
            elif hasattr(state, 'temperature'):
                current_temp = state.temperature
            else:
                self.logger.warning("Could not extract temperature from state")
                current_temp = 22.0  # Default

            # Get occupancy information
            occupancy = disturbance.occupancy

            # Determine active setpoints based on occupancy
            cooling_setpoint, heating_setpoint = self._get_active_setpoints(
                occupancy, disturbance
            )

            # Calculate control action
            hvac_power = self._calculate_hvac_power(
                current_temp=current_temp,
                cooling_setpoint=cooling_setpoint,
                heating_setpoint=heating_setpoint,
                timestep=timestep
            )

            # Update internal state
            self.current_hvac_power = hvac_power

            # Create thermal action
            thermal_action = ThermalAction()
            thermal_action.hvac_power = hvac_power
            thermal_action.hvac_mode = self._determine_hvac_mode(hvac_power)

            return thermal_action

        except Exception as e:
            self.logger.error(f"Error in thermal controller step: {e}")
            # Return safe default action
            return ThermalAction(hvac_power=0.0)

    def _get_active_setpoints(self, occupancy, disturbance) -> tuple:
        """Determine active setpoints based on occupancy and schedule."""
        base_cooling = self.setpoint_cooling
        base_heating = self.setpoint_heating

        # Check if building is occupied
        is_occupied = getattr(occupancy, 'is_occupied', True)

        # Setback during unoccupied periods
        if not is_occupied:
            cooling_setpoint = base_cooling + 2.0  # Allow warmer
            heating_setpoint = base_heating - 2.0  # Allow cooler
        else:
            cooling_setpoint = base_cooling
            heating_setpoint = base_heating

        return cooling_setpoint, heating_setpoint

    def _calculate_hvac_power(self, current_temp: float, cooling_setpoint: float,
                            heating_setpoint: float, timestep: float) -> float:
        # Calculate temperature errors
        cooling_error = current_temp - cooling_setpoint
        heating_error = heating_setpoint - current_temp
        can_change_state = True

        # Determine control action based on mode
        if self.mode == HVACMode.COOLING:
            power = self._cooling_control(cooling_error, can_change_state)
        elif self.mode == HVACMode.HEATING:
            power = self._heating_control(heating_error, can_change_state)
        elif self.mode == HVACMode.AUTO:
            power = self._auto_control(cooling_error, heating_error, can_change_state)
        else:  # OFF mode
            power = 0.0

        # Update state tracking
        new_is_on = abs(power) > 0
        if new_is_on != self.hvac_is_on:
            self.last_state_change_time = timestep
            self.hvac_is_on = new_is_on

        return power

    def _cooling_control(self, cooling_error: float, can_change_state: bool) -> float:
        """Cooling-only control logic."""
        if cooling_error > self.deadband:
            # Too warm - start/increase cooling
            if cooling_error > 2 * self.deadband:
                return self.stage2_power  # High cooling
            else:
                return self.stage1_power  # Low cooling
        elif cooling_error < -self.deadband and can_change_state:
            # Cool enough - turn off
            return 0.0
        else:
            # Maintain current state
            return self.current_hvac_power

    def _heating_control(self, heating_error: float, can_change_state: bool) -> float:
        """Heating-only control logic."""
        if heating_error > self.deadband:
            # Too cool - start/increase heating
            if heating_error > 2 * self.deadband:
                return -self.stage2_power  # High heating (negative)
            else:
                return -self.stage1_power  # Low heating (negative)
        elif heating_error < -self.deadband and can_change_state:
            # Warm enough - turn off
            return 0.0
        else:
            # Maintain current state
            return self.current_hvac_power

    def _auto_control(self, cooling_error: float, heating_error: float,
                     can_change_state: bool) -> float:
        """Automatic heating/cooling control."""
        # Cooling needed
        if cooling_error > self.deadband:
            if cooling_error > 2 * self.deadband:
                return self.stage2_power
            else:
                return self.stage1_power

        # Heating needed
        elif heating_error > self.deadband:
            if heating_error > 2 * self.deadband:
                return -self.stage2_power
            else:
                return -self.stage1_power

        # In comfort zone - turn off if allowed
        elif can_change_state:
            return 0.0

        # Maintain current state
        else:
            return self.current_hvac_power

    def _determine_hvac_mode(self, power: float) -> str:
        """Determine HVAC operating mode from power level."""
        if power > 0:
            return "cooling"
        elif power < 0:
            return "heating"
        else:
            return "off"

    def reset(self) -> None:
        """Reset controller to initial state."""
        self.current_hvac_power = 0.0
        self.last_mode = HVACMode.OFF
        self.hvac_is_on = False
        self.last_state_change_time = 0
        self.clear_history()
        self.logger.debug(f"Reset thermal controller: {self.name}")

    def get_state(self) -> Dict[str, Any]:
        """Get current controller state."""
        return {
            'current_hvac_power': self.current_hvac_power,
            'hvac_is_on': self.hvac_is_on,
            'last_state_change_time': self.last_state_change_time,
            'mode': self.mode.value
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set controller state."""
        self.current_hvac_power = state.get('current_hvac_power', 0.0)
        self.hvac_is_on = state.get('hvac_is_on', False)
        self.last_state_change_time = state.get('last_state_change_time', 0)
        if 'mode' in state:
            self.mode = HVACMode(state['mode'])