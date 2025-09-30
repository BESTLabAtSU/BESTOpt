"""

"""

import numpy as np
from typing import Dict, Any, Optional
from ..core.base import BaseModule
from ..core.data_structure import State, Action, Disturbance, Observation, ThermalAction, HVACMode


class SupervisoryController(BaseModule):
    """
    Rule-based thermal controller.
    """

    def __init__(self, config: Dict[str, Any], name: str = "SupervisoryController"):
        """Initialize rule-based thermal controller."""
        super().__init__(config, name)

        # Control parameters
        self.domain = config.get("domain", "thermal")
        self.mode = HVACMode(config.get("mode", "auto"))
        self.precool_config = config.get("precooling", None)

        # Comfort settings
        # @TODO These settings should be more dynamic in the future
        self.base_cooling = config.get("base_cooling", 24.0)  # °C
        self.base_heating = config.get("base_heating", 18.0)  # °C
        self.deadband = config.get("deadband", 0.5)  # °C

        # HVAC capacity settings
        self.cooling_power_max = config.get("cooling_power_max", 4000.0)  # W
        self.heating_power_max = config.get("heating_power_max", 4000.0)  # W
        self.stage1_power = config.get("stage1_power", 2000.0)  # W (50% capacity)
        self.stage2_power = config.get("stage2_power", 4000.0)  # W (100% capacity)

        # Control states
        self.current_hvac_thermal_load = 0.0
        self.last_mode = HVACMode.OFF

    def initialize(self) -> None:
        """Initialize the controller."""
        self.current_hvac_thermal_load = 0.0
        self.current_supply_air_flow_rate = 0.0
        self.current_supply_air_temperature = 13
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

            # Determine active setpoints based on occupancy
            cooling_setpoint, heating_setpoint = self._get_active_setpoints(disturbance, self.precool_config)

            supply_air_flow_rate, supply_air_temperature = self._supervisory(
                current_temp=current_temp,
                cooling_setpoint=cooling_setpoint,
                heating_setpoint=heating_setpoint,
                timestep=timestep
            )

            self.current_supply_air_flow_rate = supply_air_flow_rate
            self.current_supply_air_temperature = supply_air_temperature

            # Create thermal action
            thermal_action = ThermalAction()
            thermal_action.supervisory_supply_air_flow_rate = supply_air_flow_rate
            thermal_action.supervisory_supply_air_temperature = supply_air_temperature
            thermal_action.supervisory_cooling_setpoint = cooling_setpoint
            thermal_action.supervisory_heating_setpoint = heating_setpoint
            #@ TODO update the following function
            # thermal_action.hvac_mode = self._determine_hvac_mode(hvac_power)

            return thermal_action

        except Exception as e:
            self.logger.error(f"Error in thermal controller step: {e}")
            # Return safe default action
            return ThermalAction(thermal_load=0.0)

    def _get_active_setpoints(self, disturbance, precool_config) -> tuple:
        """Determine active setpoints based on occupancy and schedule."""
        occupancy = disturbance.occupancy.occupancy_fraction
        step_of_day = disturbance.occupancy.step_of_day

        base_cooling = self.base_cooling
        base_heating = self.base_heating

        # Check if building is occupied
        if occupancy>0.0:
            # occupied
            cooling_setpoint = base_cooling
            heating_setpoint = base_heating
        else:
            cooling_setpoint = base_cooling + 2.0
            heating_setpoint = base_heating - 2.0

        # Check if need pre-cooling
        if precool_config:
            self.pre_degree = precool_config.get("degree", 2.0)
            self.pre_hour = precool_config.get("hours", 2)
            if step_of_day >= disturbance.prices.peak_start - self.pre_hour * 4 and step_of_day < disturbance.prices.peak_start:
                cooling_setpoint = base_cooling - self.pre_degree
                heating_setpoint = base_heating + self.pre_degree
        else:
            self.pre_degree = 0
            self.pre_hour = 0

        return cooling_setpoint, heating_setpoint

    def _supervisory(self, current_temp: float, cooling_setpoint: float,
                            heating_setpoint: float, timestep: float) -> float:
        # Calculate temperature errors
        cooling_error = current_temp - cooling_setpoint
        heating_error = heating_setpoint - current_temp
        can_change_state = True #@TODO add cyclying constraint later

        # Determine control action based on mode
        if self.mode == HVACMode.COOLING:
            supply_air_flow_rate, supply_air_temperature = self._cooling_control(cooling_error, can_change_state)
        elif self.mode == HVACMode.HEATING:
            supply_air_flow_rate, supply_air_temperature = self._heating_control(heating_error, can_change_state)
        elif self.mode == HVACMode.AUTO:
            supply_air_flow_rate, supply_air_temperature = self._auto_control(cooling_error, heating_error, can_change_state)
        else:  # OFF mode
            supply_air_flow_rate, supply_air_temperature = 0.0, 0.0

        return supply_air_flow_rate, supply_air_temperature

    def _cooling_control(self, cooling_error: float, can_change_state: bool) -> float:
        """Cooling-only control logic."""
        if self.current_supply_air_flow_rate == 0.0:
            if cooling_error>self.deadband:
                self.current_supply_air_flow_rate = 0.5
            else:
                self.current_supply_air_flow_rate = 0.0
        elif self.current_supply_air_flow_rate == 0.5:
            if cooling_error>self.deadband*2:
                self.current_supply_air_flow_rate = 1
            elif cooling_error<self.deadband*-1:
                self.current_supply_air_flow_rate = 0
            else:
                self.current_supply_air_flow_rate = 0.5
        else:
            if cooling_error<self.deadband:
                self.current_supply_air_flow_rate = 0.5
            elif cooling_error<self.deadband*-1:
                self.current_supply_air_flow_rate = 0
            else:
                self.current_supply_air_flow_rate = 1

        self.supply_air_temperature = 13
        return self.current_supply_air_flow_rate, self.supply_air_temperature

    # @Revise heating, auto later
    def _heating_control(self, heating_error: float, can_change_state: bool) -> float:
        """Heating-only control logic."""
        if heating_error > self.deadband:
            # Too cool - start/increase heating
            if heating_error > 2 * self.deadband:
                return self.stage2_power
            else:
                return self.stage1_power
        elif heating_error < -self.deadband and can_change_state:
            # Warm enough - turn off
            return 0.0
        else:
            # Maintain current state
            return self.current_hvac_thermal_load

    def _auto_control(self, cooling_error: float, heating_error: float,
                     can_change_state: bool) -> float:
        """Automatic heating/cooling control."""
        # Cooling needed
        if cooling_error > self.deadband:
            if cooling_error > 2 * self.deadband:
                return self.stage2_power*-1
            else:
                return self.stage1_power*-1

        # Heating needed
        elif heating_error > self.deadband:
            if heating_error > 2 * self.deadband:
                return self.stage2_power
            else:
                return self.stage1_power

        # In comfort zone - turn off if allowed
        elif can_change_state:
            return 0.0

        # Maintain current state
        else:
            return self.current_hvac_thermal_load

    def _determine_hvac_mode(self, thermal_load: float) -> str:
        """Determine HVAC operating mode from thermal_load level."""
        if thermal_load < 0:
            return "cooling"
        elif thermal_load > 0:
            return "heating"
        else:
            return "off"

    def reset(self) -> None:
        """Reset controller to initial state."""
        self.current_hvac_thermal_load = 0.0
        self.last_mode = HVACMode.OFF
        self.hvac_is_on = False
        self.last_state_change_time = 0
        self.clear_history()
        self.logger.debug(f"Reset thermal controller: {self.name}")

    def get_state(self) -> Dict[str, Any]:
        """Get current controller state."""
        return {
            'current_hvac_thermal_load': self.current_hvac_thermal_load,
            'hvac_is_on': self.hvac_is_on,
            'last_state_change_time': self.last_state_change_time,
            'mode': self.mode.value
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set controller state."""
        self.current_hvac_thermal_load = state.get('current_hvac_thermal_load', 0.0)
        self.hvac_is_on = state.get('hvac_is_on', False)
        self.last_state_change_time = state.get('last_state_change_time', 0)
        if 'mode' in state:
            self.mode = HVACMode(state['mode'])