"""
Electric Vehicle (EV) charging module.
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import EVComponentState, DERSystemAction, Disturbance


class EVStatus(Enum):
    """EV operational status."""
    CONNECTED_IDLE = "connected_idle"
    CHARGING = "charging"
    DISCHARGING = "discharging"  # V2G/V2B
    DISCONNECTED = "disconnected"
    FAULT = "fault"


class EVModule(BaseModule):
    """
    Electric Vehicle energy storage and charging model.

    Features:
    - State of Charge (SOC) tracking
    - Connection/disconnection scheduling
    - Charging/discharging efficiency
    - C-rate limitations
    - V2G/V2B capability
    - Simplified degradation model
    """

    def __init__(self, config: Dict[str, Any], name: str = "EV"):
        """
        Initialize EV module.

        Args:
            config: EV configuration parameters
            name: Module name
        """
        super().__init__(config, name)

        # Driving behavior configuration
        self.driving_power_kw = config.get("driving_power_kw", 15.0)  # Average power while driving
        self.driving_speed_kmh = config.get("driving_speed_kmh", 40.0)  # Average urban speed
        self.consumption_kwh_per_km = config.get("consumption_kwh_per_km", 0.18)  # Typical EV efficiency

        # Trip simulation parameters
        self.avg_trip_distance_km = config.get("avg_trip_distance_km", 15.0)  # One-way trip
        self.min_rest_duration_hours = config.get("min_rest_duration_hours", 0.5)  # Min parking time
        self.max_rest_duration_hours = config.get("max_rest_duration_hours", 4.0)  # Max parking time

        # Driving state tracking
        _initially_connected = config.get("initially_connected", True)
        self._prev_occupancy = 1.0 if _initially_connected else 0.0
        self._is_driving = False
        self._driving_time_remaining = 0.0
        self._rest_time_remaining = 0.0
        self._trip_phase = "home" if _initially_connected else "away_parked"
        self._total_trip_energy = 0.0

        # EV specifications
        self.capacity_kwh = config.get("rated_capacity_kWh", 60.0)  # Battery capacity in kWh
        self.nominal_voltage = config.get("nominal_voltage", 400.0)  # Volts
        self.max_charge_power_kw = config.get("max_charge_power_kw", 11.0)  # AC charging power (kW)
        self.max_discharge_power_kw = config.get("max_discharge_power_kw", 11.0)  # V2G/V2B power (kW)

        # SOC limits
        self.soc_min = config.get("soc_min", 0.2)  # Minimum SOC (20% - preserve battery life)
        self.soc_max = config.get("soc_max", 0.9)  # Maximum SOC (90% - preserve battery life)
        self.soc_initial = config.get("initial_soc", 0.5)  # Initial SOC
        self.soc_departure_target = config.get("soc_departure_target", 0.8)  # Target SOC for departure

        # Efficiency parameters
        self.charge_efficiency = config.get("charge_efficiency", 0.95)  # AC charging efficiency
        self.discharge_efficiency = config.get("discharge_efficiency", 0.95)  # V2G efficiency
        self.self_discharge_rate = config.get("self_discharge_rate", 0.00005)  # per hour (0.005%)

        # C-rate limits (simplified compared to battery)
        self.max_c_rate_charge = config.get("max_c_rate_charge", 0.5)  # 0.5C charging
        self.max_c_rate_discharge = config.get("max_c_rate_discharge", 0.5)  # 0.5C discharging

        # Connection schedule (list of tuples: (arrival_hour, departure_hour))
        self.connection_schedule = config.get("connection_schedule", [(18, 7)])  # Default: 6PM to 7AM
        self.always_connected = config.get("always_connected", False)  # Override for always connected

        # V2G/V2B settings
        self.v2g_enabled = config.get("v2g_enabled", True)  # Allow discharging
        self.v2g_min_soc = config.get("v2g_min_soc", 0.3)  # Don't discharge below this SOC

        # State tracking
        self.current_soc = self.soc_initial
        self.is_connected = config.get("initially_connected", True)
        self.last_connection_time = 0.0
        self.last_disconnection_time = 0.0
        self.total_energy_charged_kwh = 0.0
        self.total_energy_discharged_kwh = 0.0
        self.total_sessions = 0

        # Operational status
        self.status = EVStatus.CONNECTED_IDLE if self.is_connected else EVStatus.DISCONNECTED
        self.fault_message = ""

        # Power limits in Watts
        self.max_charge_power_w = self.max_charge_power_kw * 1000
        self.max_discharge_power_w = self.max_discharge_power_kw * 1000

        self.logger = logging.getLogger(f"{__name__}.{name}")

    def initialize(self) -> None:
        """Initialize the EV module."""
        self.logger.info(f"Initialized EV module: {self.name}")
        self.logger.info(f"Capacity: {self.capacity_kwh} kWh")
        self.logger.info(f"Max Charge Power: {self.max_charge_power_kw} kW")
        self.logger.info(f"Initial SOC: {self.soc_initial:.1%}")
        self.logger.info(f"V2G Enabled: {self.v2g_enabled}")

        # Set initial state
        self.current_soc = self.soc_initial

    def step(self,
             state: EVComponentState,
             action: Dict,
             disturbance: Disturbance,
             timestep: int) -> EVComponentState:
        """
        Execute EV charging/discharging for current timestep.

        Handles:
        - Normal charging when connected at home
        - Dynamic driving simulation when away (drive/rest cycles)
        - Realistic energy consumption based on distance/speed
        """
        try:
            timestep_hours = 900 / 3600  # 15 minutes = 0.25 hours

            # Get current occupancy
            current_occupancy = getattr(disturbance.occupancy, 'occupancy_fraction', 1.0)

            # Handle occupancy transitions and driving behavior
            driving_power_kw = self._update_driving_state(current_occupancy, timestep_hours)

            # Determine actual power based on whether driving or at home
            if self._trip_phase in ("departing", "returning") and self._is_driving:
                # Currently driving - consume energy
                power_actual_kw = 0.0  # No grid power while driving
                driving_energy_kwh = driving_power_kw * timestep_hours

                # Update SOC from driving
                soc_loss = driving_energy_kwh / self.capacity_kwh
                new_soc = max(self.soc_min, state.soc - soc_loss)
                self._total_trip_energy += driving_energy_kwh

                self.logger.debug(
                    f"Driving: {driving_power_kw:.1f}kW, "
                    f"SOC: {state.soc:.1%} -> {new_soc:.1%}, "
                    f"Phase: {self._trip_phase}"
                )

                state.soc = new_soc
                state.is_connected = False
                state.operation_mode = "DRIVING"

            elif self._trip_phase == "away_parked":
                # Parked away from home - no charging (could add workplace charging here)
                power_actual_kw = 0.0
                state.is_connected = False
                state.operation_mode = "PARKED_AWAY"

                # Apply minimal self-discharge while parked
                state.soc = self._apply_self_discharge(state.soc, timestep_hours)

            else:
                # At home - normal charging/V2G behavior
                state.is_connected = True
                power_command_kw = action.get('net_power_kw', 0.0)

                # Apply operational constraints
                power_actual_kw = self._apply_constraints(power_command_kw, state.soc)

                # Calculate energy transferred
                energy_delta_kwh = self._calculate_energy_transfer(power_actual_kw, timestep_hours)

                # Update SOC
                new_soc = self._update_soc(state.soc, energy_delta_kwh)
                new_soc = self._apply_self_discharge(new_soc, timestep_hours)

                state.soc = new_soc

                # Update operation mode
                if power_actual_kw > 0.1:
                    state.operation_mode = "CHARGING"
                    self.status = EVStatus.CHARGING
                elif power_actual_kw < -0.1:
                    state.operation_mode = "V2G"
                    self.status = EVStatus.DISCHARGING
                else:
                    state.operation_mode = "IDLE"
                    self.status = EVStatus.CONNECTED_IDLE

                # Update cumulative metrics
                if power_actual_kw > 0:
                    self.total_energy_charged_kwh += energy_delta_kwh
                elif power_actual_kw < 0:
                    self.total_energy_discharged_kwh += abs(energy_delta_kwh)

            # Store power for external reference
            state.power_w = power_actual_kw * 1000

            # Update internal tracking
            self.current_soc = state.soc
            self.is_connected = state.is_connected

        except Exception as e:
            self.logger.error(f"Error in EV step calculation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.status = EVStatus.FAULT
            self.fault_message = str(e)
            state.power_w = 0.0

        return state

    def _update_driving_state(self, current_occupancy: float, timestep_hours: float) -> float:
        """
        Update driving state machine and return current driving power consumption.

        State machine:
        - home: Person at home, EV connected
        - departing: Person left, EV driving to destination (with rest stops)
        - away_parked: Arrived at destination, parked
        - returning: Person coming back, EV driving home (with rest stops)

        Returns:
            Driving power in kW (0 if not currently driving)
        """
        prev_occupancy = self._prev_occupancy
        self._prev_occupancy = current_occupancy

        # Detect occupancy transitions
        person_left = prev_occupancy > 0.5 and current_occupancy <= 0.5
        person_returned = prev_occupancy <= 0.5 and current_occupancy > 0.5

        # State transitions based on occupancy changes
        if person_left and self._trip_phase == "home":
            # Start departure trip
            self._trip_phase = "departing"
            self._start_drive_segment()
            self._total_trip_energy = 0.0
            self.logger.info(f"EV departing - starting trip, SOC: {self.current_soc:.1%}")

        elif person_returned and self._trip_phase in ("away_parked", "departing"):
            # Start return trip
            self._trip_phase = "returning"
            self._start_drive_segment()
            self.logger.info(f"EV returning home, SOC: {self.current_soc:.1%}")

        # Process current driving/rest state
        driving_power = 0.0

        if self._trip_phase in ("departing", "returning"):
            if self._is_driving:
                # Currently driving
                self._driving_time_remaining -= timestep_hours
                driving_power = self._get_driving_power()

                if self._driving_time_remaining <= 0:
                    # Finished this drive segment
                    if self._trip_phase == "departing":
                        # Arrived at destination
                        self._trip_phase = "away_parked"
                        self._start_rest_period()
                        self.logger.info(
                            f"Arrived at destination, parking. Trip energy: {self._total_trip_energy:.2f} kWh")
                    else:
                        # Arrived home
                        self._trip_phase = "home"
                        self._is_driving = False
                        self.logger.info(f"Arrived home. Total trip energy: {self._total_trip_energy:.2f} kWh")

            else:
                # Currently resting/stopped during trip
                self._rest_time_remaining -= timestep_hours

                if self._rest_time_remaining <= 0:
                    # Rest over, continue driving
                    self._start_drive_segment()

        elif self._trip_phase == "away_parked":
            # Parked at destination, waiting for return signal
            self._rest_time_remaining -= timestep_hours
            # Could add random errand trips here if desired

        return driving_power

    def _start_drive_segment(self) -> None:
        """Initialize a new driving segment."""
        self._is_driving = True

        # Calculate drive time based on distance and speed
        # Add some randomness (±30%)
        distance = self.avg_trip_distance_km * (0.7 + 0.6 * np.random.random())
        drive_time = distance / self.driving_speed_kmh

        self._driving_time_remaining = drive_time

        self.logger.debug(f"Starting drive segment: {distance:.1f}km, {drive_time * 60:.1f}min")

    def _start_rest_period(self) -> None:
        """Initialize a rest/parking period."""
        self._is_driving = False

        # Random rest duration between min and max
        rest_duration = self.min_rest_duration_hours + \
                        (self.max_rest_duration_hours - self.min_rest_duration_hours) * np.random.random()

        self._rest_time_remaining = rest_duration

        self.logger.debug(f"Starting rest period: {rest_duration * 60:.1f}min")

    def _get_driving_power(self) -> float:
        """
        Get current driving power consumption with some variability.

        Simulates varying driving conditions (acceleration, cruising, traffic).
        """
        # Base consumption from efficiency and speed
        base_power = self.consumption_kwh_per_km * self.driving_speed_kmh

        # Add variability (±40%) to simulate real driving
        variability = 0.6 + 0.8 * np.random.random()

        return base_power * variability

    def _update_connection_status(self, state: EVComponentState, timestep: int) -> None:
        """
        Update EV connection status based on schedule.

        Args:
            state: Current EV state
            timestep: Current simulation timestep in seconds
        """
        if self.always_connected:
            state.is_active = True
            self.is_connected = True
            return

        # Convert timestep to hour of day (0-24)
        hour_of_day = (timestep / 3600) % 24

        # Check if EV should be connected based on schedule
        was_connected = self.is_connected
        self.is_connected = False

        for arrival_hour, departure_hour in self.connection_schedule:
            if departure_hour > arrival_hour:
                # Normal case: arrival before departure (e.g., 18-23)
                if arrival_hour <= hour_of_day < departure_hour:
                    self.is_connected = True
                    break
            else:
                # Overnight case: departure after midnight (e.g., 18-7)
                if hour_of_day >= arrival_hour or hour_of_day < departure_hour:
                    self.is_connected = True
                    break

        # Update state
        state.is_active = self.is_connected

        # Track connection events
        if self.is_connected and not was_connected:
            self.last_connection_time = timestep
            self.total_sessions += 1
            self.logger.info(f"EV connected at hour {hour_of_day:.1f}")
        elif not self.is_connected and was_connected:
            self.last_disconnection_time = timestep
            self.logger.info(f"EV disconnected at hour {hour_of_day:.1f}, SOC: {state.soc:.1%}")

    def _apply_constraints(self, power_command_kw: float, current_soc: float) -> float:
        """
        Apply EV operational constraints.

        Args:
            power_command_kw: Requested power in kW (positive=charge, negative=discharge)
            current_soc: Current state of charge (0-1)

        Returns:
            Actual power after applying constraints (kW)
        """
        if power_command_kw > 0:  # Charging
            # Check SOC limit
            if current_soc >= self.soc_max:
                return 0.0

            # Apply power and C-rate limits
            max_charge = min(
                self.max_charge_power_kw,
                self.max_c_rate_charge * self.capacity_kwh
            )

            # SOC-based derating (slower charging near full)
            if current_soc > 0.8:
                soc_factor = 1.0 - (current_soc - 0.8) * 5  # Rapid taper above 80%
                max_charge *= max(0.1, soc_factor)

            return min(power_command_kw, max_charge)

        elif power_command_kw < 0:  # Discharging (V2G/V2B)
            # Check if V2G is enabled
            if not self.v2g_enabled:
                return 0.0

            # Check SOC limit for V2G
            if current_soc <= self.v2g_min_soc:
                return 0.0

            # Apply power and C-rate limits
            max_discharge = min(
                self.max_discharge_power_kw,
                self.max_c_rate_discharge * self.capacity_kwh
            )

            # SOC-based derating (protect battery at low SOC)
            if current_soc < 0.4:
                soc_factor = (current_soc - self.v2g_min_soc) / (0.4 - self.v2g_min_soc)
                max_discharge *= max(0.1, soc_factor)

            return max(power_command_kw, -max_discharge)

        return 0.0

    def _calculate_energy_transfer(self, power_kw: float, timestep_hours: float) -> float:
        """
        Calculate energy transferred accounting for efficiency.

        Args:
            power_kw: Power in kW
            timestep_hours: Timestep duration in hours

        Returns:
            Energy change in kWh (positive = added to battery)
        """
        if power_kw > 0:  # Charging
            energy_kwh = power_kw * timestep_hours * self.charge_efficiency
        elif power_kw < 0:  # Discharging
            energy_kwh = power_kw * timestep_hours / self.discharge_efficiency
        else:
            energy_kwh = 0.0

        return energy_kwh

    def _update_soc(self, current_soc: float, energy_delta_kwh: float) -> float:
        """
        Update SOC based on energy transfer.

        Args:
            current_soc: Current SOC (0-1)
            energy_delta_kwh: Energy change in kWh

        Returns:
            New SOC (0-1)
        """
        # Calculate SOC change
        soc_delta = energy_delta_kwh / self.capacity_kwh
        new_soc = current_soc + soc_delta

        # Enforce limits
        new_soc = max(self.soc_min, min(self.soc_max, new_soc))

        return new_soc

    def _apply_self_discharge(self, soc: float, timestep_hours: float) -> float:
        """
        Apply self-discharge losses (minimal for EVs).

        Args:
            soc: Current SOC
            timestep_hours: Timestep in hours

        Returns:
            Updated SOC after self-discharge
        """
        discharge_factor = 1.0 - (self.self_discharge_rate * timestep_hours)
        return soc * discharge_factor

    def get_available_charge_power(self, soc: float) -> float:
        """
        Get available charging power at current SOC.

        Args:
            soc: Current SOC

        Returns:
            Available charge power in kW
        """
        if not self.is_connected or soc >= self.soc_max:
            return 0.0

        power = self.max_charge_power_kw

        # SOC-based derating
        if soc > 0.8:
            power *= (1.0 - (soc - 0.8) * 5)

        return power

    def get_available_discharge_power(self, soc: float) -> float:
        """
        Get available V2G discharge power at current SOC.

        Args:
            soc: Current SOC

        Returns:
            Available discharge power in kW
        """
        if not self.is_connected or not self.v2g_enabled or soc <= self.v2g_min_soc:
            return 0.0

        power = self.max_discharge_power_kw

        # SOC-based derating
        if soc < 0.4:
            power *= (soc - self.v2g_min_soc) / (0.4 - self.v2g_min_soc)

        return power

    def reset(self) -> None:
        """Reset EV module to initial state."""
        self.current_soc = self.soc_initial
        self.is_connected = self.config.get("initially_connected", True)
        self.last_connection_time = 0.0
        self.last_disconnection_time = 0.0
        self.total_energy_charged_kwh = 0.0
        self.total_energy_discharged_kwh = 0.0
        self.total_sessions = 0

        # Reset driving state
        _initially_connected = self.config.get("initially_connected", True)
        self._prev_occupancy = 1.0 if _initially_connected else 0.0
        self._is_driving = False
        self._driving_time_remaining = 0.0
        self._rest_time_remaining = 0.0
        self._trip_phase = "home" if _initially_connected else "away_parked"
        self._total_trip_energy = 0.0

        self.status = EVStatus.CONNECTED_IDLE if self.is_connected else EVStatus.DISCONNECTED
        self.fault_message = ""
        self.logger.debug(f"Reset EV module: {self.name}")

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of EV state.

        Returns:
            Dictionary with key metrics
        """
        return {
            "soc": self.current_soc,
            "capacity_kwh": self.capacity_kwh,
            "is_connected": self.is_connected,
            "status": self.status.value,
            "total_charged_kwh": self.total_energy_charged_kwh,
            "total_discharged_kwh": self.total_energy_discharged_kwh,
            "total_sessions": self.total_sessions,
            "v2g_enabled": self.v2g_enabled,
            "available_charge_power_kw": self.get_available_charge_power(self.current_soc),
            "available_discharge_power_kw": self.get_available_discharge_power(self.current_soc),
            "time_to_target_soc_hours": self._estimate_time_to_target()
        }

    def _estimate_time_to_target(self) -> float:
        """
        Estimate time to reach departure target SOC.

        Returns:
            Hours needed to reach target SOC
        """
        if not self.is_connected or self.current_soc >= self.soc_departure_target:
            return 0.0

        energy_needed = (self.soc_departure_target - self.current_soc) * self.capacity_kwh
        avg_charge_power = self.max_charge_power_kw * 0.8  # Assume 80% of max on average

        return energy_needed / (avg_charge_power * self.charge_efficiency)