"""
Distributed Energy Resources (DER) System Module
Aggregates and manages PV, Battery, and EV components
"""

from typing import Dict, Any, Optional
import logging

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import (
    ElectricalAction, Disturbance, DERSystemState,
    PVState, BatteryState, EVState, BatteryMode, EVMode
)
from bestopt.env.modules.ders.component.pv import PVModule
from bestopt.env.modules.ders.component.battery import BatteryModule
from bestopt.env.modules.ders.component.ev import EVModule


class DERModule(BaseModule):
    """
    DER System module that aggregates PV, Battery, and EV components.

    This module:
    - Manages multiple DER components as a unified system
    - Coordinates power flows between components
    - Provides aggregated metrics for the electrical domain
    - Handles component initialization and state management
    """

    def __init__(self, config: Dict[str, Any], name: str = "der_system"):
        """
        Initialize DER system with configured components.

        Args:
            config: DER system configuration
            name: Module name
        """
        super().__init__(config, name)

        # System configuration
        self.system_config = config.get("system_config", {})

        # Component availability flags
        self.has_pv = "pv" in config and config["pv"]
        self.has_battery = "battery" in config and config["battery"]
        self.has_ev = "ev" in config and config["ev"]

        # Initialize submodules based on configuration
        self.pv = None
        self.battery = None
        self.ev = None

        if self.has_pv:
            self.pv = PVModule(config.get("pv", {}), name=f"{name}_pv")

        if self.has_battery:
            self.battery = BatteryModule(config.get("battery", {}), name=f"{name}_battery")

        if self.has_ev:
            self.ev = EVModule(config.get("ev", {}), name=f"{name}_ev")

        # System-level parameters
        self.max_grid_import = config.get("max_grid_import", 15000)  # Watts
        self.max_grid_export = config.get("max_grid_export", 10000)  # Watts

        # Tracking metrics
        self.total_pv_generation_kwh = 0.0
        self.total_battery_throughput_kwh = 0.0
        self.total_ev_energy_kwh = 0.0
        self.operating_hours = 0.0

        self.logger = logging.getLogger(f"{__name__}.{name}")

    def initialize(self) -> None:
        """Initialize all DER components."""
        self.logger.info(f"Initializing DER system: {self.name}")

        if self.has_pv:
            self.pv.initialize()
            self.logger.info("- PV system initialized")

        if self.has_battery:
            self.battery.initialize()
            self.logger.info("- Battery system initialized")

        if self.has_ev:
            self.ev.initialize()
            self.logger.info("- EV charger initialized")

        self.logger.info(f"DER system initialization complete. Components: "
                         f"PV={self.has_pv}, Battery={self.has_battery}, EV={self.has_ev}")

    def step(
            self,
            state: ElectricalDomainState,
            action: ElectricalAction,
            disturbance: Disturbance,
            resolution: int,
            timestep: float
    ) -> Dict[str, Any]:
        """
        Execute DER system operation for current timestep.

        This method:
        1. Updates each component based on action and disturbances
        2. Aggregates power flows and generation
        3. Updates the electrical domain state

        Args:
            state: Current electrical domain state
            action: Electrical control action with power flow commands
            disturbance: Current disturbances (weather, prices, etc.)
            resolution: Time resolution in seconds
            timestep: Current simulation timestep

        Returns:
            Dictionary with DER system metrics
        """
        results = {
            "pv_generation": 0.0,
            "battery_power": 0.0,
            "ev_power": 0.0,
            "total_generation": 0.0,
            "total_storage_power": 0.0,
            "components": {}
        }

        try:
            # Step 1: Update PV generation
            if self.has_pv and "pv_1" in state.pv_systems:  # Assuming single PV for now
                pv_result = self.pv.step(
                    state=state.pv_systems["pv_1"],
                    action=action,
                    disturbance=disturbance,
                    resolution=resolution,
                    timestep=timestep
                )
                results["pv_generation"] = pv_result.get("power_generation", 0.0)
                results["components"]["pv"] = pv_result
                self.total_pv_generation_kwh += pv_result.get("energy_generated_kwh", 0.0)

            # Step 2: Update Battery operation
            if self.has_battery and "battery_1" in state.batteries:  # Assuming single battery
                battery_result = self.battery.step(
                    state=state.batteries["battery_1"],
                    action=action,
                    disturbance=disturbance,
                    timestep=timestep
                )
                results["battery_power"] = battery_result.get("power_actual", 0.0)
                results["components"]["battery"] = battery_result
                self.total_battery_throughput_kwh += abs(battery_result.get("energy_transferred_kwh", 0.0))

            # Step 3: Update EV charging/V2G
            if self.has_ev and "ev_1" in state.ev_chargers:  # Assuming single EV charger
                ev_result = self.ev.step(
                    state=state.ev_chargers["ev_1"],
                    action=action,
                    disturbance=disturbance,
                    timestep=timestep
                )
                results["ev_power"] = ev_result.get("power_actual", 0.0)
                results["components"]["ev"] = ev_result
                self.total_ev_energy_kwh += abs(ev_result.get("energy_transferred_kwh", 0.0))

            # Step 4: Aggregate DER metrics for the electrical domain
            state.total_generation = results["pv_generation"]

            # Storage power (negative = discharging to grid/building)
            state.total_storage_discharge = 0.0
            state.total_storage_charge = 0.0

            if results["battery_power"] < 0:  # Discharging
                state.total_storage_discharge += abs(results["battery_power"])
            else:  # Charging
                state.total_storage_charge += results["battery_power"]

            if self.has_ev and results["ev_power"] < 0:  # V2G
                state.total_storage_discharge += abs(results["ev_power"])
            elif self.has_ev and results["ev_power"] > 0:  # Charging
                state.total_storage_charge += results["ev_power"]

            # Calculate net DER contribution (positive = export, negative = import)
            results["total_generation"] = state.total_generation
            results["total_storage_power"] = state.total_storage_discharge - state.total_storage_charge

            # Update operating hours
            self.operating_hours += resolution / 3600.0

            # Log significant operations
            if abs(results["total_generation"]) > 10 or abs(results["total_storage_power"]) > 10:
                self.logger.debug(
                    f"DER Operation - PV: {results['pv_generation']:.1f}W, "
                    f"Battery: {results['battery_power']:.1f}W, "
                    f"EV: {results['ev_power']:.1f}W"
                )

            return results

        except Exception as e:
            self.logger.error(f"Error in DER system step: {e}")
            return results

    def reset(self) -> None:
        """Reset all DER components to initial state."""
        self.logger.info(f"Resetting DER system: {self.name}")

        if self.has_pv:
            self.pv.reset()

        if self.has_battery:
            self.battery.reset()

        if self.has_ev:
            self.ev.reset()

        # Reset tracking metrics
        self.total_pv_generation_kwh = 0.0
        self.total_battery_throughput_kwh = 0.0
        self.total_ev_energy_kwh = 0.0
        self.operating_hours = 0.0

    def get_state(self) -> Dict[str, Any]:
        """
        Get current state of all DER components.

        Returns:
            Dictionary containing state information for all components
        """
        state = {
            "system": {
                "total_pv_generation_kwh": self.total_pv_generation_kwh,
                "total_battery_throughput_kwh": self.total_battery_throughput_kwh,
                "total_ev_energy_kwh": self.total_ev_energy_kwh,
                "operating_hours": self.operating_hours
            },
            "components": {}
        }

        if self.has_pv:
            state["components"]["pv"] = self.pv.get_state_summary()

        if self.has_battery:
            state["components"]["battery"] = self.battery.get_state_summary()

        if self.has_ev and hasattr(self.ev, 'get_state_summary'):
            state["components"]["ev"] = self.ev.get_state_summary()

        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set state for all DER components.

        Args:
            state: State dictionary to restore
        """
        if "system" in state:
            system_state = state["system"]
            self.total_pv_generation_kwh = system_state.get("total_pv_generation_kwh", 0.0)
            self.total_battery_throughput_kwh = system_state.get("total_battery_throughput_kwh", 0.0)
            self.total_ev_energy_kwh = system_state.get("total_ev_energy_kwh", 0.0)
            self.operating_hours = system_state.get("operating_hours", 0.0)

        if "components" in state:
            components = state["components"]

            if self.has_pv and "pv" in components and hasattr(self.pv, 'set_state'):
                self.pv.set_state(components["pv"])

            if self.has_battery and "battery" in components and hasattr(self.battery, 'set_state'):
                self.battery.set_state(components["battery"])

            if self.has_ev and "ev" in components and hasattr(self.ev, 'set_state'):
                self.ev.set_state(components["ev"])

    def get_available_flexibility(self) -> Dict[str, float]:
        """
        Calculate available power flexibility from DER components.

        Returns:
            Dictionary with upward and downward flexibility in Watts
        """
        flexibility = {
            "upward": 0.0,  # Additional power that can be provided
            "downward": 0.0,  # Power that can be absorbed
            "components": {}
        }

        # Battery flexibility
        if self.has_battery and hasattr(self.battery, 'get_state_summary'):
            battery_state = self.battery.get_state_summary()
            flexibility["upward"] += battery_state.get("available_discharge_power_w", 0.0)
            flexibility["downward"] += battery_state.get("available_charge_power_w", 0.0)
            flexibility["components"]["battery"] = {
                "upward": battery_state.get("available_discharge_power_w", 0.0),
                "downward": battery_state.get("available_charge_power_w", 0.0)
            }

        # EV flexibility (if connected and V2G enabled)
        if self.has_ev and hasattr(self.ev, 'get_available_flexibility'):
            ev_flex = self.ev.get_available_flexibility()
            flexibility["upward"] += ev_flex.get("upward", 0.0)
            flexibility["downward"] += ev_flex.get("downward", 0.0)
            flexibility["components"]["ev"] = ev_flex

        # PV curtailment capability
        if self.has_pv and hasattr(self.pv, 'get_state_summary'):
            pv_state = self.pv.get_state_summary()
            current_generation = pv_state.get("power_generation", 0.0)
            if current_generation > 0:
                flexibility["downward"] += current_generation  # Can curtail PV
                flexibility["components"]["pv"] = {
                    "curtailable": current_generation
                }

        return flexibility

    def get_forecast(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """
        Get generation and flexibility forecast for planning.

        Args:
            hours_ahead: Number of hours to forecast

        Returns:
            Dictionary with forecasted metrics
        """
        forecast = {
            "pv_generation": None,
            "battery_availability": None,
            "ev_availability": None
        }

        if self.has_pv and hasattr(self.pv, 'get_forecast'):
            forecast["pv_generation"] = self.pv.get_forecast(hours_ahead)

        # Add battery and EV availability forecasts if methods exist

        return forecast
