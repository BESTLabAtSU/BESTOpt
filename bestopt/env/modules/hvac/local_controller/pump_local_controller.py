from typing import Dict, Any, Optional
from math import inf

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import ThermalAction, HVACLocalAction, ThermalDomainState
from bestopt.env.core.constants import WATER_DENSITY, WATER_SPECIFIC_HEAT, AIR_DENSITY, AIR_SPECIFIC_HEAT


class PumpLocalController(BaseModule):
    """
    Local pump controller. Water flow is controlled based on a constant 
    temperature difference across the coil.

    Config:
        - delta_T (float, default 5.0): Desired temperature difference across the coil [K].
        - pump_flowrate_max (float, default inf): Maximum allowable flowrate [m^3/s].

    Input:
        - ThermalAction.thermal_load [W] 

    Output:
        - HVACLocalAction.pump_flowrate [m^3/s] = thermal_load / (delta_T * rho * cp)
    """

    def __init__(self, config: Dict[str, Any], name: str = "PumpLocalController"):
        super().__init__(config, name)
        self.delta_T: float = float(config.get("delta_T", 5.0))  # K
        # if self.delta_T <= 0:
        #     raise ValueError("delta_T must be a positive number.")

        self.pump_flowrate_max: float = float(config.get("pump_flowrate_max", inf))  # m^3/s
        if self.pump_flowrate_max <= 0:
            raise ValueError("pump_flowrate_max must be a positive number.")

    def initialize(self) -> None:
        """Initialize controller state."""
        self._initialized = True
        self.current_flowrate = None

    def step(
        self,
        state: ThermalDomainState,
        action: ThermalAction,
        timestep: float
    ) -> HVACLocalAction:
        """
        Compute local pump command and return a local-controller action.

        Args:
            action (ThermalAction): Contains the thermal load in Watts.
            timestep (float): Current timestep [s].

        Returns:
            HVACLocalAction: The computed pump flowrate command.
        """
        # this is why I am thinking we also need to pass an 'ID' like parameters later when extend to multizone building
        return_air_temp = state.thermal_zones["zone0"].temperature
        #@ TODO I use flowrate and temperature setpoint here to estimate thermal demand, is it OK?
        thermal_load = AIR_DENSITY * action.supervisory_supply_air_flow_rate * AIR_SPECIFIC_HEAT * (action.supervisory_supply_air_temperature - return_air_temp)

        # Calculate required flowrate
        pump_flowrate = abs(thermal_load / (
            self.delta_T * WATER_DENSITY * WATER_SPECIFIC_HEAT
        ))

        # Enforce maximum limit
        pump_flowrate = min(pump_flowrate, self.pump_flowrate_max)

        self.current_flowrate = pump_flowrate

        local_action = HVACLocalAction()
        local_action.pump_flowrate = pump_flowrate

        # Optionally record state for debugging/monitoring
        # self._record_state({
        #     "thermal_load": thermal_load,
        #     "pump_flowrate": pump_flowrate
        # })

        return local_action

    def reset(self) -> None:
        """Reset controller state."""
        self.clear_history()
        self.current_flowrate = None
        self._initialized = True
