"""
Integrates PV, Battery, EV with local controller
"""

from typing import Dict, Any
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import (
    ElectricalAction, Disturbance, ElectricalDomainState,
    PVState, BatteryState, EVState
)

from bestopt.env.modules.ders.component.pv import PVModule
from bestopt.env.modules.ders.component.battery import BatteryModule
from bestopt.env.modules.ders.component.ev import EVModule
from bestopt.env.modules.ders.local_controller.der_local_controller import DERLocalController


class PVBATEVModule(BaseModule):
    """
    DER System that includes:
      - 1x PV (generation)
      - 1x Battery (storage)
      - 1x EV (flexible load/storage)
      - 1x Local Controller (power dispatch)

    Control sequence:
      SupervisoryController → DERLocalController → PV/Battery/EV Components

    Inputs:
      - action.battery_power_setpoint (high-level battery command)
      - action.ev_power_setpoint (high-level EV command)
      - weather.solar_radiation
      - disturbance (occupancy for EV connection)

    Outputs:
      - pv_state.generation_W
      - battery_state.power_W, battery_state.battery_soc
      - ev_state.power_W, ev_state.ev_soc, ev_state.is_connected
      - Total DER system power, energy
    """

    def __init__(self, config: Dict[str, Any], name: str = "der_system"):
        super().__init__(config, name)

        # === Submodules ===
        self.pv = PVModule(config.get("pv", {}), name="pv")
        self.battery = BatteryModule(config.get("battery", {}), name="battery")
        self.ev = EVModule(config.get("ev", {}), name="ev")
        self.local_ctrl = DERLocalController(config.get("local_controller", {}), name="der_local_ctrl")

        # === States ===
        self.pv_state = PVState(component_id="pv", component_type="pv", domain="electrical")
        self.battery_state = BatteryState(component_id="battery", component_type="battery", domain="electrical")
        self.ev_state = EVState(component_id="ev", component_type="ev", domain="electrical")

        # === Tracking ===
        self.total_generation_W = 0.0
        self.total_storage_power_W = 0.0
        self.net_power_W = 0.0

    def initialize(self) -> None:
        """Initialize all DER components."""
        self.pv.initialize()
        self.battery.initialize()
        self.ev.initialize()
        self.local_ctrl.initialize()
        self._initialized = True
        self.logger.info(f"DER system '{self.name}' initialized")

    def reset(self) -> None:
        """Reset all DER components."""
        self.pv.reset()
        self.battery.reset()
        self.ev.reset()
        self.local_ctrl.reset()
        self._initialized = False
        self.initialize()

    def step(
            self,
            state: ElectricalDomainState,
            action: ElectricalAction,
            disturbance: Disturbance,
            resolution: int,
            timestep: float
    ) -> Dict[str, Any]:
        """
        Step all DER components in the system.

        Execution order:
        1. PV generation (weather-dependent, no control)
        2. Local controller (dispatch power to battery/EV based on supervisory command)
        3. Battery step (charge/discharge)
        4. EV step (charge if connected)
        5. Aggregate results
        """

        # === 1. PV generation (uncontrolled, weather-dependent) ===
        self.pv.step(self.pv_state, action, disturbance, resolution, timestep)

        # === 2. Local controller dispatch ===
        # Local controller takes supervisory action and PV generation,
        # then decides actual battery/EV power commands
        local_action = self.local_ctrl.step(
            supervisory_action=action,
            pv_generation_W=self.pv_state.generation_W,
            battery_state=self.battery_state,
            ev_state=self.ev_state,
            building_load_W=state.total_demand if hasattr(state, 'total_demand') else 0.0,
            timestep=timestep
        )

        # === 3. Battery step ===
        self.battery.step(self.battery_state, local_action, disturbance, timestep)

        # === 4. EV step ===
        self.ev.step(self.ev_state, local_action, disturbance, resolution, timestep)

        # === 5. Aggregate outputs ===
        self.total_generation_W = self.pv_state.generation_W
        self.total_storage_power_W = self.battery_state.power_W + self.ev_state.power_W
        self.net_power_W = self.total_generation_W + self.total_storage_power_W  # negative = discharge

        return {
            "pv_generation_W": self.pv_state.generation_W,
            "battery_power_W": self.battery_state.power_W,
            "battery_soc": self.battery_state.battery_soc,
            "ev_power_W": self.ev_state.power_W,
            "ev_soc": self.ev_state.ev_soc,
            "ev_connected": self.ev_state.is_connected,
            "net_power_W": self.net_power_W
        }

    def get_state(self) -> Dict[str, Any]:
        """Get system state."""
        return {
            "pv": self.pv.get_state(),
            "battery": self.battery.get_state(),
            "ev": self.ev.get_state()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set system state."""
        if "pv" in state:
            self.pv.set_state(state["pv"])
        if "battery" in state:
            self.battery.set_state(state["battery"])
        if "ev" in state:
            self.ev.set_state(state["ev"])