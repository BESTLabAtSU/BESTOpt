"""
Ideal HVAC module.
"""

from typing import Dict, Any
from ...core.base import BaseModule
from ...core.data_structure import ThermalAction, FanState, Action, Disturbance  # 'state' not used here

class FanModule(BaseModule):
    """
    Supply fan module that CONSUMES an airflow setpoint and UPDATES a FanState in place.

    Input (from action): ThermalAction.supplyfan_flow_sp  [m^3/s]

    Output (written in-place to FanState):
    - state.flow_m3s
    - state.power_kw
    - state.energy_kwh_cum  (accumulated over steps)

    Model
    - Fan affinity law:  P = P_rated * (Q / Q_rated)^exponent
    - Config keys:
        * rated_flow_m3s  (or rated_flow)  Q_rated [m^3/s]
        * rated_power_kw                   P at Q_rated [kW]
        * power_exponent (default 3.0)     cube-law exponent

    State
    - Expects a FanState instance passed as `state`.
        * state.airflow_m3s     : echoed airflow [m^3/s]
        * state.power_kw        : electric power via affinity law [kW]
        * state.energy_kwh_cum  : accumulated energy over steps [kWh]

    Action
        * ThermalAction.supplyfan_flow_sp

    """

    def __init__(self, config: Dict[str, Any], name: str = "supply_fan"):
        super().__init__(config, name)
        # accept both 'rated_flow_m3s' and 'rated_flow' for convenience
        q_rated = config.get("rated_flow_m3s", config.get("rated_flow", 1.0))
        self.rated_flow_m3s: float = float(q_rated)                # Q_rated
        self.rated_power_kw: float = float(config.get("rated_power_kw", 1.0))  # P@Q_rated
        self.power_exponent: float = float(config.get("power_exponent", 3.0))  # ~3 for fans

        if self.rated_flow_m3s <= 0.0:
            self.logger.warning(f"{self.name}: rated_flow_m3s <= 0, power will be forced to 0.")

    def initialize(self) -> None:
        self._initialized = True

    def step(
        self,
        state: "FanState",
        action: "ThermalAction",
        disturbance: "Disturbance",
        timestep: float
    ) -> Dict[str, Any]:
        """
        One step:
          - read setpoint (m^3/s)
          - compute power via affinity law
          - accumulate energy for this timestep
          - write results IN-PLACE into FanState
        Return {} to satisfy BaseModule interface (environment may ignore it).
        """
        # 1) read setpoint with compatibility: ThermalAction or Action.thermal
        sp = getattr(action, "supplyfan_flow_sp", None)
        # If caller passes a bus-style Action with .thermal, support it as well
        if sp is None and hasattr(action, "thermal"):
            sp = getattr(action.thermal, "supplyfan_flow_sp", None)

        flow = 0.0 if sp is None else float(sp)
        if flow < 0.0:
            self.logger.warning(f"{self.name}: negative flow received; clamped to 0.0")
            flow = 0.0

        # 2) power via affinity law
        if self.rated_flow_m3s > 0.0 and self.rated_power_kw >= 0.0:
            ratio = flow / self.rated_flow_m3s
            power_kw = self.rated_power_kw * (ratio ** self.power_exponent)
        else:
            power_kw = 0.0

        # 3) energy for this step [kWh]; timestep in seconds
        energy_kwh = power_kw * (timestep / 3600.0) if timestep and timestep > 0.0 else 0.0

        # 4) in-place update
        state.airflow_m3s = flow
        state.power_kw = power_kw
        state.energy_kwh_cum += energy_kwh

        # 5) optional history snapshot
        self._record_state({"flow_m3s": flow, "power_kw": power_kw, "energy_kwh": energy_kwh})


    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()