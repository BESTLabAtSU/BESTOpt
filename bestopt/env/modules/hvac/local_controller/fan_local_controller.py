from typing import Dict, Any, Optional
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import HVACSystemAction, FanComponentAction, FanComponentState, ComponentType


class FanLocalController(BaseModule):
    """
    Local fan controller.

    Input  : upstream airflow setpoint from SupervisoryController  [m^3/s] 
            ThermalAction.supervisory_supply_air_flow_rate 
    Output : ThermalAction.fan_supply_air_flow_rate [m^3/s] = gain * input

    Config:
      - gain (float, default 0.99): local attenuation/efficiency factor
      - input_attr (str, optional): name of the airflow setpoint attribute to read
                                    default looks for 'supplyfan_flow_sp' first
    """

    def __init__(self, config: Dict[str, Any], name: str = "FanLocalController"):
        super().__init__(config, name)
        self.gain: float = float(config.get("gain", 1.0))

    def initialize(self) -> None:
        self._initialized = True

    def step(
            self,
            state: FanComponentState,
            action: HVACSystemAction,
            timestep: float,
    ) -> "FanComponentAction":

        """Compute local fan command and return a local-controller action."""

        upstream_flow = action.supply_airflow_setpoint_m3s

        if upstream_flow is None:
            cmd = None
        else:
            cmd = max(0.0, self.gain * float(upstream_flow))

        local_action = FanComponentAction(
            component_id=state.system_id,
            component_type=ComponentType.FAN.value
        )
        local_action.airflow_setpoint_m3s = cmd

        # self._record_state({"upstream_flow": upstream_flow, "cmd_flow": cmd})

        return local_action

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()
