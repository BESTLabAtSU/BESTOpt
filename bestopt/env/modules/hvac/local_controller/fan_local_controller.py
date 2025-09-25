from typing import Dict, Any, Optional
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import ThermalAction, HVACLocalAction, Disturbance  


class FanLocalController(BaseModule):
    """
    Local fan controller.

    Input  : upstream airflow setpoint from SupervisoryController  [m^3/s] 
            ThermalAction.supervisory_supply_air_flow_rate 
    Output : HVACLocalAction.fan_supply_air_flow_rate [m^3/s] = gain * input

    Config:
      - gain (float, default 0.99): local attenuation/efficiency factor
      - input_attr (str, optional): name of the airflow setpoint attribute to read
                                    default looks for 'supplyfan_flow_sp' first
    """

    def __init__(self, config: Dict[str, Any], name: str = "FanLocalController"):
        super().__init__(config, name)
        self.gain: float = float(config.get("gain", 1.0))
        self.input_attr: Optional[str] = config.get("input_attr", None)


    def initialize(self) -> None:
        self._initialized = True


    def step(
        self,          
        action: Any,                     
        timestep: float             
    ) -> "HVACLocalAction":
        
        """Compute local fan command and return a local-controller action."""
        
        upstream_flow = action.supervisory_supply_air_flow_rate
        if upstream_flow is None:
            cmd = None
        else:
            cmd = max(0.0, self.gain * float(upstream_flow))

        local_action = HVACLocalAction()
        local_action.fan_supply_air_flow_rate = cmd

        # self._record_state({"upstream_flow": upstream_flow, "cmd_flow": cmd})

        return local_action
    

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()
