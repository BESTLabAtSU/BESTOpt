from typing import Dict, Any, Optional, List
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import HVACSystemAction, FanComponentAction, FanComponentState, ComponentType


class FanLocalController(BaseModule):
    """
    Local fan controller.

    Input : action.supervisory_supply_air_flow_rate  [m^3/s]
    Output: HVACLocalAction.fan_supply_air_flow_rate [m^3/s]

    Config:
      - gain: float = 0.99
      - rated_flow_m3s: float = 1.0

      - ctrl_type: str 
        # If ctrl_type="linear", the fan stage is determined by flooring the demand to the nearest available stage.
        # If ctrl_type="staged", the output air flow rate is gain (default=1.0) * input air flow rate.

      - stage_breaks: List[float] = [0.25, 0.5, 0.75, 1.0]  # normalized upper bounds
      - stage_levels: List[float] = [0.25, 0.5, 0.75, 1.0]  # normalized outputs per stage
    """

    def __init__(self, config: Dict[str, Any], name: str = "FanLocalController"):
        super().__init__(config, name)
        self.gain: float = float(config.get("gain", 1.0))

        self.rated_flow_m3s: float = float(config.get("rated_flow_m3s", 1.0))
        self.ctrl_type: str = str(config.get("ctrl_type", "Staged"))

        ctrl = str(config.get("ctrl_type", "linear")).strip().lower()
        if ctrl not in ("linear", "staged"):
            self.logger.warning(
                f'{self.name}: ctrl_type must be "linear" or "staged"; fallback to "linear".'
            )
            ctrl = "linear"
        self.ctrl_type: str = ctrl
        self.staged: bool = (ctrl == "staged")
                

        # Optional custom staging
        self.stage_breaks: List[float] = list(config.get("stage_breaks", [0.25, 0.5, 0.75, 1.0]))
        self.stage_levels: List[float] = list(config.get("stage_levels", [0.25, 0.5, 0.75, 1.0]))

        # Basic sanity
        if self.rated_flow_m3s <= 0:
            # self.logger.warning(f"{self.name}: rated_flow_m3s <= 0, staging will be bypassed.")
            self.staged = False
        if self.staged:
            if not self.stage_breaks or not self.stage_levels or len(self.stage_levels) != len(self.stage_breaks):
                # self.logger.warning(f"{self.name}: no stage config specifying, Fallback to 3-stage defaults.")
                self.stage_breaks = [0.25, 0.5, 0.75, 1.0]
                self.stage_levels = [0.25, 0.5, 0.75, 1.0]

    def initialize(self) -> None:
        self._initialized = True

    def step(
        self,
        action: Any,
        timestep: float
    ) -> "HVACLocalAction":

        # 1) upstream setpoint
        upstream_flow = getattr(action, "supervisory_supply_air_flow_rate", None)
        if upstream_flow is None and hasattr(action, "thermal"):
            upstream_flow = getattr(action.thermal, "supervisory_supply_air_flow_rate", None)

        # 2) apply gain
        if upstream_flow is None:
            want = None
        else:
            want = max(0.0, self.gain * float(upstream_flow))

        # 3) staging logic (if enabled)
        if want is None or want == 0.0 or not self.staged:
            cmd = want if want is not None else None
        else:
            # normalize and snap to a stage
            q_norm = min(1.0, max(0.0, want / self.rated_flow_m3s))
            idx = 0
            for i, ub in enumerate(self.stage_breaks):
                if q_norm <= ub:
                    idx = i
                    break
            # map to output level
            q_out_norm = self.stage_levels[idx]
            cmd = q_out_norm * self.rated_flow_m3s

        # 4) emit local action
        local_action = HVACLocalAction()
        local_action.fan_supply_air_flow_rate = cmd

        # self._record_state({"upstream": upstream_flow, "want": want, "cmd": cmd})

        return local_action

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()
