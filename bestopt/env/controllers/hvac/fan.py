from typing import Dict, Any, Optional
from ...core.base import BaseModule
from ...core.data_structure import HVACState, ThermalAction, Disturbance


class FanLocalController(BaseModule):
    def __init__(self, config: Dict[str, Any], name: str = "FanLocalController"):
        """Initialize fan local controller."""
        super().__init__(config, name)
        self.fan_type = config.get("fan_type", "ideal")

        if self.fan_type == "constant":
            self.design_air_flow_rate = config.get("design_air_flow_rate", 0.5)

        elif self.fan_type == "staged":
            self.stage_air_flow_rate = config.get("stage_air_flow_rate")

        elif self.fan_type == "variable":
            self.min_air_flow_rate = config.get("min_air_flow_rate", 0.1)
            self.max_air_flow_rate = config.get("max_air_flow_rate", 1.0)

        # State tracking
        self.last_command = None
        self.fan_was_on = False

    def initialize(self) -> None:
        self.last_command = None
        self.fan_was_on = False
        self.logger.info(f"Initialized {self.fan_type} fan local controller: {self.name}")

    def step(self, supervisory_action: ThermalAction, state: HVACState,
             disturbance: Disturbance, timestep: float) -> Dict[str, Any]:
        """
        """
        try:
            requested_air_flow_rate = supervisory_action.supervisory_supply_air_flow_rate

            # call local controller
            if self.fan_type == "ideal":
                fan_command = self._ideal_control(requested_air_flow_rate)
            elif self.fan_type == "constant":
                fan_command = self._constant_control(requested_air_flow_rate)
            elif self.fan_type == "staged":
                fan_command = self._staged_control(requested_air_flow_rate)
            elif self.fan_type == "variable":
                fan_command = self._variable_control(requested_air_flow_rate)
            else:
                self.logger.warning(f"Unknown fan type: {self.fan_type}")

            self.last_command = fan_command

            return fan_command

        except Exception as e:
            self.logger.error(f"Error in fan local controller: {e}")
            return None

    def _ideal_control(self, requested_flow_ratio: float) -> Dict[str, Any]:
        pass

    def _constant_control(self, requested_flow_ratio: float) -> Dict[str, Any]:
        pass

    def _staged_control(self, requested_flow_ratio: float) -> Dict[str, Any]:
        fan_stage = min(
            self.stage_air_flow_rate,
            key=lambda s: abs(self.stage_air_flow_rate[s] - requested_flow_ratio)
        )
        return {"stage": fan_stage}

    def _variable_control(self, requested_flow_ratio: float) -> Dict[str, Any]:
        pass

    def reset(self) -> None:
        """Reset controller to initial state."""
        self.logger.debug(f"Reset fan local controller: {self.name}")
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get current controller state."""
        pass

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set controller state."""
        pass