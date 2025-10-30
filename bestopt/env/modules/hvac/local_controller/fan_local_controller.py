from typing import Dict, Any
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import HVACSystemAction, FanComponentAction, FanComponentState, ComponentType


class FanLocalController(BaseModule):
    """
    Fan local controller with three behaviors:
      - "vfd"      : tracks upstream as analog (first-order smoothing + rate limit + clamp to rated)
      - "staged"   : discrete stages {0, 0.33, 0.67, 1.0} * rated_flow with hysteresis + min dwell + rate limit
      - "constant" : free ON/OFF; output is {0, on_fraction*rated} with min dwell + rate limit
    """

    def __init__(self, config: Dict[str, Any], name: str = "FanLocalController"):
        super().__init__(config, name)

        self.gain: float = float(config.get("gain", 0.97))
        ctrl = str(config.get("ctrl_type", "vfd")).strip().lower()
        if ctrl not in ("vfd", "staged", "constant"):
            self.logger.warning(f'{self.name}: ctrl_type must be "vfd" | "staged" | "constant"; fallback to "vfd".')
            ctrl = "vfd"
        self.ctrl_type: str = ctrl

        self.rated_flow_m3s: float = float(config.get("rated_flow_m3s", 1.0))
        self.rate_limit_m3s_per_s: float = float(config.get("rate_limit_m3s_per_s", 0.2))
        self.off_threshold_m3s: float = float(config.get("off_threshold_m3s", 1e-4))

        self.tau_s: float = float(config.get("time_constant_s", 2.0))

        # ---- Staged  ----
        self.stage_hyst: float = float(config.get("stage_hysteresis", 0.04))
        self.min_dwell_steps: int = int(config.get("min_dwell_steps", 5))

        stages = config.get("stages", None)
        levels_cfg = config.get("stage_levels", None)
        breaks_cfg = config.get("stage_breaks", None)

        def _derive_breaks(levels):
            if len(levels) < 2:
                return [1.0]
            mids = [(levels[i] + levels[i+1]) * 0.5 for i in range(len(levels) - 1)]
            return mids + [1.0]

        if isinstance(levels_cfg, list) and len(levels_cfg) >= 2:
            lv = [float(x) for x in levels_cfg]
            lv.sort()
            lv[0] = 0.0
            lv[-1] = 1.0
            if isinstance(breaks_cfg, list) and len(breaks_cfg) == len(lv):
                bk = [float(x) for x in breaks_cfg]
            else:
                bk = _derive_breaks(lv)
            self.stage_levels = lv
            self.stage_breaks = bk

        elif isinstance(stages, int) and stages >= 2:
            N = min(stages, 50)
            step = 1.0 / (N - 1)
            lv = [round(i * step, 10) for i in range(N)]
            bk = _derive_breaks(lv)
            self.stage_levels = lv
            self.stage_breaks = bk

        else:
            # default to 4 levels
            lv = [0.0, 1.0/3.0, 2.0/3.0, 1.0]
            bk = _derive_breaks(lv)
            self.stage_levels = lv
            self.stage_breaks = bk

            
        # ---- Constant ----
        self.on_fraction: float = float(config.get("on_fraction", 1.0))
        self.on_cmd_threshold: float = float(config.get("on_cmd_threshold", 0.05))
        self.const_min_dwell_steps: int = int(config.get("const_min_dwell_steps", 3))

        self._q_prev: float = 0.0
        self._stage_idx: int = 0
        self._dwell_counter: int = 0

        self._const_on: bool = None
        self._const_dwell_counter: int = 0

        if self.rated_flow_m3s < 0:
            self.logger.warning(f"{self.name}: rated_flow_m3s < 0; clamped to 0.")
            self.rated_flow_m3s = 0.0
        if self.ctrl_type == "staged":
            if (not self.stage_levels) or (not self.stage_breaks) or (len(self.stage_levels) != len(self.stage_breaks)):
                self.logger.warning(f"{self.name}: invalid stage config; reset to 4-level defaults.")
                self.stage_levels = [0.0, 0.33, 0.67, 1.0]
                self.stage_breaks = [0.165, 0.5, 0.835, 1.0]

    def initialize(self) -> None:
        self._initialized = True
        self._q_prev = 0.0
        self._stage_idx = 0
        self._dwell_counter = 0
        self._const_on = None
        self._const_dwell_counter = 0

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _rate_limit(self, target: float, prev: float, dt: float) -> float:
        if dt and self.rate_limit_m3s_per_s > 0:
            max_step = self.rate_limit_m3s_per_s * dt
            return self._clip(target, prev - max_step, prev + max_step)
        return target

    def _cmd_vfd(self, want: float, dt: float) -> float:
        rated = max(0.0, self.rated_flow_m3s)
        want = self._clip(want, 0.0, rated)
        if dt and self.tau_s > 0:
            alpha = self._clip(dt / (self.tau_s + 1e-9), 0.0, 1.0)
        else:
            alpha = 1.0
        q_raw = self._q_prev + alpha * (want - self._q_prev)
        q_cmd = self._rate_limit(q_raw, self._q_prev, dt)
        q_cmd = min(q_cmd, rated)
        self._q_prev = q_cmd
        return 0.0 if q_cmd < self.off_threshold_m3s else q_cmd

    def _cmd_constant(self, upstream: float | None) -> float:
        rated = max(0.0, self.rated_flow_m3s)
        if upstream is None:
            return 0.0
        if float(upstream) <= 0.0:
            return 0.0
        level = self._clip(self.on_fraction, 0.0, 1.0)
        return min(level * rated, rated)

    def _cmd_staged(self, want: float, dt: float) -> float:
        rated = max(0.0, self.rated_flow_m3s)
        if rated <= 0.0:
            return 0.0

        u = 0.0 if want is None else float(want) / rated
        if u <= 0.0:
            target_flow = 0.0
            self._stage_idx = 0
        else:
            levels = self.stage_levels
            idx = None
            for i in range(1, len(levels)):
                if u <= levels[i]:
                    idx = i
                    break
            if idx is None:
                idx = len(levels) - 1  

            self._stage_idx = idx
            level = self._clip(levels[idx], 0.0, 1.0)
            target_flow = level * rated

        q_cmd = self._rate_limit(target_flow, self._q_prev, dt)
        q_cmd = min(q_cmd, rated)
        self._q_prev = q_cmd
        return 0.0 if q_cmd < self.off_threshold_m3s else q_cmd



    def step(
        self,
        state: FanComponentState,
        action: HVACSystemAction,
        timestep: float,
    ) -> "FanComponentAction":
        upstream = getattr(action, "supply_airflow_setpoint_m3s", None)
        want = None if upstream is None else max(0.0, self.gain * float(upstream))
        dt = float(timestep or 0.0)

        if self.ctrl_type == "staged":
            want = float(want or 0.0)
            cmd = self._cmd_staged(want, dt)
        elif self.ctrl_type == "constant":
            cmd = self._cmd_constant(upstream)
        else:
            want = float(want or 0.0)
            cmd = self._cmd_vfd(want, dt)

        la = FanComponentAction(
            component_id=state.system_id,
            component_type=ComponentType.FAN.value
        )
        la.airflow_setpoint_m3s = cmd

        self.logger.debug(f"[{self.name}] mode={self.ctrl_type} rated={self.rated_flow_m3s} cmd={cmd}")
        print(f"[{self.name}] mode={self.ctrl_type} rated={self.rated_flow_m3s} cmd={cmd}")
        return la
