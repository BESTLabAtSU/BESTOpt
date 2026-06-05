"""
Gymnasium-compatible wrapper for BESTOptEnvironment.

Time-aware dynamic reward weights

Key change from v6:
  Instead of static weights, comfort and HVAC energy weights vary based on
  occupancy status and proximity to peak hours. This creates a clear reward
  landscape where precooling emerges naturally.

  COMFORT VIOLATION weight:
    - Occupied:   HIGH   → must maintain comfort
    - Unoccupied: LOW    → mild penalty, allow drift

  HVAC ENERGY weight (4 zones):
    - Peak hours:              HIGH   → avoid expensive cooling
    - Pre-peak window (2h):    LOW    → allow precooling
    - Other + occupied:        BASE   → normal operation
    - Other + unoccupied:      HIGH   → don't waste energy on empty building

  The precooling behavior emerges from the LOW→HIGH transition:
  the agent learns to front-load cooling in the cheap window before
  peak makes it expensive.

Action space (Box, shape=2, [-1, 1]):
  setpoint mode:
    [0] cooling_setpoint_offset  → [-3, +3] °C from occupancy-based base
    [1] battery_power            → [-max, +max] kW
  airflow mode:
    [0] supply_airflow_fraction  → [0, 1] (mapped from [-1, 1])
    [1] battery_power            → [-max, +max] kW

Observation space (Box, shape=16):
    [0]  zone_temperature / 50
    [1]  outdoor_temperature / 50
    [2]  solar_irradiance / 1000
    [3]  electricity_price / 100
    [4]  peak_signal             (0 or 1)
    [5]  occupancy               [0, 1]
    [6]  battery_soc             [0, 1]
    [7]  pv_generation / 10      kW
    [8]  hvac_power / 10         kW
    [9]  net_grid / 10           kW
    [10] time_of_day_sin
    [11] time_of_day_cos
    [12] base_cooling_sp / 30
    [13] hours_until_peak / 6
    [14] comfort_margin / 5
    [15] is_unoccupied           (1=unoccupied, 0=occupied)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple

from bestopt.env.core.environment import BESTOptEnvironment
from bestopt.env.core.data_structure import (
    ClusterAction, HVACSystemAction, DERSystemAction,
    SystemType, DomainType
)


class BESTOptGymEnv(gym.Env):

    metadata = {"render_modes": []}

    # === Setpoint offset range (setpoint mode) ===
    SP_OFFSET_MAX = 3.0        # °C
    COOLING_SP_ABS_MIN = 20.0  # safety floor
    COOLING_SP_ABS_MAX = 28.0  # safety ceiling

    # === Airflow range (airflow mode) ===
    AIRFLOW_FRAC_MIN = 0.0     # off
    AIRFLOW_FRAC_MAX = 1.0     # full capacity
    FIXED_SUPPLY_TEMP = 13.0   # °C, fixed supply air temperature

    BAT_POWER_MAX_KW = 5.0

    def __init__(
        self,
        config: Dict[str, Any],
        cluster_id: str = "residential_cluster_1",
        building_id: str = "SFH_1",
        hvac_system_id: str = "hvac_system_1",
        der_system_id: str = "der_system_1",
        reward_weights: Optional[Dict[str, float]] = None,
        num_days: Optional[int] = None,
        randomize_start: bool = False,
        valid_start_range: Optional[Tuple[str, str]] = None,
        control_mode: str = "setpoint",
    ):
        super().__init__()

        # --- Control mode ---
        assert control_mode in ("setpoint", "airflow"), \
            f"control_mode must be 'setpoint' or 'airflow', got '{control_mode}'"
        self.control_mode = control_mode

        if num_days is not None:
            config['environment']['parameters']['duration'] = num_days * 24 * 3600

        self.env = BESTOptEnvironment(config)

        # Episode randomization
        self.randomize_start = randomize_start
        self.num_days = num_days or 1
        self.default_start = config['environment']['parameters'].get(
            'simulation_start_time', '2023-08-01 00:00:00'
        )

        if valid_start_range:
            self._start_min = pd.Timestamp(valid_start_range[0])
            self._start_max = pd.Timestamp(valid_start_range[1])
        else:
            self._start_min = pd.Timestamp("2023-07-01")
            self._start_max = pd.Timestamp("2023-09-01")

        self._resolution = config['environment']['parameters'].get('resolution', 900)
        self._steps_per_day = 86400 // self._resolution

        # Identifiers
        self.cluster_id = cluster_id
        self.building_id = building_id
        self.hvac_system_id = hvac_system_id
        self.der_system_id = der_system_id
        self.building_system_id = f"{building_id}_building"

        # Read base setpoints from thermal controller config
        ctrl_cfg = self._get_thermal_controller_config()
        self.base_cooling = ctrl_cfg.get("base_cooling", 24.0)
        self.base_heating = ctrl_cfg.get("base_heating", 18.0)
        self.precool_config = ctrl_cfg.get("precooling", None)

        # ------------------------------------------------------------------
        # Reward weights — dynamic weights with configurable levels
        # ------------------------------------------------------------------
        rw = reward_weights or {}

        # Comfort violation weights (occupied vs unoccupied)
        self.w_comfort_occupied = rw.get("comfort_occupied", 10.0)
        self.w_comfort_unoccupied = rw.get("comfort_unoccupied", 0.5)

        # HVAC energy weights (4 zones)
        self.w_hvac_peak = rw.get("hvac_peak", 5.0)         # during peak hours
        self.w_hvac_prepeak = rw.get("hvac_prepeak", 0.3)    # 2h before peak (allow precooling)
        self.w_hvac_base = rw.get("hvac_base", 1.0)          # normal occupied
        self.w_hvac_unoccupied = rw.get("hvac_unoccupied", 3.0)  # unoccupied non-prepeak

        # Pre-peak window duration (hours)
        self.prepeak_hours = rw.get("prepeak_hours", 2.0)

        # Other weights (static, same as before)
        self.w_grid = rw.get("grid", 1.0)
        self.w_curtail = rw.get("curtail", 1.5)

        # Comfort deadband (unchanged from v6)
        self.comfort_deadband = rw.get("comfort_deadband", 1.0)

        self._init_battery_limits()

        # State tracking
        self._current_cooling_sp = self.base_cooling
        self._base_cooling_sp = self.base_cooling
        self._current_airflow_frac = 0.0

        # ------------------------------------------------------------------
        # Spaces: 2 actions, 16 observations
        # ------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        obs_low  = np.array([0, -0.5, 0, 0, 0, 0, 0, 0, 0,  0, -1, -1, 0, 0, -1, 0], dtype=np.float32)
        obs_high = np.array([1,  1.0, 1.5, 1, 1, 1, 1, 1.5, 1.5, 1.5, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._last_reward_breakdown = {}
        self._obs_values = {}

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if self.randomize_start:
            start_time = self._pick_random_start()
            self._update_start_time(start_time)
        else:
            self._update_start_time(self.default_start)

        self.env.reset()
        self._current_cooling_sp = self.base_cooling
        self._current_airflow_frac = 0.0

        # One idle step to populate states
        self.env.step()
        obs = self._extract_observation()
        return obs, {"start_time": str(self.env.simulation_start_time)}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        external_actions = self._build_external_actions(action)
        obs_dict, done, info = self.env.step(external_actions=external_actions)
        obs = self._extract_observation()
        reward = self._compute_reward()
        info["reward_breakdown"] = self._last_reward_breakdown
        info["obs_values"] = self._obs_values
        return obs, reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass

    def _pick_random_start(self) -> str:
        total_range_days = (self._start_max - self._start_min).days
        max_offset_days = max(0, total_range_days - self.num_days)
        offset = self.np_random.integers(0, max_offset_days + 1) if max_offset_days > 0 else 0
        start = self._start_min + pd.Timedelta(days=int(offset))
        return str(start)

    def _update_start_time(self, new_start_time: str):
        self.env.simulation_start_time = new_start_time
        weather_mod = self.env.disturbance_modules.get('weather')
        if weather_mod and hasattr(weather_mod, 'update_start_time'):
            weather_mod.update_start_time(new_start_time)

    # ------------------------------------------------------------------
    # Base setpoint logic (mirrors thermal SupervisoryController)
    # ------------------------------------------------------------------

    def _get_base_cooling_setpoint(self) -> float:
        disturbance = self.env.disturbance
        occupancy = disturbance.occupancy.occupancy_fraction
        step_of_day = disturbance.occupancy.step_of_day

        if occupancy > 0.0:
            cooling_sp = self.base_cooling
        else:
            cooling_sp = self.base_cooling + 2.0

        if self.precool_config:
            pre_degree = self.precool_config.get("degree", 2.0)
            pre_hour = self.precool_config.get("hours", 2)
            peak_start = disturbance.prices.peak_start
            if peak_start - pre_hour * 4 <= step_of_day < peak_start:
                cooling_sp = self.base_cooling - pre_degree

        return cooling_sp

    def _get_hours_until_peak(self) -> float:
        prices = self.env.disturbance.prices
        is_peak = getattr(prices, 'peaksignal', False)

        if is_peak:
            return 0.0

        step_of_day = self.env.disturbance.occupancy.step_of_day
        peak_start = getattr(prices, 'peak_start', 52)

        steps_until = peak_start - step_of_day
        if steps_until < 0:
            steps_per_day = self._steps_per_day
            steps_until = steps_per_day - step_of_day + peak_start

        hours_until = steps_until * (self._resolution / 3600)
        return min(hours_until, 6.0)

    def _is_prepeak(self) -> bool:
        """Check if current time is within the pre-peak window."""
        hours_until_peak = self._get_hours_until_peak()
        is_peak = self.env.disturbance.prices.peaksignal

        # Pre-peak = not yet peak, but within prepeak_hours window
        if is_peak:
            return False
        return hours_until_peak <= self.prepeak_hours

    # ------------------------------------------------------------------
    # Action translation (unchanged from v6)
    # ------------------------------------------------------------------

    def _build_external_actions(self, action: np.ndarray) -> Dict[str, ClusterAction]:
        action = np.clip(action, -1.0, 1.0).astype(float)

        base_cool = self._get_base_cooling_setpoint()
        self._base_cooling_sp = base_cool

        cluster_action = ClusterAction(cluster_id=self.cluster_id)

        hvac_action = HVACSystemAction(
            system_id=self.hvac_system_id,
            system_type=SystemType.HVAC.value
        )

        if self.control_mode == "setpoint":
            cooling_sp = base_cool + action[0] * self.SP_OFFSET_MAX
            cooling_sp = np.clip(cooling_sp, self.COOLING_SP_ABS_MIN, self.COOLING_SP_ABS_MAX)
            self._current_cooling_sp = cooling_sp

            hvac_action.cooling_setpoint_c = cooling_sp
            hvac_action.heating_setpoint_c = self.base_heating
            hvac_action.supply_airflow_setpoint_m3s = None
            hvac_action.supply_temp_setpoint_c = None

        elif self.control_mode == "airflow":
            airflow_frac = (action[0] + 1.0) / 2.0
            airflow_frac = np.clip(airflow_frac, self.AIRFLOW_FRAC_MIN, self.AIRFLOW_FRAC_MAX)
            self._current_airflow_frac = airflow_frac

            hvac_action.cooling_setpoint_c = base_cool
            hvac_action.heating_setpoint_c = self.base_heating
            hvac_action.supply_airflow_setpoint_m3s = airflow_frac
            hvac_action.supply_temp_setpoint_c = self.FIXED_SUPPLY_TEMP

            self._current_cooling_sp = base_cool

        # Battery action
        bat_power_kw = action[1] * self.BAT_POWER_MAX_KW

        cluster_action.thermal.system_actions[self.hvac_system_id] = hvac_action

        der_action = DERSystemAction(
            system_id=self.der_system_id,
            system_type=SystemType.DER.value
        )
        der_action.battery_power = {"bat_1": bat_power_kw}
        cluster_action.electrical.system_actions[self.der_system_id] = der_action

        return {self.cluster_id: cluster_action}

    # ------------------------------------------------------------------
    # Observation (16 dims, same as v6)
    # ------------------------------------------------------------------

    def _extract_observation(self) -> np.ndarray:
        cs = self.env.cluster_states[self.cluster_id]
        der_mod = self.env.system_modules[self.der_system_id]
        hvac_mod = self.env.system_modules[self.hvac_system_id]
        weather = self.env.disturbance.weather
        prices = self.env.disturbance.prices
        occupancy = self.env.disturbance.occupancy

        # Raw values
        zone_temp = cs.thermal.systems[self.building_system_id].components['zone0'].temperature
        assert zone_temp is not None, "zone_temp is None"

        outdoor_temp = weather.outdoor_dry_bulb_temp
        solar_irradiance = weather.solar_radiation_w_m2
        electricity_price = prices.electricity_price
        is_peak = float(prices.peaksignal)
        occ_fraction = occupancy.occupancy_fraction

        bat_soc = der_mod.battery_states['bat_1'].soc
        pv_gen_kw = der_mod.pv_states['pv_1'].generation_w / 1000.0

        hvac_power_kw = hvac_mod.FCU_power_total_W / 1000.0
        elec_comp = cs.electrical.systems[self.building_system_id].components['electrical']
        base_load_kw = elec_comp.building_power_w / 1000.0
        total_load_kw = hvac_power_kw + base_load_kw

        bat_power_kw = der_mod.battery_states['bat_1'].power_w / 1000.0
        net_grid_kw = max(0.0, total_load_kw + bat_power_kw - pv_gen_kw)
        curtailment_kw = max(0.0, pv_gen_kw - total_load_kw - bat_power_kw)

        time_of_day = (self.env.current_step * self.env.res / 3600) % 24
        time_sin = np.sin(2 * np.pi * time_of_day / 24)
        time_cos = np.cos(2 * np.pi * time_of_day / 24)

        base_cool = self._get_base_cooling_setpoint()
        hours_until_peak = self._get_hours_until_peak()

        # Comfort margin
        if occ_fraction > 0.0:
            comfort_ceiling = base_cool + self.comfort_deadband
            comfort_margin = comfort_ceiling - zone_temp
        else:
            comfort_margin = 5.0

        is_unoccupied = 1.0 if occ_fraction <= 0.0 else 0.0

        # 16-dim normalized observation
        obs = np.array([
            zone_temp / 50.0,               # [0]
            outdoor_temp / 50.0,             # [1]
            solar_irradiance / 1000.0,       # [2]
            electricity_price / 100.0,       # [3]
            is_peak,                         # [4]
            occ_fraction,                    # [5]
            bat_soc,                         # [6]
            pv_gen_kw / 10.0,               # [7]
            hvac_power_kw / 10.0,            # [8]
            net_grid_kw / 10.0,              # [9]
            time_sin,                        # [10]
            time_cos,                        # [11]
            base_cool / 30.0,                # [12]
            hours_until_peak / 6.0,          # [13]
            comfort_margin / 5.0,            # [14]
            is_unoccupied,                   # [15]
        ], dtype=np.float32)

        # Store raw values
        self._obs_values = {
            "zone_temp": float(zone_temp),
            "outdoor_temp": float(outdoor_temp),
            "solar_irradiance": float(solar_irradiance),
            "electricity_price": float(electricity_price),
            "is_peak": float(is_peak),
            "occupancy": float(occ_fraction),
            "is_unoccupied": float(is_unoccupied),
            "bat_soc": float(bat_soc),
            "pv_gen_kw": float(pv_gen_kw),
            "base_load_kw": float(base_load_kw),
            "hvac_power_kw": float(hvac_power_kw),
            "total_load_kw": float(total_load_kw),
            "net_grid_kw": float(net_grid_kw),
            "curtailment_kw": float(curtailment_kw),
            "bat_power_kw": float(bat_power_kw),
            "time_of_day": float(time_of_day),
            "hours_until_peak": float(hours_until_peak),
            "base_cooling_sp": float(base_cool),
            "actual_cooling_sp": float(self._current_cooling_sp),
            "comfort_margin": float(comfort_margin),
            "comfort_ceiling": float(base_cool + self.comfort_deadband) if occ_fraction > 0 else None,
            "airflow_frac": float(self._current_airflow_frac) if self.control_mode == "airflow" else None,
            "is_prepeak": float(self._is_prepeak()),
        }

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ------------------------------------------------------------------
    # Reward — time-aware dynamic weights
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        """
        Reward with dynamic weights based on occupancy and peak timing.

        Comfort weight:
            occupied   → w_comfort_occupied   (high, e.g. 10.0)
            unoccupied → w_comfort_unoccupied  (low,  e.g. 0.5)

        HVAC energy weight (4 zones):
            peak hours              → w_hvac_peak        (high, e.g. 5.0)
            pre-peak window (2h)    → w_hvac_prepeak     (low,  e.g. 0.3)
            other + occupied        → w_hvac_base        (base, e.g. 1.0)
            other + unoccupied      → w_hvac_unoccupied  (high, e.g. 3.0)

        Precooling emerges naturally: the low energy weight in pre-peak
        followed by high weight during peak creates an incentive to
        front-load cooling before peak starts.
        """
        d = self._obs_values
        dt_h = self.env.res / 3600.0

        is_peak = d["is_peak"] > 0.5
        is_occupied = d["occupancy"] > 0.0
        is_prepeak = d["is_prepeak"] > 0.5

        # =============================================================
        # 1. HVAC energy cost — dynamic weight
        # =============================================================
        hvac_kw = d["hvac_power_kw"]
        r_hvac = hvac_kw * dt_h  # raw energy in kWh

        if is_peak:
            w_hvac_dynamic = self.w_hvac_peak
        elif is_prepeak:
            w_hvac_dynamic = self.w_hvac_prepeak
        elif is_occupied:
            w_hvac_dynamic = self.w_hvac_base
        else:
            # Unoccupied, not pre-peak → high penalty
            w_hvac_dynamic = self.w_hvac_unoccupied

        # =============================================================
        # 2. Comfort violation — dynamic weight
        # =============================================================
        zone_temp = d["zone_temp"]

        cooling_limit = d["base_cooling_sp"] + self.comfort_deadband
        heating_limit = self.base_heating - self.comfort_deadband

        if zone_temp > cooling_limit:
            violation = zone_temp - cooling_limit
            r_comfort = violation ** 2
        elif zone_temp < heating_limit:
            violation = heating_limit - zone_temp
            r_comfort = violation ** 2
        else:
            r_comfort = 0.0

        if is_occupied:
            w_comfort_dynamic = self.w_comfort_occupied
        else:
            w_comfort_dynamic = self.w_comfort_unoccupied

        # =============================================================
        # 3. Grid cost (static weight)
        # =============================================================
        net_grid_kw = d["net_grid_kw"]
        price_per_kwh = d["electricity_price"] / 100.0
        r_grid = net_grid_kw * price_per_kwh * dt_h

        # =============================================================
        # 4. PV curtailment (static weight)
        # =============================================================
        curtailment_kw = d["curtailment_kw"]
        r_curtail = curtailment_kw * dt_h

        # =============================================================
        # Total reward
        # =============================================================
        reward = (
            - w_hvac_dynamic * r_hvac
            - w_comfort_dynamic * r_comfort
            - self.w_grid * r_grid
            - self.w_curtail * r_curtail
        )

        self._last_reward_breakdown = {
            "r_hvac": r_hvac,
            "r_comfort": r_comfort,
            "r_grid": r_grid,
            "r_curtail": r_curtail,
            "w_hvac_dynamic": w_hvac_dynamic,
            "w_comfort_dynamic": w_comfort_dynamic,
            "hvac_zone": (
                "peak" if is_peak else
                "prepeak" if is_prepeak else
                "occupied" if is_occupied else
                "unoccupied"
            ),
            "net_grid_kw": net_grid_kw,
            "hvac_power_kw": hvac_kw,
            "curtailment_kw": curtailment_kw,
            "zone_temp": zone_temp,
            "cooling_ceiling": cooling_limit,
            "heating_floor": heating_limit,
            "comfort_deadband": self.comfort_deadband,
            "is_unoccupied": d["is_unoccupied"],
            "is_prepeak": d["is_prepeak"],
            "reward": reward,
        }

        return float(reward)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_thermal_controller_config(self) -> Dict:
        for ctrl_id, ctrl_cfg in self.env.controllers_config.items():
            if ctrl_cfg.get('system_id') == self.hvac_system_id:
                return ctrl_cfg.get('parameters', {})
        return {}

    def _init_battery_limits(self):
        try:
            sys_cfg = self.env.systems_config.get(self.der_system_id, {})
            params = sys_cfg.get("parameters", {})
            bat_cfg = params.get("system_config", {}).get("batteries", {})
            if isinstance(bat_cfg, dict):
                for v in bat_cfg.values():
                    cap = v.get("capacity_kwh", 5.0)
                    c_rate = v.get("charge_speed", 0.5)
                    self.BAT_POWER_MAX_KW = cap * c_rate
                    break
            elif isinstance(bat_cfg, list) and bat_cfg:
                cap = bat_cfg[0].get("capacity_kwh", 5.0)
                c_rate = bat_cfg[0].get("charge_speed", 0.5)
                self.BAT_POWER_MAX_KW = cap * c_rate
        except Exception:
            pass