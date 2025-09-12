# Simplified water–air coil (heating/cooling) with effectiveness method.
# Writes outlet temperatures IN-PLACE to CoilState. No action is required.

from typing import Dict, Any
from ...core.base import BaseModule
from ...core.data_structure import CoilState, Disturbance  # action is unused here

class CoilModule(BaseModule):
    """
    Standalone coil module.

    Inputs (read from CoilState):
      - airflow_m3s          [m^3/s]
      - waterflow_m3s        [m^3/s]
      - air_inlet_temp_C     [°C]
      - water_inlet_temp_C   [°C]

    Outputs (written in-place to CoilState):
      - air_outlet_temp_C    [°C]
      - water_outlet_temp_C  [°C]

    Model
    - Simplified effectiveness method (placeholder for NTU–ε):
        Q = ε * C_min * (T_hot,in − T_cold,in),  with  C = m_dot * c_p
    - Properties are treated as constants (configurable):
        ρ_air, c_p,air,  ρ_water, c_p,water
    - Limitations: no phase change/condensation, no bypass, ε is constant.

    State
    - Expects a CoilState instance passed as `state`.
    - Reads (and does NOT modify):
        * state.airflow_m3s         [m^3/s]
        * state.waterflow_m3s       [m^3/s]
        * state.air_inlet_temp_C    [°C]
        * state.water_inlet_temp_C  [°C]
    - Writes IN-PLACE:
        * state.air_outlet_temp_C   [°C]
        * state.water_outlet_temp_C [°C]

    Action
    - None required for now (pass `None` is fine). Future versions may consume valve positions or setpoints.
    """

    def __init__(self, config: Dict[str, Any], name: str = "coil"):
        super().__init__(config, name)

        # --- Effectiveness and properties (defaults OK for quick testing) ---
        self.effectiveness: float = float(config.get("effectiveness", 0.7))
        self.effectiveness = max(0.0, min(1.0, self.effectiveness))

        # Air properties (approx. near 20–25°C)
        self.rho_air: float = float(config.get("rho_air", 1.2))         # kg/m^3
        self.cp_air_kJkgK: float = float(config.get("cp_air_kJkgK", 1.005))  # kJ/(kg·K)

        # Water properties (approx. liquid water near room temp)
        self.rho_water: float = float(config.get("rho_water", 997.0))   # kg/m^3
        self.cp_water_kJkgK: float = float(config.get("cp_water_kJkgK", 4.186))  # kJ/(kg·K)

    def initialize(self) -> None:
        self._initialized = True

    def step(
        self,
        state: "CoilState",
        action: Any,                      # not used
        disturbance: "Disturbance",
        timestep: float
    ) -> Dict[str, Any]:
        """Compute outlet temps from inlet temps and flows; write in-place to state."""
        # 0) Read inputs (do NOT change them)
        Va = float(getattr(state, "airflow_m3s", 0.0))       # m^3/s
        Vw = float(getattr(state, "waterflow_m3s", 0.0))     # m^3/s
        Ta_in = float(getattr(state, "air_inlet_temp_C", 0.0))
        Tw_in = float(getattr(state, "water_inlet_temp_C", 0.0))

        # 1) Mass flow and capacity rates (C in kW/K because cp is kJ/kg-K and m_dot is kg/s)
        mdot_air = max(0.0, Va) * self.rho_air
        mdot_wat = max(0.0, Vw) * self.rho_water
        C_air = mdot_air * self.cp_air_kJkgK
        C_wat = mdot_wat * self.cp_water_kJkgK

        # No flow => no heat transfer
        if C_air <= 0.0 or C_wat <= 0.0 or Ta_in == Tw_in:
            state.air_outlet_temp_C = Ta_in
            state.water_outlet_temp_C = Tw_in
            self._record_state({"Ta_out": Ta_in, "Tw_out": Tw_in, "q_kw": 0.0})
            return {}

        # 2) Identify hot/cold sides by inlet temperatures
        if Ta_in >= Tw_in:
            # Typical cooling coil: air is hot, water is cold
            Th_in, Tc_in = Ta_in, Tw_in
            C_hot, C_cold = C_air, C_wat
            hot_side, cold_side = "air", "water"
        else:
            # Heating coil: water is hot, air is cold
            Th_in, Tc_in = Tw_in, Ta_in
            C_hot, C_cold = C_wat, C_air
            hot_side, cold_side = "water", "air"

        C_min = min(C_hot, C_cold)
        dT_in = Th_in - Tc_in
        if dT_in <= 0.0 or C_min <= 0.0:
            # No driving temperature difference or no capacity
            state.air_outlet_temp_C = Ta_in
            state.water_outlet_temp_C = Tw_in
            self._record_state({"Ta_out": Ta_in, "Tw_out": Tw_in, "q_kw": 0.0})
            return {}

        # 3) Effectiveness method (placeholder; can swap to NTU–ε later)
        eps = self.effectiveness
        q_kw = eps * C_min * dT_in  # heat transfer rate [kW], always from hot->cold

        # 4) Outlet temperatures (energy balance)
        Th_out = Th_in - q_kw / C_hot
        Tc_out = Tc_in + q_kw / C_cold

        # 5) Map back to air/water outlets
        if hot_side == "air":
            state.air_outlet_temp_C = Th_out
            state.water_outlet_temp_C = Tc_out
        else:
            state.water_outlet_temp_C = Th_out
            state.air_outlet_temp_C = Tc_out

        # Optional: record snapshot (and may add dynamic attributes if desired)
        self._record_state({
            "Ta_in": Ta_in, "Tw_in": Tw_in,
            "Ta_out": state.air_outlet_temp_C, "Tw_out": state.water_outlet_temp_C,
            "q_kw": q_kw, "eps": eps
        })

    def reset(self) -> None:
        self._state_history.clear()
        self._initialized = False
        self.initialize()
