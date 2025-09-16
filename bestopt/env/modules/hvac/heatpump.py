# bestopt/env/modules/hvac/heatpump.py
"""
Heat Pump module (two-port, source & sink).
"""

from typing import Dict, Any
from ...core.base import BaseModule
from ...core.data_structure import HeatPumpState, Disturbance


class HeatPumpModule(BaseModule):
    """
    Heat Pump module that exchanges heat between source and sink streams.
    Updates a HeatPumpState in place.

    Inputs (from HeatPumpState):
        - source_inlet_temp_C
        - sink_inlet_temp_C
        - source_flow_m3s
        - sink_flow_m3s
        - sink_outlet_temp_set_C

    Outputs:
        - source_outlet_temp_C
        - sink_outlet_temp_C
        - thermal_power_W
        - elec_power_W
        - energy_J_cum

    Model:
        Q_demand = m_sink * cp * (T_set − T_sink_in)
        Q_actual = clamp(Q_demand, −capacity, +capacity)
        P_el = |Q_actual| / COP
    """

    def __init__(self, config: Dict[str, Any], name: str = "heat_pump"):
        super().__init__(config, name)
        self.capacity = config.get("capacity_W", 5000.0)   # [W]
        self.cop = config.get("cop", 3.5)                  # constant COP
        self.cp = config.get("cp", 4180.0)                 # J/kg-K (default water)
        self.rho = config.get("rho", 1000.0)               # kg/m³ (default water)

    def step(
        self,
        action: Dict[str, Any],        # not used here
        state: HeatPumpState,
        disturbance: Disturbance,
        dt: float,
    ):
        # Mass flow rates [kg/s]
        m_sink = state.sink_flow_m3s * self.rho
        m_source = state.source_flow_m3s * self.rho

        # Demand based on setpoint
        Q_demand = m_sink * self.cp * (state.sink_outlet_temp_set_C - state.sink_inlet_temp_C)
        Q_actual = max(-self.capacity, min(self.capacity, Q_demand))

        # Electrical consumption
        elec_power = abs(Q_actual) / self.cop if self.cop > 0 else 0.0

        # Update sink outlet temperature
        if m_sink > 0:
            state.sink_outlet_temp_C = state.sink_inlet_temp_C + Q_actual / (m_sink * self.cp)
        else:
            state.sink_outlet_temp_C = state.sink_inlet_temp_C

        # Update source outlet temperature
        if m_source > 0:
            state.source_outlet_temp_C = state.source_inlet_temp_C - Q_actual / (m_source * self.cp)
        else:
            state.source_outlet_temp_C = state.source_inlet_temp_C

        # Update powers
        state.thermal_power_W = Q_actual
        state.elec_power_W = elec_power
        state.energy_J_cum += elec_power * dt

    def reset(self, state: HeatPumpState):
        """
        Reset dynamic/cumulative variables.
        """
        state.source_outlet_temp_C = 0.0
        state.sink_outlet_temp_C = 0.0
        state.thermal_power_W = 0.0
        state.elec_power_W = 0.0
        state.energy_J_cum = 0.0
        
    def initialize(self, state: HeatPumpState):
        """
        Initialize state (required by BaseModule).
        """
        self.reset(state)