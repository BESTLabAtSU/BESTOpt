# bestopt/env/modules/hvac/test_heatpump.py

from bestopt.env.modules.hvac.component.heatpump import HeatPumpModule
from bestopt.env.core.data_structure import HeatPumpState, Disturbance

def test_heatpump():
    config = {"capacity_W": 4000, "cop": 3.5}
    hp = HeatPumpModule(config)
    state = HeatPumpState(
        source_inlet_temp_C=10.0,
        sink_inlet_temp_C=30.0,
        source_flow_m3s=0.05,
        sink_flow_m3s=0.05,
        sink_outlet_temp_set_C=35.0,
    )
    disturbance = Disturbance()
    dt = 3600  # 1 hour

    hp.step({}, state, disturbance, dt)

    print("Sink outlet temp [°C]:", state.sink_outlet_temp_C)
    print("Source outlet temp [°C]:", state.source_outlet_temp_C)
    print("Thermal power [W]:", state.thermal_power_W)
    print("Electric power [W]:", state.elec_power_W)
    print("Cumulative energy [kWh]:", state.energy_J_cum / 3.6e6)

if __name__ == "__main__":
    test_heatpump()
