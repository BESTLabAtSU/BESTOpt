from ..component.chiller import ChillerModule
from ....core.data_structure import ChillerState, ThermalAction, PumpState, CoilState

# Configuration for chiller
cfg = {
    "eta_carnot": 0.4,
    "min_cop": 2.0,
    "max_cop": 10.0,
    "min_chws_temp_c": 5.0,
    "max_chws_temp_c": 10.0,
    "enable_history": True
}

# Instantiate the chiller
chiller = ChillerModule(cfg, name="main_chiller")
chiller.initialize()

# Create state instances
cs = ChillerState(component_id="main_chiller", component_type="", domain="")
coil_state = CoilState()
pump_state = PumpState()

# Simulated schedules
chw_return_schedule = [12.0, 12.5, 13.0, 12.0, 11.5, 11.0,
                       10.5, 10.0, 9.5, 9.0, 8.5, 8.0,
                       8.0, 8.5, 9.0, 9.5, 10.0, 10.5,
                       11.0, 11.5, 12.0, 12.5, 13.0, 12.5]  # [°C]

chw_flow_schedule = [0.02] * 24  # [m³/s]
chws_sp_schedule = [7.0] * 24  # [°C]
cond_temp_schedule = [30.0] * 24  # [°C]

# Timestep: 15 minutes
dt_sec = 900

# Run for 24 timesteps
print("=== 24-Step Chiller Simulation ===")
for t in range(24):
    coil_state.water_outlet_temp_C = chw_return_schedule[t]
    pump_state.waterflow_m3s = chw_flow_schedule[t]

    act = ThermalAction(
        chws_temp_c_sp=chws_sp_schedule[t],
        condenser_temp_c_sp=cond_temp_schedule[t]
    )

    chiller.step(cs, action=act, coil_state=coil_state, pump_state=pump_state, timestep=dt_sec)

    print(f"Step {t:02d} | Q={cs.cooling_W:.0f} W | COP={cs.cop:.2f} | CHW out={cs.chws_temp_c:.1f}°C | "
          f"Flow={cs.chw_flow_m3s:.3f} m³/s | P={cs.power_W:.1f} W | E_cum={cs.energy_J_cum:.1f} J")

# Summary
print("\nTotal energy consumed (J):", round(cs.energy_J_cum, 2))
