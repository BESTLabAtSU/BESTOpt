from bestopt.env.modules.hvac.chiller import ChillerModule
from bestopt.env.core.data_structure import ChillerState, ThermalAction

cfg = {
    "rated_capacity_W": 150_000.0,  # 150 kW
    "eta_carnot": 0.4,
    "min_plr": 0.15,
    "max_plr": 1.03,
    "min_cop": 2.0,
    "max_cop": 10.0,
    "min_chws_temp_c": 5.0,
    "max_chws_temp_c": 10.0,
    "enable_history": True
}

chiller = ChillerModule(cfg, name="main_chiller")
chiller.initialize()

state = ChillerState(component_id="main_chiller", component_type="", domain="")

act = ThermalAction(
    chiller_cooling_W_sp=80_000,
    chws_temp_c_sp=7.0,
    condenser_temp_c_sp=25.0
)

dt_sec = 900

chiller.step(state=state, action=act, timestep=dt_sec)

print("\n=== Single Step ===")
print(f"Cooling (W):        {state.cooling_W:.0f}")
print(f"COP:                {state.cop:.2f}")
print(f"CHW Temp (°C):      {state.chws_temp_c:.2f}")
print(f"CHW Flow (m³/s):    {state.chw_flow_m3s:.4f}")
print(f"Power (W):          {state.power_W:.1f}")
print(f"Energy (J):         {state.energy_J_cum:.1f}")

print("\n=== Multi-Step Simulation ===")
cooling_schedule = [80_000, 100_000, 60_000, 120_000, 0, 200_000]
setpoint_schedule = [7.0, 7.5, 6.5, 6.0, 7.0, 7.0]

for t, (q_W, t_chws) in enumerate(zip(cooling_schedule, setpoint_schedule)):
    act = ThermalAction(chiller_cooling_W_sp=q_W, chws_temp_c_sp=t_chws, condenser_temp_c_sp=30.0)
    chiller.step(state=state, action=act, timestep=dt_sec)
    print(f"Step {t:02d}: Q={state.cooling_W:.0f} W, T_chws={state.chws_temp_c:.1f} °C, "
          f"flow={state.chw_flow_m3s:.4f} m³/s, COP={state.cop:.2f}, P={state.power_W:.0f} W, "
          f"E_cum={state.energy_J_cum:.1f} J")

print("\nTotal energy consumed (J):", round(state.energy_J_cum, 2))