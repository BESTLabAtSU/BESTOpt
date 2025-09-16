from bestopt.env.modules.hvac.chiller import ChillerModule
from bestopt.env.core.data_structure import ChillerState, ThermalAction, Disturbance

# Configuration for chiller
cfg = {
    "rated_capacity_kw": 150.0,
    "eta_carnot": 0.4,
    "min_plr": 0.15,
    "max_plr": 1.03,
    "min_cop": 2.0,
    "max_cop": 10.0,
    "min_chws_temp_c": 5.0,
    "max_chws_temp_c": 10.0,
    "enable_history": True
}

# Instantiate the ChillerModule
chiller = ChillerModule(cfg, name="main_chiller")
chiller.initialize()

# Create a standalone ChillerState
cs = ChillerState()

# Create a thermal action and disturbance
act = ThermalAction(chiller_cooling_kw_sp=80.0, chws_temp_c_sp=7.0)
dst = Disturbance(condenser_temp_c=34.0)

dt_sec = 900  # 15 minutes per step

# Single step test
chiller.step(state=cs, action=act, disturbance=dst, timestep=dt_sec)

print("\n=== Single Step ===")
print(f"Cooling (kW):       {cs.cooling_kw:.2f}")
print(f"COP:                {cs.cop:.2f}")
print(f"CHW Temp (°C):      {cs.chws_temp_c:.2f}")
print(f"CHW Flow (m³/s):    {cs.chw_flow_m3s:.4f}")
print(f"Power (W):          {cs.power_W:.2f}")
print(f"Energy (J):         {cs.energy_J_cum:.2f}")

# Schedule of loads and CHWS setpoints
cooling_schedule = [80, 100, 60, 120, 0, 90]
setpoint_schedule = [7.0, 7.5, 6.5, 6.0, 7.0, 7.0]

print("\n=== Multi-Step Simulation ===")
for t, (q_kw, t_chws) in enumerate(zip(cooling_schedule, setpoint_schedule)):
    act = ThermalAction(chiller_cooling_kw_sp=q_kw, chws_temp_c_sp=t_chws)
    dst = Disturbance(condenser_temp_c=34.0)
    chiller.step(state=cs, action=act, disturbance=dst, timestep=dt_sec)
    print(f"Step {t:02d}: Q={cs.cooling_kw:.1f} kW, T_chws={cs.chws_temp_c:.1f} °C, "
          f"flow={cs.chw_flow_m3s:.4f} m³/s, COP={cs.cop:.2f}, P={cs.power_W:.1f} W, "
          f"E_cum={cs.energy_J_cum:.1f} J")

print("\nTotal energy consumed (J):", round(cs.energy_J_cum, 2))
