from bestopt.env.modules.hvac.fan import FanModule
from bestopt.env.core.data_structure import FanState, ThermalAction, Disturbance

cfg = {"rated_flow": 2.0, "rated_power_kw": 4.0, "power_exponent": 3.0, "enable_history": True}
fan = FanModule(cfg, name="supply_fan")
fan.initialize()

# create a standalone FanState (not stored under ThermalDomainState)
fs = FanState(component_id="supply_fan", component_type="", domain="")  # __post_init__ will set domain/type

# create an action with a setpoint
act = ThermalAction(supplyfan_flow_sp=1.2)


dt_sec = 900  # 15 minutes / step

# one step
fan.step(state=fs, action=act, disturbance=Disturbance(), timestep=900)

print(fs.flow_m3s)         # 1.2
print(fs.power_kw)         # 4.0 * (1.2/2.0)**3
print(fs.energy_kwh_cum)   # power_kw * 0.25



# Many steps

flow_schedule = [1.0, 1.2, 0.8, 1.5, 1.0, 1.2, 0.8, 1.5]


# 4) Loop over steps
for t, q in enumerate(flow_schedule):
    act = ThermalAction(supplyfan_flow_sp=q)
    fan.step(state=fs, action=act, disturbance=Disturbance(), timestep=dt_sec)
    print(f"Step {t:02d}: flow={fs.flow_m3s:.2f} m^3/s, power={fs.power_kw:.3f} kW, cumE={fs.energy_kwh_cum:.3f} kWh")

print("\nTotal energy (kWh):", round(fs.energy_kwh_cum, 3))



