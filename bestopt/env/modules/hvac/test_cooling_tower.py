from bestopt.env.modules.hvac.cooling_tower import CoolingTowerModule
from bestopt.env.core.data_structure import CoolingTowerState, ThermalAction

cfg = {
    "rated_capacity_W": 200_000.0,  # 200 kW
    "rated_fan_power_W": 5000.0,
    "min_approach_C": 3.0,
    "max_approach_C": 7.0
}

tower = CoolingTowerModule(cfg, name="tower1")
tower.initialize()

state = CoolingTowerState(component_id="tower1", component_type="", domain="")

action = ThermalAction(cooling_tower_load_W_sp=180000.0, wet_bulb_temp_c=24.0)

timestep_sec = 900

tower.step(state=state, action=action, timestep=timestep_sec)

print("=== Cooling Tower State ===")
print(f"Heat Rejected (W):  {state.heat_rejected_W:.2f}")
print(f"Outlet Temp (°C):   {state.outlet_temp_c:.2f}")
print(f"Fan Power (W):      {state.fan_power_W:.2f}")
print(f"Energy (J):         {state.energy_J_cum:.2f}")