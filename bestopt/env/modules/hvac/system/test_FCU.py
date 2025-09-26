import numpy as np
from bestopt.env.modules.hvac.system.FCU import FCUModule
from bestopt.env.core.data_structure import ThermalAction, Disturbance, WeatherData, PumpState

# === FCU System Config ===
fcu_config = {
    "fan": {"rated_flow_m3s": 8, "rated_power_W": 8*1000}, #fan_power_per_flow ≈ 1,000 – 1,500 W per m³/s
    "fan_ctrl": {"ctrl_type": "linear"},
    "coil": {"epsilon": 0.8},
    "pump": {"rated_flow_m3s": 0.01, "rated_power_W": 0.05*100_000}, # pump_power_per_flow = 100,000 W per m³/s
    "chiller": {"rated_capacity_W": 100_000, "rated_cop": 5.5},
    "tower": {
        "rated_capacity_W": 120_000,
        "rated_fan_power_W": 2000,
        "pump_power_per_flow": 1800,
        "min_approach_C": 3.0,
        "max_approach_C": 7.0
    }
}

# === Instantiate FCU system ===
fcu = FCUModule(config=fcu_config)
fcu.initialize()

# === Setup inputs ===
timestep_sec = 3600  # 60 minutes
n_steps = 24

# Fake weather & disturbance
weather = WeatherData()
weather.outdoor_wet_bulb_temperature = 24.0
disturbance = Disturbance(weather=weather)

# === Simulate 24 steps ===
print("=== FCU 24-step Simulation ===")
for t in range(n_steps):
    action = ThermalAction()
    
    # === Fake Supervisory Controller Outputs ===
    action.thermal_load = -8_0000 + 2000 * (t % 6)  # Cooling demand (negative)
    action.supervisory_supply_air_temperature = 13.0       # SAT setpoint
    action.supervisory_supply_air_flow_rate = abs(action.thermal_load)/1005/1.225/13 # m³/s
    action.chws_temp_c_sp = 7.0              # CHW setpoint
    action.condenser_temp_c_sp = 30.0        # CW setpoint
    action.return_air_temperature = 25.0 + 1 * (t % 6)           # Simulated zone return air

    result = fcu.step(state.thermal, action=action, disturbance=disturbance, timestep=timestep_sec)

    print(
        f"Step {t:02d} | "
        f"Q_set={action.thermal_load:6.0f} W | "
        f"Q_actual={result['Q_zone_actual_W']:6.0f} W | "
        f"SAT_set={action.supervisory_supply_air_temperature:4.1f}°C | "
        f"SAT_actual={result['SAT_actual_C']:4.1f}°C | "
        f"Air_flow={result['SA_flow_actual_m3s']:.2f} m³/s | "
        f"CHW_flow={result['CHW_flow_actual_m3s']:.5f} m³/s | "
        f"Power={result['FCU_power_total_W']:6.0f} W | "
        f"Sys_COP={abs(result['Q_zone_actual_W'])/result['FCU_power_total_W']:4.2f} | "
        f"E_cum={result['FCU_energy_cumulative_J']/1000/3600:4.1f} kWh"
    )

# === Summary ===
print("\nFinal cumulative energy consumed (J):", result["FCU_energy_cumulative_J"])
