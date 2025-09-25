import numpy as np
from bestopt.env.modules.hvac.package.FCU import FCU
from bestopt.env.core.data_structure import ThermalAction, Disturbance, WeatherData

# === FCU System Config ===
fcu_config = {
    "fan": {"rated_flow_m3s": 1.2, "rated_power_W": 800},
    "fan_ctrl": {"ctrl_type": "linear"},
    "coil": {"epsilon": 0.85},
    "pump": {"rated_flow_m3s": 0.05, "rated_power_W": 300},
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
fcu = FCU(config=fcu_config)
fcu.initialize()

# === Setup inputs ===
timestep_sec = 900  # 15 minutes
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
    action.thermal_load = -8_000 + 200 * (t % 6)  # Cooling demand (negative)
    action.supervisory_supply_air_temperature = 15.0       # SAT setpoint
    action.supervisory_supply_air_flow_rate = 1.0  # m³/s
    action.chws_temp_c_sp = 7.0              # CHW setpoint
    action.condenser_temp_c_sp = 30.0        # CW setpoint
    action.return_air_temperature = 26.0            # Simulated zone return air

    result = fcu.step(action=action, disturbance=disturbance, timestep=timestep_sec)

    print(
        f"Step {t:02d} | "
        f"Q_zone={result['Q_zone_actual_W']:6.0f} W | "
        f"SAT={result['SAT_actual_C']:4.1f}°C | "
        f"Flow={result['SA_flow_actual_m3s']:.2f} m³/s | "
        f"Power={result['FCU_power_total_W']:6.1f} W | "
        f"E_cum={result['FCU_energy_cumulative_J']:9.1f} J"
    )

# === Summary ===
print("\nFinal cumulative energy consumed (J):", result["FCU_energy_cumulative_J"])
