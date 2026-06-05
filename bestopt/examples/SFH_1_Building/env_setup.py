"""
Run simulation and save data for comparison plotting.
Updated for simplified power flow model:
  - battery_power: +charge / -discharge (kW)
  - ev_charging: automatic, always >= 0 (kW)
  - grid_import: net grid (+import, -export) (kW)
  - curtailment: curtailed PV (kW)
"""
from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment
from bestopt.scripts.runtime_plotter import create_hvac_dashboard
from bestopt.scripts.elec_plotter import create_electrical_dashboard
import os
import numpy as np
import pickle
from pathlib import Path

PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

# Load configuration
config_path = os.path.join(PROJECT_ROOT_PATH, "examples", "SFH_1_Building", "config_setup.json")
cm = ConfigurationManager(config_path)
env = BESTOptEnvironment(cm.config)

# Create dashboards
hvac_plotter = create_hvac_dashboard(max_points=96 * 1, window_title="HVAC System Monitor")
electrical_plotter = create_electrical_dashboard(max_points=96 * 1, window_title="Electrical System Monitor")

# ===== Data collection arrays =====
collected = {
    'timesteps': [],
    # Thermal
    'zone_temperature': [],
    'cooling_setpoint': [],
    'heating_setpoint': [],
    'hvac_power': [],               # W
    'hvac_power_kw': [],            # kW
    'hvac_thermal_load': [],        # W
    'supply_air_temp_actual': [],
    'supply_air_temp_setpt': [],
    'supply_air_flow_actual': [],
    'supply_air_flow_setpt': [],
    # Electrical - generation & storage
    'pv_generation': [],            # kW
    'battery_soc': [],              # 0-1
    'ev_tesla_soc': [],             # 0-1
    'ev_nissan_soc': [],            # 0-1
    # Electrical - simplified power flow
    'battery_power': [],            # kW (+charge, -discharge)
    'ev_charging': [],              # kW (always >= 0)
    'grid_import': [],              # kW (+import, -export)
    'net_grid': [],                 # kW (same as grid_import in new model)
    'curtailment': [],              # kW
    'total_demand': [],             # kW (building + EV charging)
    # Load disaggregation
    'total_load': [],               # kW (building_base + HVAC)
    'cooking': [],                  # kW
    'pc': [],                       # kW
    'tv': [],                       # kW
    'lighting': [],                 # kW
    # Signals
    'is_peak': [],
    'electricity_price': [],
}

# ===== Run Simulation =====
for timestep in range(env.total_step):
    observations, done, info = env.step()

    cluster_id = 'residential_cluster_1'
    building_id = 'SFH_1'
    hvac_system_id = env.building_system_map[building_id].get('thermal')
    der_system_id = env.building_system_map[building_id].get('electrical')

    # --- Thermal data ---
    zone_temperature = env.cluster_states[cluster_id].thermal.systems['SFH_1_building'].components['zone0'].temperature
    cooling_setpoint = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].cooling_setpoint_c
    heating_setpoint = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].heating_setpoint_c

    hvac_system = env.system_modules[hvac_system_id]
    der_system = env.system_modules[der_system_id]

    supply_air_temp_real = hvac_system.SAT_actual_C
    supply_air_flow_real = hvac_system.SA_flow_actual_m3s
    HVAC_power = hvac_system.FCU_power_total_W
    HVAC_thermal_load = hvac_system.Q_zone_actual_W
    supply_air_temp_setpt = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].supply_temp_setpoint_c
    supply_air_flow_setpt = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].supply_airflow_setpoint_m3s
    water_flow_real = hvac_system.CHW_flow_actual_m3s
    chiller_supply_water_temp = hvac_system.CHW_supply_temp_C

    # --- Electrical data (simplified power flow) ---
    bat_soc = der_system.battery_states['bat_1'].soc
    ev_tesla_soc = der_system.ev_states['ev_tesla'].soc
    ev_nissan_soc = der_system.ev_states['ev_nissan'].soc
    pv_generation = der_system.pv_states['pv_1'].generation_w / 1000  # kW

    building_base_power = env.cluster_states[cluster_id].electrical.systems['SFH_1_building'].components['electrical'].building_power_w / 1000
    HVAC_power_kw = HVAC_power / 1000
    building_total_load = HVAC_power_kw + building_base_power

    cooking = env.electrical_zone_modules['SFH_1.zone0'].cooking_power / 1000
    pc = env.electrical_zone_modules['SFH_1.zone0'].pc_power / 1000
    tv = env.electrical_zone_modules['SFH_1.zone0'].tv_power / 1000
    lighting = env.electrical_zone_modules['SFH_1.zone0'].lighting_power / 1000

    # New simplified power flow from DERSystemAction
    der_action = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1']

    battery_power_dict = getattr(der_action, 'battery_power', {})
    total_battery_power = sum(battery_power_dict.values())

    ev_charging_dict = getattr(der_action, 'ev_charging', {})
    total_ev_charging = sum(ev_charging_dict.values())

    grid_import = getattr(der_action, 'grid_import', 0.0)
    curtailment_val = getattr(der_action, 'curtailment', 0.0)

    # Total demand = building load + EV charging (matches PowerBalance definition)
    total_demand = building_total_load + total_ev_charging

    is_peak = env.disturbance.prices.peaksignal
    electricity_price = env.disturbance.prices.electricity_price

    # --- Collect data ---
    collected['timesteps'].append(timestep)
    collected['zone_temperature'].append(zone_temperature)
    collected['cooling_setpoint'].append(cooling_setpoint)
    collected['heating_setpoint'].append(heating_setpoint)
    collected['hvac_power'].append(HVAC_power)
    collected['hvac_power_kw'].append(HVAC_power_kw)
    collected['hvac_thermal_load'].append(HVAC_thermal_load)
    collected['supply_air_temp_actual'].append(supply_air_temp_real)
    collected['supply_air_temp_setpt'].append(supply_air_temp_setpt)
    collected['supply_air_flow_actual'].append(supply_air_flow_real)
    collected['supply_air_flow_setpt'].append(supply_air_flow_setpt)
    collected['pv_generation'].append(pv_generation)
    collected['battery_soc'].append(bat_soc)
    collected['ev_tesla_soc'].append(ev_tesla_soc)
    collected['ev_nissan_soc'].append(ev_nissan_soc)
    collected['battery_power'].append(total_battery_power)
    collected['ev_charging'].append(total_ev_charging)
    collected['grid_import'].append(grid_import)
    collected['net_grid'].append(grid_import)  # same in new model
    collected['curtailment'].append(curtailment_val)
    collected['total_demand'].append(total_demand)
    collected['total_load'].append(building_total_load)
    collected['cooking'].append(cooking)
    collected['pc'].append(pc)
    collected['tv'].append(tv)
    collected['lighting'].append(lighting)
    collected['is_peak'].append(is_peak)
    collected['electricity_price'].append(electricity_price)

    print(f"building_load={building_total_load:.3f}, ev={total_ev_charging:.3f}, "
          f"bat={total_battery_power:.3f}, pv={pv_generation:.3f}, "
          f"grid={grid_import:.3f}, "
          f"expected_grid={building_total_load + total_ev_charging + total_battery_power - pv_generation:.3f}")

    # --- Update dashboards ---
    hvac_plotter.add_data_point(
        timestep=timestep,
        zone_temperature=zone_temperature,
        hvac_thermal_load=HVAC_thermal_load,
        hvac_power=HVAC_power,
        supervisory_cooling_setpoint=cooling_setpoint,
        supervisory_heating_setpoint=heating_setpoint,
        supply_air_temp_real=supply_air_temp_real,
        supply_air_temp_setpt=supply_air_temp_setpt,
        supply_air_flow_real=supply_air_flow_real,
        supply_air_flow_setpt=supply_air_flow_setpt,
        water_flow_real=water_flow_real,
        chiller_supply_water_temp=chiller_supply_water_temp
    )

    electrical_plotter.add_data_point(
        timestep=timestep,
        pv_generation_kw=pv_generation,
        battery_soc=bat_soc,
        ev_tesla_soc=ev_tesla_soc,
        ev_nissan_soc=ev_nissan_soc,
        total_load_kw=building_total_load,
        hvac_power_kw=HVAC_power_kw,
        cooking_kw=cooking,
        pc_kw=pc,
        tv_kw=tv,
        lighting_kw=lighting,
        is_peak=is_peak,
        battery_power_kw=total_battery_power,
        ev_charging_kw=total_ev_charging,
        grid_import_kw=grid_import,
        curtailment_kw=curtailment_val,
        electricity_price=electricity_price,
    )

    if timestep % 100 == 0:
        print(f"Step {timestep}/{env.total_step}: "
              f"Temp={zone_temperature:.1f}C, "
              f"PV={pv_generation:.2f}kW, "
              f"Load={building_total_load:.2f}kW, "
              f"Bat={total_battery_power:+.2f}kW, "
              f"EV_chg={total_ev_charging:.2f}kW, "
              f"Grid={grid_import:.2f}kW, "
              f"Curt={curtailment_val:.2f}kW, "
              f"Peak={'Yes' if is_peak else 'No'}")

    if done:
        break

# ===== Save simulation data =====
save_dir = Path("./simulation_results/operation")
save_dir.mkdir(parents=True, exist_ok=True)

# Convert lists to numpy arrays
save_data = {k: np.array(v) for k, v in collected.items()}

# Choose a simulation name (change this per experiment)
SIMULATION_NAME = "baseline_agentic"

save_path = save_dir / f"{SIMULATION_NAME}.pkl"
with open(save_path, 'wb') as f:
    pickle.dump({
        'data': save_data,
        'metadata': {
            'simulation_name': SIMULATION_NAME,
            'total_steps': env.current_step,
            'resolution_s': env.res,
            'duration_s': env.dur,
            'power_flow_model': 'simplified',
            'battery_convention': '+charge/-discharge (kW)',
            'grid_convention': '+import/-export (kW)',
        }
    }, f)
print(f"\nSimulation data saved to: {save_path}")

# Stop dashboards
print("\nStopping dashboards and saving animations...")
hvac_plotter.stop()
# hvac_plotter.save_as_gif('sfh1_hvac_dashboard.gif', fps=20)
electrical_plotter.stop()
electrical_plotter.save_as_gif('sfh1_electrical_dashboard.gif', fps=20)

print(f"\nSimulation completed: {env.current_step} steps")
print("Saved animations: sfh1_hvac_dashboard.gif, sfh1_electrical_dashboard.gif")