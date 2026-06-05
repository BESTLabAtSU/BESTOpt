"""
Configuration Generator for Demand Response Flexibility Study
=============================================================
5 Cases × 20 Buildings:
  Case 1 (Baseline):     No precooling, no PV/battery, NO EVs, COP=3.0
  Case 2 (Pre-cooling):  2h precooling @ 2°C offset, no PV/battery, no EVs, COP=3.0
  Case 3 (HP Retrofit):  No precooling, no PV/battery, no EVs, COP=4.5
  Case 4 (EV):           No precooling, no PV/battery, 1 EV (60 kWh), COP=3.0
  Case 5 (PV+Battery):   No precooling, 5kW PV + 20kWh battery, no EVs, COP=3.0

KEY DESIGN:
  - Baseline has NO EVs so each case adds exactly one DR measure.
  - Occupancy disturbance gets a fixed seed (occupancy_seed=42) so ALL
    five cases produce the exact same leave/back schedule.
  - Building params are deterministic per building index (GLOBAL_SEED).
  - Base setpoints (base_cooling / base_heating) are stored per building
    for fair comfort evaluation_old across all cases (including precooling).
"""

import logging
import os
import random
from bestopt.env.core.config_manager import ConfigurationManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

# ===========================
# STUDY PARAMETERS
# ===========================

N_BUILDINGS = 20
GLOBAL_SEED = 42
OCCUPANCY_SEED = 42

SIM_START_TIME = "2023-08-01 00:00:00"
RESOLUTION_S = 900
DURATION_1DAY = 86400
DURATION_2WEEKS = 86400 * 14

# ---------- Case definitions ----------
CASE_DEFINITIONS = {
    "baseline": {
        "label": "Baseline",
        "precooling_config": None,
        "chiller_cop": 3.0,
        "install_pv_battery": False,
        "pv_capacity_kW": 0,
        "battery_capacity_kWh": 0,
        "install_ev": False,
        "ev_capacity_kWh": 0,
    },
    "precooling": {
        "label": "Pre-cooling (2h, 2°C)",
        "precooling_config": {"degree": 2, "hours": 2},
        "chiller_cop": 3.0,
        "install_pv_battery": False,
        "pv_capacity_kW": 0,
        "battery_capacity_kWh": 0,
        "install_ev": False,
        "ev_capacity_kWh": 0,
    },
    "hp_retrofit": {
        "label": "HP Retrofit (COP 4.5)",
        "precooling_config": None,
        "chiller_cop": 4.5,
        "install_pv_battery": False,
        "pv_capacity_kW": 0,
        "battery_capacity_kWh": 0,
        "install_ev": False,
        "ev_capacity_kWh": 0,
    },
    "ev": {
        "label": "EV (60 kWh)",
        "precooling_config": None,
        "chiller_cop": 3.0,
        "install_pv_battery": False,
        "pv_capacity_kW": 0,
        "battery_capacity_kWh": 0,
        "install_ev": True,
        "ev_capacity_kWh": 60,
    },
    "pv_battery": {
        "label": "PV+Battery (5kW/20kWh)",
        "precooling_config": None,
        "chiller_cop": 3.0,
        "install_pv_battery": True,
        "pv_capacity_kW": 5,
        "battery_capacity_kWh": 20,
        "install_ev": False,
        "ev_capacity_kWh": 0,
    },
}

CASE_ORDER = ["baseline", "precooling", "hp_retrofit", "ev", "pv_battery"]

# ---------- Per-building randomisation ----------
BASE_PARAMS = {
    "hvac": {
        "base_cooling": 24.0,
        "base_heating": 18.0,
        "cooling_power_max": 4000.0,
        "heating_power_max": 4000.0,
        "chiller_capacity": 3500,
    },
    "electrical": {
        "lighting_daytime": 600,
        "lighting_nighttime": 1800,
        "appliance_cooking": 2000,
        "appliance_tv": 200,
        "appliance_pc": 400,
        "appliance_dishwashing": 1000,
    },
}

VARIATION_RANGES = {
    "hvac": {
        "base_cooling":      {"min": -2, "max": 2},
        "base_heating":      {"min": -2, "max": 2},
        "cooling_power_max": {"min": -1000, "max": 1000},
        "heating_power_max": {"min": -1000, "max": 1000},
        "chiller_capacity":  {"min": -1000, "max": 1000},
    },
    "electrical": {
        "lighting_daytime":      {"min": -0.30, "max": 0.30, "mode": "percent"},
        "lighting_nighttime":    {"min": -0.30, "max": 0.30, "mode": "percent"},
        "appliance_cooking":     {"min": -0.20, "max": 0.20, "mode": "percent"},
        "appliance_tv":          {"min": -0.20, "max": 0.20, "mode": "percent"},
        "appliance_pc":          {"min": -0.20, "max": 0.20, "mode": "percent"},
        "appliance_dishwashing": {"min": -0.20, "max": 0.20, "mode": "percent"},
    },
}


# ===========================
# HELPERS
# ===========================

def _vary(base_value, vrange, rng, clamp=None):
    mode = vrange.get("mode", "abs")
    delta = rng.uniform(vrange["min"], vrange["max"])
    varied = base_value * (1.0 + delta) if mode == "percent" else base_value + delta
    if clamp:
        varied = max(clamp[0], min(clamp[1], varied))
    return round(varied, 2)


def _generate_building_params(building_index: int):
    """Deterministic per-building params — identical across all 5 cases."""
    rng = random.Random(GLOBAL_SEED + building_index * 100)
    hvac = {k: _vary(v, VARIATION_RANGES["hvac"][k], rng)
            for k, v in BASE_PARAMS["hvac"].items()}
    elec = {k: _vary(v, VARIATION_RANGES["electrical"][k], rng)
            for k, v in BASE_PARAMS["electrical"].items()}
    return hvac, elec


def get_all_building_base_setpoints():
    """
    Return {building_id: (base_cooling, base_heating)} for fair comfort
    evaluation_old. These are the ORIGINAL setpoints BEFORE any precooling offset.
    """
    setpoints = {}
    for i in range(1, N_BUILDINGS + 1):
        hvac_p, _ = _generate_building_params(i)
        setpoints[f"SFH_{i}"] = (hvac_p["base_cooling"], hvac_p["base_heating"])
    return setpoints


# ===========================
# BUILDING CONFIGURATOR
# ===========================

def _add_building(cm, building_id, cluster_id, case_def, hvac_p, elec_p):
    cop = case_def["chiller_cop"]

    cm.add_building(
        cluster_id=cluster_id,
        building_id=building_id,
        parameters={
            "building_type": "single_family_home",
            "building_number": int(building_id.split("_")[1]),
        },
        thermal_zones=["zone0"],
        electrical_zones=["zone0"],
    )

    cm.add_thermal_zone_module(
        building_id=building_id,
        zone_id="zone0",
        parameters={
            "model_args": {
                "para": {"Int_h": 8, "Ext_h": 14, "epochs": 20},
                "modeltype": "PI-modnn",
                "startday": 1, "trainday": 180, "testday": 1,
                "datapath": os.path.join(PROJECT_ROOT_PATH, "data", "SFH",
                                         "BLDG", "clean", "SFH_1.csv"),
                "temp_unit": "C", "device": "cuda:0", "save_name": "SFH_1",
            },
            "model_path": os.path.join(
                PROJECT_ROOT_PATH, "examples", "Saved", "SFH_1",
                "Trained_mdlEnco48_Deco96", "PI-modnn_180daysTest_on07-01.pth"),
            "scaler_path": os.path.join(
                PROJECT_ROOT_PATH, "examples", "Scaler", "SFH_1",
                "ModNN_scaler.pkl"),
            "historical_data_path": os.path.join(
                PROJECT_ROOT_PATH, "data", "SFH", "BLDG", "clean", "SFH_1.csv"),
            "encoder_length": 48,
            "retrain": "Off",
            "simulation_start_time": SIM_START_TIME,
        },
        class_path="bestopt.env.modules.building.thermalzone.ThermalDynamicsModule",
    )

    cm.add_electrical_zone_module(
        building_id=building_id,
        zone_id="zone0",
        parameters={
            "lighting": {
                "daytime": elec_p["lighting_daytime"],
                "nighttime": elec_p["lighting_nighttime"],
            },
            "appliance": {
                "cooking": elec_p["appliance_cooking"],
                "tv": elec_p["appliance_tv"],
                "pc": elec_p["appliance_pc"],
                "dishwashing": elec_p["appliance_dishwashing"],
            },
        },
        class_path="bestopt.env.modules.building.electricalzone.ElectricalDynamicModule",
    )

    # --- HVAC system ---
    hvac_sys_id = f"hvac_system_{building_id}"
    cm.add_system(
        cluster_id=cluster_id,
        system_id=hvac_sys_id,
        system_type="hvac_systems",
        parameters={
            "system_name": f"FCU System for {building_id}",
            "system_config": {
                "fan": {"rated_flow_m3s": 1, "rated_power_W": 1000},
                "fan_ctrl": {"ctrl_type": "linear"},
                "coil": {"epsilon": 0.8},
                "pump": {"rated_flow_m3s": 0.005, "rated_power_W": 500},
                "chiller": {
                    "rated_capacity_W": hvac_p["chiller_capacity"],
                    "rated_cop": cop,
                },
                "tower": {
                    "rated_capacity_W": hvac_p["chiller_capacity"],
                    "rated_fan_power_W": 2000,
                    "pump_power_per_flow": 1800,
                    "min_approach_C": 3.0,
                    "max_approach_C": 7.0,
                },
            },
        },
        class_path="bestopt.env.modules.hvac.system.FCU.FCUModule",
    )

    # --- DER system (EV and/or PV+Battery, or empty) ---
    der_sys_id = f"der_system_{building_id}"
    der_config = {
        "system_name": f"DER System for {building_id}",
        "system_config": {},
    }

    # EV — only for case 4
    if case_def["install_ev"]:
        der_config["system_config"]["evs"] = [
            {
                "id": f"ev_{building_id}",
                "rated_capacity_kWh": case_def["ev_capacity_kWh"],
                "initial_soc": 0.3,
                "charge_speed": 0.25,
                "discharge_speed": 0.5,
                "charge_efficiency": 0.95,
                "initially_connected": True,
            },
        ]

    # PV + Battery — only for case 5
    if case_def["install_pv_battery"]:
        der_config["system_config"]["pv"] = {
            "rated_capacity_kW": case_def["pv_capacity_kW"],
        }
        der_config["system_config"]["bat"] = {
            "rated_capacity_kWh": case_def["battery_capacity_kWh"],
            "initial_soc": 0.3,
            "charge_speed": 0.25,
            "discharge_speed": 0.5,
            "charge_efficiency": 0.95,
        }

    cm.add_system(
        cluster_id=cluster_id,
        system_id=der_sys_id,
        system_type="der_systems",
        parameters=der_config,
        class_path="bestopt.env.modules.ders.system.der.DERModule",
    )

    cm.assign_system_to_buildings(hvac_sys_id, [building_id])
    cm.assign_system_to_buildings(der_sys_id, [building_id])

    # --- HVAC controller ---
    hvac_ctrl_id = f"hvac_controller_{building_id}"
    hvac_ctrl_params = {
        "domain": "thermal",
        "type": "rule-based",
        "mode": "cooling",
        "base_cooling": hvac_p["base_cooling"],
        "base_heating": hvac_p["base_heating"],
        "deadband": 0.5,
        "cooling_power_max": hvac_p["cooling_power_max"],
        "heating_power_max": hvac_p["heating_power_max"],
    }
    if case_def["precooling_config"] is not None:
        hvac_ctrl_params["precooling"] = case_def["precooling_config"]

    cm.add_system_controller(
        controller_id=hvac_ctrl_id,
        system_id=hvac_sys_id,
        parameters=hvac_ctrl_params,
        class_path="bestopt.env.controllers.thermal.SupervisoryController",
    )

    # --- DER controller ---
    der_ctrl_id = f"der_controller_{building_id}"
    der_ctrl_params = {
        "domain": "electrical",
        "type": "rule-based",
        "mode": "self_consumption",
        "max_grid_import": 10000,
        "max_grid_export": 5000,
        "system_config": {},
    }

    if case_def["install_ev"]:
        der_ctrl_params["ev_soc_min"] = 0.2
        der_ctrl_params["ev_soc_target"] = 0.8
        der_ctrl_params["ev_v2g_enabled"] = True
        der_ctrl_params["system_config"]["ev"] = {
            "rated_capacity_kWh": case_def["ev_capacity_kWh"],
            "initial_soc": 0.3,
        }

    if case_def["install_pv_battery"]:
        der_ctrl_params["bat_soc_min"] = 0.1
        der_ctrl_params["bat_soc_max"] = 0.9
        der_ctrl_params["system_config"]["pv"] = {
            "rated_capacity_kW": case_def["pv_capacity_kW"],
        }
        der_ctrl_params["system_config"]["bat"] = {
            "rated_capacity_kWh": case_def["battery_capacity_kWh"],
            "initial_soc": 0.3,
        }

    cm.add_system_controller(
        controller_id=der_ctrl_id,
        system_id=der_sys_id,
        parameters=der_ctrl_params,
        class_path="bestopt.env.controllers.electrical.SupervisoryController",
    )

    return hvac_sys_id, der_sys_id, hvac_ctrl_id, der_ctrl_id


# ===========================
# FULL CASE BUILDER
# ===========================

def build_case_config(case_name: str, duration_s: int = DURATION_1DAY) -> str:
    case_def = CASE_DEFINITIONS[case_name]
    logging.info(f"Building config for case '{case_name}': {case_def['label']}")

    cm = ConfigurationManager()
    cluster_id = "residential_cluster"
    cm.add_cluster(cluster_id, parameters={"location": "Syracuse, NY"})

    all_buildings, all_systems = [], []
    building_configs = {}

    for i in range(1, N_BUILDINGS + 1):
        building_id = f"SFH_{i}"
        hvac_p, elec_p = _generate_building_params(i)
        hvac_sys, der_sys, hvac_ctrl, der_ctrl = _add_building(
            cm, building_id, cluster_id, case_def, hvac_p, elec_p)
        all_buildings.append(building_id)
        all_systems.extend([hvac_sys, der_sys])
        building_configs[building_id] = {
            "hvac_system": hvac_sys, "der_system": der_sys,
            "hvac_controller": hvac_ctrl, "der_controller": der_ctrl,
        }

    # ================================================================
    # DISTURBANCES
    # ================================================================
    cm.add_disturbance("weather", parameters={
        "file_path": os.path.join(PROJECT_ROOT_PATH, "data", "SFH",
                                  "DIST", "weather", "weather.csv"),
        "simulation_start_time": SIM_START_TIME,
    }, class_path="bestopt.env.disturbances.weather.WeatherModule")

    cm.add_disturbance("occupancy", parameters={
        "occupancy_seed": OCCUPANCY_SEED,
    }, class_path="bestopt.env.disturbances.occupancy.OccupancyModule")

    cm.add_disturbance("price", parameters={},
        class_path="bestopt.env.disturbances.price.PriceModule")

    # Environment
    cm.add_environment(parameters={
        "resolution": RESOLUTION_S,
        "duration": duration_s,
        "enable_history": True,
        "logging_level": "INFO",
        "simulation_start_time": SIM_START_TIME,
    }, class_path="bestopt.environment.BESTOptEnvironment")

    # Selections
    cm.select_cluster(cluster_id)
    cm.select_buildings(all_buildings)
    cm.select_systems(all_systems)
    for bid, cfg in building_configs.items():
        cm.select_controller_for_system(cfg["hvac_system"], cfg["hvac_controller"])
        cm.select_controller_for_system(cfg["der_system"], cfg["der_controller"])
    cm.select_disturbances(["weather", "occupancy", "price"])
    cm.select_environment()

    warnings = cm.validate_configuration()
    if warnings:
        logging.warning(f"Case '{case_name}' warnings: {warnings}")

    output_dir = os.path.join(PROJECT_ROOT_PATH, "examples", "DR_Flexibility_Study")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"config_{case_name}_{N_BUILDINGS}bldg.json")
    cm.save_final_configuration(output_file)
    logging.info(f"Saved: {output_file}")
    return output_file


def main():
    configs = {}
    for duration_tag, duration_s in [("1day", DURATION_1DAY),
                                      ("2weeks", DURATION_2WEEKS)]:
        for case_name in CASE_ORDER:
            key = f"{case_name}_{duration_tag}"
            path = build_case_config(case_name, duration_s)
            configs[key] = path
            print(f"  ✓ {key:30s} → {path}")
    print(f"\n✓ All {len(configs)} configs generated.")
    return configs


if __name__ == "__main__":
    main()