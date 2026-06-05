"""
Simulation Runner for Demand Response Flexibility Study
=======================================================
5 Cases:
  1. Baseline     — no EV, no PV/battery, COP 3.0
  2. Pre-cooling  — 2h/2°C precooling offset
  3. HP Retrofit  — COP 4.5
  4. EV           — 1 EV (60 kWh)
  5. PV+Battery   — 5 kW PV + 20 kWh battery

KEY: Each building record also stores `base_cooling_setpoint` and
     `base_heating_setpoint` (the original, un-offset values) so that
     comfort can be evaluated fairly across all cases.

Supports multiprocessing: cases run in parallel across available cores.
"""

import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from multiprocessing import Pool, cpu_count

from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment

# Import the config module to get base setpoints
from config_setup import (
    get_all_building_base_setpoints, CASE_ORDER, N_BUILDINGS,
)

PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

BUILDING_IDS = [f"SFH_{i}" for i in range(1, N_BUILDINGS + 1)]
CLUSTER_ID = "residential_cluster"

CASE_NAMES = CASE_ORDER


# ===========================
# DATA COLLECTOR
# ===========================

def _empty_building_record():
    """Return a fresh dict for one building's time-series."""
    return {
        "timestamps": [],
        # thermal
        "zone_temperature": [],
        "cooling_setpoint": [],           # actual (may include precooling offset)
        "heating_setpoint": [],           # actual (may include precooling offset)
        "base_cooling_setpoint": [],      # original base — for fair comfort eval
        "base_heating_setpoint": [],      # original base — for fair comfort eval
        "supply_air_temp_actual": [],
        "supply_air_flow_actual": [],
        # HVAC power
        "hvac_power_kw": [],
        "hvac_thermal_load_kw": [],
        # electrical loads (disaggregated)
        "lighting_kw": [],
        "cooking_kw": [],
        "pc_kw": [],
        "tv_kw": [],
        "building_load_kw": [],
        # DER
        "pv_generation_kw": [],
        "battery_soc": [],
        "battery_power_kw": [],
        "ev_charging_kw": [],
        "ev_soc": [],
        # grid
        "grid_import_kw": [],
        "curtailment_kw": [],
        "total_demand_kw": [],
        # price
        "electricity_price": [],
        "is_peak": [],
    }


def run_single_case(config_path: str, base_setpoints: dict,
                    max_steps: int = None, verbose: bool = True):
    """
    Run one simulation case for all buildings.

    Parameters
    ----------
    config_path : str
    base_setpoints : dict
        {building_id: (base_cooling, base_heating)} from config generator.
    max_steps : int, optional
    verbose : bool
        If True, print per-step progress (may interleave in parallel mode).

    Returns
    -------
    building_data : dict[building_id → time-series dict]
    """
    cm = ConfigurationManager(config_path)
    env = BESTOptEnvironment(cm.config)
    total_steps = max_steps if max_steps else env.total_step

    building_data = {bid: _empty_building_record() for bid in BUILDING_IDS}

    if verbose:
        print(f"  Running {total_steps} steps …")

    for step in range(total_steps):
        observations, done, info = env.step()

        current_time = (
            datetime.strptime(env.simulation_start_time, "%Y-%m-%d %H:%M:%S")
            + timedelta(seconds=step * env.res)
        )

        for building_id in BUILDING_IDS:
            hvac_sys_id = f"hvac_system_{building_id}"
            der_sys_id  = f"der_system_{building_id}"
            bldg_sys_id = f"{building_id}_building"

            rec = building_data[building_id]
            rec["timestamps"].append(current_time)

            # Base setpoints (constant, from config — for fair comfort eval)
            base_cool, base_heat = base_setpoints[building_id]
            rec["base_cooling_setpoint"].append(base_cool)
            rec["base_heating_setpoint"].append(base_heat)

            # ============================================================
            # THERMAL DATA
            # ============================================================
            bldg_state = env.cluster_states[CLUSTER_ID].thermal.systems.get(
                bldg_sys_id)
            zone_temp = 20.0
            if bldg_state and "zone0" in bldg_state.components:
                zone_temp = bldg_state.components["zone0"].temperature
            rec["zone_temperature"].append(zone_temp)

            # Actual setpoints (may include precooling offset)
            hvac_action = env.cluster_actions[CLUSTER_ID].thermal \
                .system_actions.get(hvac_sys_id)
            if hvac_action:
                rec["cooling_setpoint"].append(hvac_action.cooling_setpoint_c)
                rec["heating_setpoint"].append(hvac_action.heating_setpoint_c)
            else:
                rec["cooling_setpoint"].append(base_cool)
                rec["heating_setpoint"].append(base_heat)

            # HVAC module actuals
            hvac_mod = env.system_modules.get(hvac_sys_id)
            hvac_power = 0.0
            hvac_thermal = 0.0
            sat_actual = zone_temp
            saf_actual = 0.0
            if hvac_mod:
                hvac_power   = getattr(hvac_mod, "FCU_power_total_W", 0.0)
                hvac_thermal = getattr(hvac_mod, "Q_zone_actual_W", 0.0)
                sat_actual   = getattr(hvac_mod, "SAT_actual_C", zone_temp)
                saf_actual   = getattr(hvac_mod, "SA_flow_actual_m3s", 0.0)

            rec["hvac_power_kw"].append(hvac_power / 1000)
            rec["hvac_thermal_load_kw"].append(hvac_thermal / 1000)
            rec["supply_air_temp_actual"].append(sat_actual)
            rec["supply_air_flow_actual"].append(saf_actual)

            # ============================================================
            # ELECTRICAL LOADS
            # ============================================================
            zone_key = f"{building_id}.zone0"
            cooking = pc = tv = lighting = 0.0
            if zone_key in env.electrical_zone_modules:
                em = env.electrical_zone_modules[zone_key]
                cooking  = getattr(em, "cooking_power", 0.0)
                pc       = getattr(em, "pc_power", 0.0)
                tv       = getattr(em, "tv_power", 0.0)
                lighting = getattr(em, "lighting_power", 0.0)

            rec["lighting_kw"].append(lighting / 1000)
            rec["cooking_kw"].append(cooking / 1000)
            rec["pc_kw"].append(pc / 1000)
            rec["tv_kw"].append(tv / 1000)

            elec_state = env.cluster_states[CLUSTER_ID].electrical.systems.get(
                bldg_sys_id)
            building_base_kw = 0.0
            if elec_state:
                ec = elec_state.components.get("electrical")
                if ec:
                    building_base_kw = getattr(ec, "building_power_w", 0.0) / 1000

            building_load_kw = hvac_power / 1000 + building_base_kw
            rec["building_load_kw"].append(building_load_kw)

            # ============================================================
            # DER DATA
            # ============================================================
            der_mod = env.system_modules.get(der_sys_id)

            pv_gen_kw = 0.0
            if der_mod and hasattr(der_mod, "pv_states"):
                for pv_state in der_mod.pv_states.values():
                    pv_gen_kw += getattr(pv_state, "generation_w", 0.0) / 1000
            rec["pv_generation_kw"].append(pv_gen_kw)

            bat_soc = 0.0
            if der_mod and hasattr(der_mod, "battery_states"):
                for bs in der_mod.battery_states.values():
                    bat_soc = getattr(bs, "soc", 0.0)
                    break
            rec["battery_soc"].append(bat_soc)

            ev_soc = 0.0
            if der_mod and hasattr(der_mod, "ev_states"):
                for ev_id, ev_state in der_mod.ev_states.items():
                    ev_soc = getattr(ev_state, "soc", 0.0)
                    break
            rec["ev_soc"].append(ev_soc)

            # ============================================================
            # DER ACTION
            # ============================================================
            der_action = env.cluster_actions[CLUSTER_ID].electrical \
                .system_actions.get(der_sys_id)

            bat_power_dict = getattr(der_action, "battery_power", {}) \
                if der_action else {}
            total_bat_power = sum(bat_power_dict.values()) \
                if bat_power_dict else 0.0
            rec["battery_power_kw"].append(total_bat_power)

            ev_charge_dict = getattr(der_action, "ev_charging", {}) \
                if der_action else {}
            total_ev_charging = sum(ev_charge_dict.values()) \
                if ev_charge_dict else 0.0
            rec["ev_charging_kw"].append(total_ev_charging)

            grid_import = getattr(der_action, "grid_import", 0.0) \
                if der_action else 0.0
            rec["grid_import_kw"].append(grid_import)

            curtailment = getattr(der_action, "curtailment", 0.0) \
                if der_action else 0.0
            rec["curtailment_kw"].append(curtailment)

            total_demand = building_load_kw + total_ev_charging
            rec["total_demand_kw"].append(total_demand)

            # ============================================================
            # PRICE SIGNAL
            # ============================================================
            rec["electricity_price"].append(
                getattr(env.disturbance.prices, "electricity_price", 12.0))
            rec["is_peak"].append(
                getattr(env.disturbance.prices, "peaksignal", False))

        # progress
        if verbose and step % 24 == 0:
            avg_t = np.mean([building_data[b]["zone_temperature"][-1]
                             for b in BUILDING_IDS])
            tot_grid = sum(building_data[b]["grid_import_kw"][-1]
                           for b in BUILDING_IDS)
            print(f"    step {step:5d}/{total_steps} "
                  f"| avg T={avg_t:.1f}°C | cluster grid={tot_grid:.1f} kW")

        if done:
            break

    if verbose:
        print(f"  ✓ completed {step+1} steps")
    return building_data


# ===========================
# MULTIPROCESSING WORKER
# ===========================

def _run_case_worker(args):
    """
    Top-level worker function for multiprocessing.
    Must be at module level (not nested) so it can be pickled.
    """
    case_name, config_path, base_setpoints, max_steps, verbose = args
    print(f"\n{'='*60}")
    print(f"CASE: {case_name} [PID {os.getpid()}]")
    print(f"{'='*60}")
    data = run_single_case(config_path, base_setpoints,
                           max_steps=max_steps, verbose=verbose)
    return case_name, data


# ===========================
# BATCH RUNNER
# ===========================

def run_all_cases(duration_tag: str = "1day", max_steps: int = None,
                  parallel: bool = True, n_workers: int = None,
                  verbose: bool = True):
    """
    Run all 5 cases, optionally in parallel.

    Parameters
    ----------
    duration_tag : str
        Label used in the output pickle filename.
    max_steps : int, optional
        Override total simulation steps.
    parallel : bool
        If True (default), run cases in parallel using multiprocessing.
    n_workers : int, optional
        Number of worker processes. Defaults to min(num_cases, cpu_count).
    verbose : bool
        Print per-step progress within each case.

    Returns
    -------
    all_results : dict[case_name → dict[building_id → time-series dict]]
    """
    study_dir = os.path.join(PROJECT_ROOT_PATH, "examples",
                             "DR_Flexibility_Study")
    results_dir = os.path.join(study_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    base_setpoints = get_all_building_base_setpoints()

    # Build task list — skip missing configs
    tasks = []
    for case_name in CASE_NAMES:
        config_path = os.path.join(
            study_dir, f"config_{case_name}_{N_BUILDINGS}bldg.json")
        if not os.path.exists(config_path):
            print(f"  ⚠ Config not found for '{case_name}', skipping. "
                  f"Run config_study_5cases.py first.")
            continue
        tasks.append((case_name, config_path, base_setpoints,
                      max_steps, verbose))

    if not tasks:
        print("  ✗ No valid configs found. Nothing to run.")
        return {}

    # ------------------------------------------------------------------
    # PARALLEL execution
    # ------------------------------------------------------------------
    if parallel and len(tasks) > 1:
        n_workers = n_workers or min(len(tasks), cpu_count())
        print(f"\n⚡ Running {len(tasks)} cases in parallel "
              f"({n_workers} workers, {cpu_count()} CPUs available)")

        with Pool(processes=n_workers) as pool:
            results_list = pool.map(_run_case_worker, tasks)

        all_results = dict(results_list)

    # ------------------------------------------------------------------
    # SEQUENTIAL execution (fallback / single case)
    # ------------------------------------------------------------------
    else:
        if not parallel:
            print(f"\n▶ Running {len(tasks)} cases sequentially")
        all_results = {}
        for task in tasks:
            case_name, data = _run_case_worker(task)
            all_results[case_name] = data

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    pkl_path = os.path.join(results_dir, f"results_{duration_tag}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n✓ Results saved to {pkl_path}")

    return all_results


# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    # --- 1-day run (96 steps @ 15 min) ---
    print("=" * 70)
    print("PHASE 1: 1-Day Simulation (all 5 cases)")
    print("=" * 70)
    results_1day = run_all_cases(
        "1day", max_steps=96,
        parallel=True,   # ← set False to disable multiprocessing
        verbose=True,     # ← set False to suppress per-step logging
    )

    # --- 2-week run (1344 steps @ 15 min) ---
    print("\n" + "=" * 70)
    print("PHASE 2: 2-Week Simulation (all 5 cases)")
    print("=" * 70)
    results_2weeks = run_all_cases(
        "2weeks", max_steps=96 * 14,
        parallel=True,
        verbose=True,
    )

    print("\n✓ All simulations complete.")