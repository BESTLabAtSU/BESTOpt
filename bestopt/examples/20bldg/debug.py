"""
Diagnostic Runner — Compare SFH_1 across all 4 cases for first 96 steps
========================================================================
Seeds np.random + torch before each case to ensure determinism.
Prints side-by-side table + automated checks for COP, precooling, PV.
"""

import os
import numpy as np
import random as stdlib_random
from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.environment import BESTOptEnvironment

PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

CLUSTER_ID = "residential_cluster"
BUILDING_ID = "SFH_1"
HVAC_SYS_ID = "hvac_system_SFH_1"
DER_SYS_ID  = "der_system_SFH_1"
BLDG_SYS_ID = "SFH_1_building"
N_BUILDINGS = 20
STEPS = 96

CASE_NAMES = ["baseline", "precooling", "hp_retrofit", "pv_battery"]
GLOBAL_SEED = 42


def _seed_all(seed):
    np.random.seed(seed)
    stdlib_random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_diagnostic(case_name):
    _seed_all(GLOBAL_SEED)

    study_dir = os.path.join(PROJECT_ROOT_PATH, "examples", "DR_Flexibility_Study")
    config_path = os.path.join(study_dir, f"config_{case_name}_{N_BUILDINGS}bldg.json")

    cm = ConfigurationManager(config_path)
    env = BESTOptEnvironment(cm.config)

    records = []
    for step in range(min(STEPS, env.total_step)):
        observations, done, info = env.step()

        bldg_state = env.cluster_states[CLUSTER_ID].thermal.systems.get(BLDG_SYS_ID)
        zone_temp = bldg_state.components["zone0"].temperature if bldg_state else 20.0

        hvac_action = env.cluster_actions[CLUSTER_ID].thermal.system_actions.get(HVAC_SYS_ID)
        cool_sp = hvac_action.cooling_setpoint_c if hvac_action else 24.0
        heat_sp = hvac_action.heating_setpoint_c if hvac_action else 18.0
        sa_flow = getattr(hvac_action, "supply_airflow_setpoint_m3s", 0.0) if hvac_action else 0.0
        sa_temp = getattr(hvac_action, "supply_temp_setpoint_c", 0.0) if hvac_action else 0.0

        hvac_mod = env.system_modules.get(HVAC_SYS_ID)
        fcu_power_w  = getattr(hvac_mod, "FCU_power_total_W", 0.0) if hvac_mod else 0.0
        q_zone_w     = getattr(hvac_mod, "Q_zone_actual_W", 0.0) if hvac_mod else 0.0
        sat_actual   = getattr(hvac_mod, "SAT_actual_C", 0.0) if hvac_mod else 0.0
        sa_flow_act  = getattr(hvac_mod, "SA_flow_actual_m3s", 0.0) if hvac_mod else 0.0

        chiller_cop = None
        if hvac_mod:
            chiller_cop = (
                getattr(hvac_mod, "chiller_cop", None) or
                getattr(hvac_mod, "rated_cop", None) or
                getattr(hvac_mod, "COP", None)
            )

        der_action = env.cluster_actions[CLUSTER_ID].electrical.system_actions.get(DER_SYS_ID)
        grid_import  = getattr(der_action, "grid_import", 0.0) if der_action else 0.0
        curtailment  = getattr(der_action, "curtailment", 0.0) if der_action else 0.0
        bat_power    = sum(getattr(der_action, "battery_power", {}).values()) if der_action else 0.0
        ev_charging  = sum(getattr(der_action, "ev_charging", {}).values()) if der_action else 0.0

        der_mod = env.system_modules.get(DER_SYS_ID)
        pv_kw = 0.0
        if der_mod and hasattr(der_mod, "pv_states"):
            for ps in der_mod.pv_states.values():
                pv_kw += getattr(ps, "generation_w", 0.0) / 1000

        is_peak = getattr(env.disturbance.prices, "peaksignal", False)
        price   = getattr(env.disturbance.prices, "electricity_price", 12.0)
        occupancy = getattr(env.disturbance.occupancy, "occupancy_fraction", 1.0)
        step_of_day = getattr(env.disturbance.occupancy, "step_of_day", step % 96)

        records.append({
            "step": step, "step_of_day": step_of_day,
            "zone_T": zone_temp, "cool_sp": cool_sp, "heat_sp": heat_sp,
            "sa_flow_sp": sa_flow, "sa_temp_sp": sa_temp,
            "sa_flow_act": sa_flow_act, "sat_actual": sat_actual,
            "fcu_power_kw": fcu_power_w / 1000, "q_zone_kw": q_zone_w / 1000,
            "chiller_cop": chiller_cop,
            "pv_kw": pv_kw, "bat_kw": bat_power, "ev_kw": ev_charging,
            "grid_kw": grid_import, "curtail_kw": curtailment,
            "is_peak": is_peak, "price": price, "occupancy": occupancy,
        })
        if done:
            break

    return records


def print_comparison(all_records):
    cases = list(all_records.keys())

    # ---- Determinism check: baseline vs hp_retrofit occupancy ----
    print(f"\n{'='*100}")
    print("CHECK 0: Cross-case determinism (occupancy schedule)")
    print(f"{'='*100}")
    if "baseline" in all_records and "hp_retrofit" in all_records:
        bl_occ = [r["occupancy"] for r in all_records["baseline"]]
        hp_occ = [r["occupancy"] for r in all_records["hp_retrofit"]]
        if bl_occ == hp_occ:
            print("  ✓ Occupancy schedule is IDENTICAL between baseline and hp_retrofit")
        else:
            diffs = sum(1 for a, b in zip(bl_occ, hp_occ) if a != b)
            print(f"  ✗ Occupancy DIFFERS at {diffs}/{len(bl_occ)} steps!")
            print(f"    → The occupancy module is not seeded properly.")
            print(f"    → Apply the occupancy_patched.py fix.")

    # ---- Determinism check: baseline vs hp_retrofit setpoints ----
    # (should be identical if occupancy + controller are deterministic)
    if "baseline" in all_records and "hp_retrofit" in all_records:
        bl_cool = [r["cool_sp"] for r in all_records["baseline"]]
        hp_cool = [r["cool_sp"] for r in all_records["hp_retrofit"]]
        if bl_cool == hp_cool:
            print("  ✓ Cooling setpoints are IDENTICAL between baseline and hp_retrofit")
        else:
            diffs = sum(1 for a, b in zip(bl_cool, hp_cool) if abs(a - b) > 0.01)
            print(f"  ✗ Cooling setpoints DIFFER at {diffs} steps — controller is non-deterministic!")

    # ---- Determinism check: baseline vs hp_retrofit supply air flow ----
    if "baseline" in all_records and "hp_retrofit" in all_records:
        bl_flows = [r["sa_flow_sp"] for r in all_records["baseline"]]
        hp_flows = [r["sa_flow_sp"] for r in all_records["hp_retrofit"]]
        if bl_flows == hp_flows:
            print("  ✓ Supply air flow setpoints are IDENTICAL (controller is deterministic)")
        else:
            diffs = sum(1 for a, b in zip(bl_flows, hp_flows) if abs(a - b) > 1e-6)
            print(f"  ✗ Supply air flow setpoints DIFFER at {diffs} steps")
            print(f"    → Even with same setpoints, the thermal model may diverge slightly")
            print(f"    → This is expected if zone temps differ (due to COP → different thermal load)")

    # ---- Step-by-step table (compact) ----
    print(f"\n{'='*100}")
    print(f"STEP-BY-STEP: SFH_1 (first 96 steps)")
    print(f"{'='*100}")
    hdr = f"{'step':>4s} {'hr':>5s} {'occ':>3s} {'peak':>4s}"
    for cn in cases:
        hdr += f" | {cn[:10]:>10s}_Tzone {cn[:10]:>10s}_coolSP {cn[:10]:>10s}_FCU_kW {cn[:10]:>10s}_grid_kW"
    print(hdr)
    print("-" * len(hdr))

    n_steps = min(len(all_records[c]) for c in cases)
    for i in range(n_steps):
        hour = i * 0.25
        r0 = all_records[cases[0]][i]
        row = f"{i:4d} {hour:5.1f}h {r0['occupancy']:3.0f} {'PEAK' if r0['is_peak'] else '    '}"
        for cn in cases:
            r = all_records[cn][i]
            row += f" | {r['zone_T']:>10.2f}  {r['cool_sp']:>10.1f}  {r['fcu_power_kw']:>10.3f}  {r['grid_kw']:>10.3f}"
        print(row)

    # ---- Summary ----
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for cn in cases:
        recs = all_records[cn]
        total_fcu = sum(r["fcu_power_kw"] for r in recs) * 0.25
        total_grid = sum(r["grid_kw"] for r in recs) * 0.25
        peak_grid = max(r["grid_kw"] for r in recs)
        total_pv = sum(r["pv_kw"] for r in recs) * 0.25
        total_bat = sum(r["bat_kw"] for r in recs) * 0.25
        total_ev = sum(r["ev_kw"] for r in recs) * 0.25
        avg_temp = np.mean([r["zone_T"] for r in recs])
        cop_val = recs[0].get("chiller_cop", "N/A")

        print(f"\n  {cn}:")
        print(f"    Chiller COP (from module):  {cop_val}")
        print(f"    HVAC energy:   {total_fcu:8.2f} kWh")
        print(f"    Grid import:   {total_grid:8.2f} kWh")
        print(f"    Peak grid:     {peak_grid:8.2f} kW")
        print(f"    PV generation: {total_pv:8.2f} kWh")
        print(f"    Battery net:   {total_bat:8.2f} kWh")
        print(f"    EV charging:   {total_ev:8.2f} kWh")
        print(f"    Avg zone temp: {avg_temp:8.2f} °C")

    # ---- Key checks ----
    print(f"\n{'='*80}")
    print("KEY CHECKS")
    print(f"{'='*80}")

    if "baseline" in all_records and "hp_retrofit" in all_records:
        bl_fcu = sum(r["fcu_power_kw"] for r in all_records["baseline"]) * 0.25
        hp_fcu = sum(r["fcu_power_kw"] for r in all_records["hp_retrofit"]) * 0.25
        ratio = hp_fcu / bl_fcu if bl_fcu > 0 else float("inf")
        expected = 3.0 / 4.5

        print(f"\n  ► HP Retrofit COP check:")
        print(f"    Baseline HVAC energy:    {bl_fcu:.2f} kWh")
        print(f"    HP Retrofit HVAC energy: {hp_fcu:.2f} kWh")
        print(f"    Ratio (actual):          {ratio:.3f}")
        print(f"    Ratio (expected ~3/4.5): {expected:.3f}")
        if abs(ratio - expected) < 0.15:
            print(f"    ✓ COP is working correctly!")
        elif abs(ratio - 1.0) < 0.05:
            print(f"    ✗ HVAC power is IDENTICAL — COP is NOT being used!")
            print(f"      → Check FCUModule: does it use config['chiller']['rated_cop']?")
            print(f"      → The thermal LOAD (Q_zone) should be the same, but electrical")
            print(f"        power should differ: P_elec = Q_zone / COP")
        else:
            print(f"    ⚠ Ratio is {ratio:.3f}, not exactly {expected:.3f}.")
            print(f"      → COP may be partially applied, or thermal dynamics diverged.")

    if "baseline" in all_records and "precooling" in all_records:
        bl_sps = [r["cool_sp"] for r in all_records["baseline"]]
        pc_sps = [r["cool_sp"] for r in all_records["precooling"]]
        diff_steps = [i for i, (a, b) in enumerate(zip(bl_sps, pc_sps)) if abs(a - b) > 0.01]

        print(f"\n  ► Pre-cooling setpoint check:")
        print(f"    Steps where cooling SP differs: {len(diff_steps)}/{len(bl_sps)}")
        if diff_steps:
            hours = [s * 0.25 for s in diff_steps]
            print(f"    Precooling window: hours {hours[0]:.1f}–{hours[-1]:.1f}")
            print(f"    Baseline SP:    {bl_sps[diff_steps[0]]:.1f}°C")
            print(f"    Precooling SP:  {pc_sps[diff_steps[0]]:.1f}°C (should be 2°C lower)")

            # Check that outside the window, setpoints match
            non_diff = [i for i in range(len(bl_sps)) if i not in diff_steps]
            sp_match = all(abs(bl_sps[i] - pc_sps[i]) < 0.01 for i in non_diff)
            if sp_match:
                print(f"    ✓ Outside precooling window, setpoints are identical")
            else:
                mismatches = sum(1 for i in non_diff if abs(bl_sps[i] - pc_sps[i]) > 0.01)
                print(f"    ⚠ {mismatches} steps outside precooling window also differ!")
        else:
            print(f"    ✗ Setpoints are IDENTICAL — precooling NOT applied!")
            peak_start = all_records["precooling"][0].get("step_of_day", "?")
            print(f"      → Peak starts at step_of_day={all_records['precooling'][50]['step_of_day']} "
                  f"(should be around 68 for 17:00)")

    if "baseline" in all_records and "pv_battery" in all_records:
        bl_grid = sum(r["grid_kw"] for r in all_records["baseline"]) * 0.25
        pv_grid = sum(r["grid_kw"] for r in all_records["pv_battery"]) * 0.25
        pv_gen = sum(r["pv_kw"] for r in all_records["pv_battery"]) * 0.25
        bat_net = sum(r["bat_kw"] for r in all_records["pv_battery"]) * 0.25
        curtail = sum(r["curtail_kw"] for r in all_records["pv_battery"]) * 0.25

        print(f"\n  ► PV+Battery check:")
        print(f"    Baseline grid:     {bl_grid:.2f} kWh")
        print(f"    PV+Bat grid:       {pv_grid:.2f} kWh")
        print(f"    PV generation:     {pv_gen:.2f} kWh")
        print(f"    Battery net:       {bat_net:.2f} kWh (+ = net charged from grid)")
        print(f"    Curtailment:       {curtail:.2f} kWh")
        print(f"    Expected grid ≈ baseline - PV_used + battery_charge_from_grid")

        # Check: grid should be less if PV offsets demand
        if pv_grid < bl_grid:
            print(f"    ✓ PV+Battery reduces grid import by {bl_grid - pv_grid:.2f} kWh")
        else:
            print(f"    ⚠ PV+Battery grid is HIGHER by {pv_grid - bl_grid:.2f} kWh")
            print(f"      → Battery charging from grid adds demand ({bat_net:.2f} kWh)")


if __name__ == "__main__":
    print("Running diagnostics for SFH_1 across all 4 cases...")
    all_records = {}
    for cn in CASE_NAMES:
        print(f"\n  Running {cn}...")
        all_records[cn] = run_diagnostic(cn)
    print_comparison(all_records)