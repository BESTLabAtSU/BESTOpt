"""
BESTOpt v7 — RL vs LLM Full Comparison

Loads LLM results from run_llm_v7.py, runs RL models (SAC/PPO/TD3)
with the same number of episodes, and generates:

  Fig 1–8:  Individual KPI boxplots (one per KPI, all controllers on x-axis)
  Fig 9:    Episodic reward boxplot
  Fig 10:   Operational trajectory — single day overlay (all controllers)
  Fig 11:   Operational trajectory — multi-panel per controller
  Fig 12:   Reward component stacked bar
  Fig 13:   LLM API cost summary
  Fig 14:   LLM vs Baseline delta chart
  Fig 15:   LLM vs RL delta chart

  + Full KPI table in terminal
  + Slide-ready analysis summary (Markdown + JSON)

HOW TO USE:
    1. Make sure run_llm_v7.py has already finished
    2. Make sure RL models exist in rl_results_v7_{mode}/
    3. Adjust CONTROL_MODE / RL_EPISODES below if needed
    4. Right-click → Run

Results saved to:  {PROJECT_ROOT}/comparison_v7_{mode}/
"""

import os
import sys
import logging
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path
from typing import Dict, List

# ── Project root (same as train_rl_v7.py) ──
PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

from bestopt.env.core.config_manager import ConfigurationManager
from bestopt.env.core.rl_wrapper import BESTOptGymEnv

try:
    from stable_baselines3 import PPO, SAC, TD3
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("WARNING: stable-baselines3 not installed, RL models will be skipped")

# =====================================================================
# *** RUN SETTINGS — edit these, then right-click Run ***
# =====================================================================
CONTROL_MODE   = "setpoint"     # must match what you ran in run_llm_v7.py
RL_EPISODES    = 5              # match your LLM episode count for fair comparison
SEED           = 42
# =====================================================================

CONFIG_PATH = os.path.join(
    PROJECT_ROOT_PATH, "examples", "SFH_1_Building", "config_setup.json"
)

REWARD_WEIGHTS = {
    "comfort_occupied": 1.5, "comfort_unoccupied": 0.3,
    "hvac_peak": 3.0, "hvac_prepeak": 0.2, "hvac_base": 1.0,
    "hvac_unoccupied": 2.0, "prepeak_hours": 2.0,
    "grid": 1.2, "curtail": 1.8, "comfort_deadband": 0.25,
}
NUM_DAYS = 5

RL_ALGO_NAMES = ["SAC", "PPO", "TD3"]
RL_ALGO_CLASSES = {"PPO": PPO, "SAC": SAC, "TD3": TD3} if HAS_SB3 else {}

DATA_KEYS = [
    "zone_temp", "outdoor_temp", "hvac_power_kw", "pv_gen_kw",
    "total_load_kw", "net_grid_kw", "bat_soc", "bat_power_kw",
    "electricity_price", "is_peak", "reward",
    "r_hvac", "r_comfort", "r_grid", "r_curtail",
    "w_hvac_dynamic", "w_comfort_dynamic", "hvac_zone",
    "curtailment_kw", "cooling_sp", "base_cooling_sp", "occupancy",
    "hours_until_peak", "comfort_margin", "is_unoccupied",
    "is_prepeak", "airflow_frac",
]

# =====================================================================
# Plotting style
# =====================================================================

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300,
})

# Colors: grouped by category
COLORS = {
    # Baseline
    "Baseline":        "#7f7f7f",
    # RL
    "SAC":             "#e63946",
    "PPO":             "#457b9d",
    "TD3":             "#2a9d8f",
    # LLM
    "LLM-ZeroShot":    "#f4a261",
    "LLM-FewShot":     "#e76f51",
    "LLM-Reflect":     "#264653",
    "LLM-Planner":     "#6a4c93",
    "LLM-PlanReflect": "#1982c4",
}

# Short display names for x-axis labels
SHORT_NAMES = {
    "Baseline": "Baseline",
    "SAC": "SAC", "PPO": "PPO", "TD3": "TD3",
    "LLM-ZeroShot": "Zero\nShot",
    "LLM-FewShot": "Few\nShot",
    "LLM-Reflect": "Reflect",
    "LLM-Planner": "Planner",
    "LLM-PlanReflect": "Plan\nReflect",
}

# Grouping for visual separation on x-axis
GROUP_LABELS = {
    "Baseline": "Rule", "SAC": "RL", "PPO": "RL", "TD3": "RL",
    "LLM-ZeroShot": "LLM", "LLM-FewShot": "LLM",
    "LLM-Reflect": "LLM", "LLM-Planner": "LLM",
    "LLM-PlanReflect": "LLM",
}


# =====================================================================
# Data loading — LLM results
# =====================================================================

def load_llm_results(control_mode):
    """Load LLM results from run_llm_v7.py output directory."""
    results_dir = os.path.join(
        PROJECT_ROOT_PATH, f"llm_results_v7_{control_mode}"
    )
    manifest_path = os.path.join(results_dir, "run_manifest.json")

    if not os.path.exists(manifest_path):
        print(f"  ERROR: No LLM results found at {results_dir}")
        print(f"  Run run_llm_v7.py first.")
        return {}

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_labels = ["Baseline"] + manifest["variants"]
    eval_data = {}

    for label in all_labels:
        label_dir = os.path.join(
            results_dir, label.replace("-", "_").lower()
        )
        if not os.path.isdir(label_dir):
            continue
        episodes = []
        for ep_file in sorted(Path(label_dir).glob("ep_*.npz")):
            raw = np.load(str(ep_file), allow_pickle=True)
            ep = {}
            for k in DATA_KEYS:
                if k in raw:
                    ep[k] = (raw[k].astype(object) if k == "hvac_zone"
                             else raw[k].astype(float))
            episodes.append(ep)
        if episodes:
            eval_data[label] = episodes
            print(f"  Loaded {label}: {len(episodes)} episodes")

    return eval_data


def load_llm_cost_trackers(control_mode):
    """Load LLM API cost reports."""
    results_dir = os.path.join(
        PROJECT_ROOT_PATH, f"llm_results_v7_{control_mode}"
    )
    trackers = {}
    for label in list(COLORS.keys()):
        if not label.startswith("LLM"):
            continue
        label_dir = os.path.join(
            results_dir, label.replace("-", "_").lower()
        )
        cost_path = os.path.join(label_dir, "cost_report.json")
        if os.path.exists(cost_path):
            with open(cost_path) as f:
                trackers[label] = json.load(f)
    return trackers


# =====================================================================
# Data loading — RL models (run fresh)
# =====================================================================

def run_rl_episodes(control_mode, n_episodes, seed):
    """Run RL models and collect episode data."""
    rl_dir = os.path.join(
        PROJECT_ROOT_PATH, f"rl_results_v7_{control_mode}"
    )
    rl_data = {}

    for algo in RL_ALGO_NAMES:
        model_path = None
        for suffix in [f"{algo.lower()}_bestopt.zip",
                       f"{algo.lower()}_bestopt"]:
            p = os.path.join(rl_dir, algo.lower(), suffix)
            if os.path.exists(p):
                model_path = p
                break
        if not model_path:
            print(f"  Warning: {algo} model not found in {rl_dir}, skipping")
            continue

        print(f"  Running {algo} ({n_episodes} episodes) ...")
        model = RL_ALGO_CLASSES[algo].load(model_path, device="cpu")
        episodes = []

        for ep in range(n_episodes):
            t0 = time.time()
            cm = ConfigurationManager(CONFIG_PATH)
            env = BESTOptGymEnv(
                config=cm.config, reward_weights=REWARD_WEIGHTS,
                num_days=NUM_DAYS, randomize_start=False,
                control_mode=control_mode,
            )
            obs, _ = env.reset(seed=seed + ep)
            data = {k: [] for k in DATA_KEYS}
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc

                d = info.get("obs_values", {})
                rb = info.get("reward_breakdown", {})

                data["zone_temp"].append(d.get("zone_temp", 0))
                data["outdoor_temp"].append(d.get("outdoor_temp", 0))
                data["hvac_power_kw"].append(d.get("hvac_power_kw", 0))
                data["pv_gen_kw"].append(d.get("pv_gen_kw", 0))
                data["total_load_kw"].append(d.get("total_load_kw", 0))
                data["net_grid_kw"].append(rb.get("net_grid_kw", 0))
                data["bat_soc"].append(d.get("bat_soc", 0))
                data["bat_power_kw"].append(d.get("bat_power_kw", 0))
                data["electricity_price"].append(d.get("electricity_price", 0))
                data["is_peak"].append(d.get("is_peak", 0))
                data["reward"].append(reward)
                data["r_hvac"].append(rb.get("r_hvac", 0))
                data["r_comfort"].append(rb.get("r_comfort", 0))
                data["r_grid"].append(rb.get("r_grid", 0))
                data["r_curtail"].append(rb.get("r_curtail", 0))
                data["w_hvac_dynamic"].append(rb.get("w_hvac_dynamic", 1.0))
                data["w_comfort_dynamic"].append(rb.get("w_comfort_dynamic", 1.0))
                data["hvac_zone"].append(rb.get("hvac_zone", "unknown"))
                data["curtailment_kw"].append(rb.get("curtailment_kw", 0))
                data["cooling_sp"].append(d.get("actual_cooling_sp", 24))
                data["base_cooling_sp"].append(d.get("base_cooling_sp", 24))
                data["occupancy"].append(d.get("occupancy", 0))
                data["hours_until_peak"].append(d.get("hours_until_peak", 6))
                data["comfort_margin"].append(d.get("comfort_margin", 0))
                data["is_unoccupied"].append(d.get("is_unoccupied", 0))
                data["is_prepeak"].append(d.get("is_prepeak", 0))
                data["airflow_frac"].append(d.get("airflow_frac", None))

            env.close()
            for k in data:
                data[k] = np.array(
                    data[k], dtype=object if k == "hvac_zone" else float
                )
            episodes.append(data)
            r = data["reward"].sum()
            print(f"    ep {ep+1}: reward={r:+.2f}  ({time.time()-t0:.1f}s)")

        rl_data[algo] = episodes
    return rl_data


# =====================================================================
# KPI computation (daily)
# =====================================================================

def compute_daily_kpis(ep, dt_h):
    steps_per_day = int(round(24.0 / dt_h))
    n_days = len(ep["zone_temp"]) // steps_per_day
    slice_keys = [k for k in ep if k != "hvac_zone"]
    daily = []

    for d in range(n_days):
        s, e = d * steps_per_day, (d + 1) * steps_per_day
        day = {k: ep[k][s:e] for k in slice_keys if len(ep[k]) >= e}

        total_e = day["total_load_kw"].sum() * dt_h
        hvac_e = day["hvac_power_kw"].sum() * dt_h
        pv = day["pv_gen_kw"].sum() * dt_h
        curt = day["curtailment_kw"].sum() * dt_h
        gi = np.clip(day["net_grid_kw"], 0, None)
        ip = day["is_peak"] > 0.5

        cost = np.where(ip, gi * 0.25 * dt_h, gi * 0.08 * dt_h).sum()

        db = REWARD_WEIGHTS.get("comfort_deadband", 0.25)
        z = day["zone_temp"]
        disc = float(np.sum(
            (z > day["base_cooling_sp"] + db) | (z < 18 - db)
        ) * dt_h)

        pk_ld = float(day["net_grid_kw"][ip].max()) if ip.any() else 0
        pk_im = float(gi[ip].sum() * dt_h) if ip.any() else 0
        tot_im = gi.sum() * dt_h
        flex = 1 - pk_im / tot_im if tot_im > 0 else 1

        exp = np.clip(-day["net_grid_kw"], 0, None).sum() * dt_h
        pv_used = max(pv - curt - exp, 0)

        daily.append({
            "operating_cost_usd":     cost,
            "total_energy_kwh":       total_e,
            "hvac_energy_kwh":        hvac_e,
            "total_grid_import_kwh":  tot_im,
            "discomfort_hours":       disc,
            "peak_load_onpeak_kw":    pk_ld,
            "peak_hour_import_kwh":   pk_im,
            "grid_flexibility_index": flex,
            "pv_curtailment_kwh":     curt,
            "pv_curtailment_pct":     curt / pv * 100 if pv > 0 else 0,
            "self_consumption":       pv_used / pv * 100 if pv > 0 else 0,
            "self_sufficiency":       (total_e - tot_im) / total_e * 100
                                      if total_e > 0 else 0,
        })
    return daily


# =====================================================================
# Helper: hourly profile
# =====================================================================

def hourly_mean(episodes, key, dt_h):
    spd = int(round(24 / dt_h))
    sph = int(round(1 / dt_h))
    bins = {h: [] for h in range(24)}
    for ep in episodes:
        arr = ep[key]
        nd = len(arr) // spd
        for d in range(nd):
            for h in range(24):
                s = d * spd + h * sph
                e = s + sph
                if e <= len(arr):
                    bins[h].extend(arr[s:e].tolist())
    return np.array([np.mean(bins[h]) if bins[h] else 0 for h in range(24)])


def detect_peak_hours(episodes, dt_h):
    spd = int(round(24 / dt_h))
    sph = int(round(1 / dt_h))
    hf = {h: [] for h in range(24)}
    for ep in episodes:
        p = ep["is_peak"]
        nd = len(p) // spd
        for d in range(nd):
            for h in range(24):
                s = d * spd + h * sph
                e = s + sph
                if e <= len(p):
                    hf[h].append(p[s:e].mean())
    return sorted([h for h in range(24) if np.mean(hf[h]) > 0.5])


# =====================================================================
# Figure: Individual KPI boxplots (one per KPI)
# =====================================================================

def plot_kpi_individual_boxplots(eval_data, dt_h, out_dir):
    """One figure per KPI, all controllers on x-axis with group separators."""
    labels = list(eval_data.keys())

    all_kpis = {}
    for label, eps in eval_data.items():
        daily = []
        for ep in eps:
            daily.extend(compute_daily_kpis(ep, dt_h))
        all_kpis[label] = daily

    kpi_defs = [
        ("operating_cost_usd",     "Operating Cost",       "USD / day",     "fig01_cost.png"),
        ("hvac_energy_kwh",        "HVAC Energy",          "kWh / day",     "fig02_hvac_energy.png"),
        ("total_grid_import_kwh",  "Grid Import",          "kWh / day",     "fig03_grid_import.png"),
        ("discomfort_hours",       "Thermal Discomfort",   "hours / day",   "fig04_discomfort.png"),
        ("peak_load_onpeak_kw",    "Peak Demand (on-peak)","kW",            "fig05_peak_demand.png"),
        ("peak_hour_import_kwh",   "Peak-Hour Grid Import","kWh / day",     "fig06_peak_import.png"),
        ("grid_flexibility_index", "Grid Flexibility",     "index (0–1)",   "fig07_flexibility.png"),
        ("pv_curtailment_kwh",     "PV Curtailment",       "kWh / day",     "fig08_curtailment.png"),
    ]

    for kpi_key, title, unit, fname in kpi_defs:
        fig, ax = plt.subplots(figsize=(7, 3.2))

        box_data = []
        box_colors = []
        x_labels = []
        x_positions = []
        pos = 0

        prev_group = None
        for label in labels:
            group = GROUP_LABELS.get(label, "")
            # Add gap between groups
            if prev_group is not None and group != prev_group:
                pos += 0.6
            prev_group = group

            vals = [k[kpi_key] for k in all_kpis[label]]
            box_data.append(vals)
            box_colors.append(COLORS.get(label, "#333"))
            x_labels.append(SHORT_NAMES.get(label, label))
            x_positions.append(pos)
            pos += 1

        bp = ax.boxplot(
            box_data, positions=x_positions, widths=0.6,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=3),
            showfliers=True,
            flierprops=dict(marker="o", markersize=2, alpha=0.4),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            medianprops=dict(linewidth=1.0, color="black"),
            boxprops=dict(linewidth=0.6),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_ylabel(unit)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.15, axis="y")

        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {path}")


# =====================================================================
# Figure: Episodic reward boxplot
# =====================================================================

def plot_reward_boxplot(eval_data, out_dir):
    """Boxplot of total episodic reward across all controllers."""
    labels = list(eval_data.keys())

    fig, ax = plt.subplots(figsize=(7, 3.2))

    box_data = []
    box_colors = []
    x_labels = []
    x_positions = []
    pos = 0
    prev_group = None

    for label in labels:
        group = GROUP_LABELS.get(label, "")
        if prev_group is not None and group != prev_group:
            pos += 0.6
        prev_group = group

        rewards = [ep["reward"].sum() for ep in eval_data[label]]
        box_data.append(rewards)
        box_colors.append(COLORS.get(label, "#333"))
        x_labels.append(SHORT_NAMES.get(label, label))
        x_positions.append(pos)
        pos += 1

    bp = ax.boxplot(
        box_data, positions=x_positions, widths=0.6,
        patch_artist=True, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=3),
        showfliers=True,
        flierprops=dict(marker="o", markersize=2, alpha=0.4),
        medianprops=dict(linewidth=1.0, color="black"),
        boxprops=dict(linewidth=0.6),
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_ylabel("Episodic Reward (5-day)")
    ax.set_title("Total Reward Comparison", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.15, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig09_reward.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# =====================================================================
# Figure: Trajectory overlay — single day, all controllers
# =====================================================================

def plot_trajectory_overlay(eval_data, dt_h, out_dir, control_mode):
    """
    Overlay all controllers on the same 24-hour profile.
    Uses hourly means averaged across episodes and days.
    """
    labels = list(eval_data.keys())
    hours = np.arange(24)

    ref = labels[0]
    pk = detect_peak_hours(eval_data[ref], dt_h)
    pks = min(pk) if pk else None
    pke = max(pk) + 1 if pk else None

    panels = [
        ("zone_temp",       "Zone Temperature (°C)",    "(a)"),
        ("hvac_power_kw",   "HVAC Power (kW)",          "(b)"),
        ("bat_soc",         "Battery SOC",              "(c)"),
        ("net_grid_kw",     "Net Grid Import (kW)",     "(d)"),
        ("pv_gen_kw",       "PV Generation (kW)",       "(e)"),
        ("curtailment_kw",  "PV Curtailment (kW)",      "(f)"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.2 * len(panels)),
                             sharex=True)
    fig.subplots_adjust(hspace=0.30, top=0.94)

    for idx, (key, ylabel, panel_label) in enumerate(panels):
        ax = axes[idx]
        if pks is not None:
            ax.axvspan(pks, pke, alpha=0.07, color="red")

        for label in labels:
            c = COLORS.get(label, "#333")
            # RL and Baseline: solid, LLM: dashed
            ls = "--" if label.startswith("LLM") else "-"
            lw = 1.0 if label.startswith("LLM") else 1.4
            alpha = 0.8 if label.startswith("LLM") else 1.0

            profile = hourly_mean(eval_data[label], key, dt_h)
            ax.plot(hours, profile, color=c, ls=ls, lw=lw,
                    alpha=alpha, label=label)

        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(f"{panel_label} {ylabel}", fontweight="bold",
                     loc="left", fontsize=8)
        ax.grid(True, alpha=0.12)
        if key == "bat_soc":
            ax.set_ylim(-0.02, 1.02)

    axes[-1].set_xlabel("Hour of Day")
    axes[-1].set_xticks(range(24))
    axes[-1].set_xlim(-0.5, 23.5)

    # Legend — split into groups
    rl_handles = [mlines.Line2D([], [], color=COLORS.get(l, "#333"),
                                lw=1.4, ls="-", label=l)
                  for l in labels if not l.startswith("LLM")]
    llm_handles = [mlines.Line2D([], [], color=COLORS.get(l, "#333"),
                                 lw=1.0, ls="--", label=l)
                   for l in labels if l.startswith("LLM")]
    all_handles = rl_handles + llm_handles
    if pks is not None:
        all_handles.append(
            mpatches.Patch(facecolor="red", alpha=0.07, label="Peak"))

    fig.legend(handles=all_handles, loc="upper center", frameon=False,
               fontsize=6, ncol=min(len(all_handles), 5),
               bbox_to_anchor=(0.5, 1.0))

    path = os.path.join(out_dir, "fig10_trajectory_overlay.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# =====================================================================
# Figure: Trajectory multi-panel (one column per controller)
# =====================================================================

def plot_trajectory_multipanel(eval_data, dt_h, out_dir, control_mode):
    """
    Multi-panel: rows = metrics, columns = controllers.
    Shows first episode day 1 raw data (not averaged).
    """
    labels = list(eval_data.keys())
    n_cols = len(labels)

    metrics = [
        ("zone_temp",     "T_zone (°C)"),
        ("hvac_power_kw", "HVAC (kW)"),
        ("bat_soc",       "Battery SOC"),
        ("net_grid_kw",   "Grid (kW)"),
    ]
    n_rows = len(metrics)

    steps_per_day = int(round(24 / dt_h))

    # Detect peak from first episode
    ref_peak = eval_data[labels[0]][0]["is_peak"][:steps_per_day]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.2 * n_cols, 1.8 * n_rows),
                             sharex=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.30, top=0.92)

    hours = np.arange(steps_per_day) * dt_h

    for col, label in enumerate(labels):
        ep = eval_data[label][0]  # first episode, first day

        for row, (key, ylabel) in enumerate(metrics):
            ax = axes[row][col] if n_cols > 1 else axes[row]
            arr = ep[key][:steps_per_day]

            # Peak shading
            in_peak = False
            for i in range(len(hours)):
                if i < len(ref_peak) and ref_peak[i] > 0.5 and not in_peak:
                    start_h = hours[i]
                    in_peak = True
                elif (i >= len(ref_peak) or ref_peak[i] <= 0.5) and in_peak:
                    ax.axvspan(start_h, hours[i], alpha=0.07, color="red")
                    in_peak = False
            if in_peak:
                ax.axvspan(start_h, hours[-1], alpha=0.07, color="red")

            c = COLORS.get(label, "#333")
            ax.plot(hours, arr, color=c, lw=0.8)
            ax.grid(True, alpha=0.1)

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=6)
            if row == 0:
                ax.set_title(SHORT_NAMES.get(label, label),
                             fontweight="bold", fontsize=7,
                             color=c)
            if row == n_rows - 1:
                ax.set_xlabel("Hour", fontsize=6)

            if key == "bat_soc":
                ax.set_ylim(-0.02, 1.02)

    path = os.path.join(out_dir, "fig11_trajectory_multipanel.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# =====================================================================
# Figure: Reward component stacked bar
# =====================================================================

def plot_reward_components(eval_data, dt_h, out_dir):
    """Stacked bar showing mean weighted reward components per controller."""
    labels = list(eval_data.keys())

    components = ["r_hvac", "r_comfort", "r_grid", "r_curtail"]
    weights =    ["w_hvac_dynamic", "w_comfort_dynamic", None, None]
    comp_names = ["HVAC Energy", "Comfort", "Grid Cost", "Curtailment"]
    comp_colors = ["#e63946", "#457b9d", "#f4a261", "#2a9d8f"]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Build x positions with group gaps
    x_positions = []
    pos = 0
    prev_group = None
    for label in labels:
        group = GROUP_LABELS.get(label, "")
        if prev_group is not None and group != prev_group:
            pos += 0.6
        prev_group = group
        x_positions.append(pos)
        pos += 1

    bar_w = 0.6

    for c_idx, (comp_key, w_key, c_name, c_color) in enumerate(
            zip(components, weights, comp_names, comp_colors)):
        means = []
        for label in labels:
            eps_vals = []
            for ep in eval_data[label]:
                if w_key:
                    val = (ep[comp_key] * ep[w_key]).sum()
                else:
                    val = ep[comp_key].sum()
                eps_vals.append(-val)  # penalties are negative
            means.append(np.mean(eps_vals))

        bottom = np.zeros(len(labels))
        if c_idx > 0:
            # Stack on top of previous
            for prev in range(c_idx):
                pk, wk = components[prev], weights[prev]
                for j, label in enumerate(labels):
                    ev = []
                    for ep in eval_data[label]:
                        if wk:
                            ev.append(-(ep[pk] * ep[wk]).sum())
                        else:
                            ev.append(-ep[pk].sum())
                    bottom[j] += np.mean(ev)

        ax.bar(x_positions, means, bar_w, bottom=bottom,
               color=c_color, alpha=0.75, label=c_name,
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([SHORT_NAMES.get(l, l) for l in labels], fontsize=7)
    ax.set_ylabel("Cumulative Penalty (episode)")
    ax.set_title("Reward Component Breakdown", fontweight="bold", fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.grid(True, alpha=0.12, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig12_reward_components.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# =====================================================================
# Figure: LLM cost summary
# =====================================================================

def plot_llm_cost(cost_trackers, out_dir):
    if not cost_trackers:
        return
    labels = list(cost_trackers.keys())
    n = len(labels)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    colors = [COLORS.get(l, "#333") for l in labels]

    ax = axes[0]
    ax.bar(range(n), [cost_trackers[l]["total_cost_usd"] for l in labels],
           color=colors, alpha=0.7)
    ax.set_xticks(range(n))
    ax.set_xticklabels([l.replace("LLM-", "") for l in labels],
                       fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("Cost (USD)")
    ax.set_title("API Cost", fontweight="bold")

    ax = axes[1]
    x = np.arange(n)
    ax.bar(x - 0.2, [cost_trackers[l]["input_tokens"] / 1000 for l in labels],
           0.35, label="Input", alpha=0.7, color="#457b9d")
    ax.bar(x + 0.2, [cost_trackers[l]["output_tokens"] / 1000 for l in labels],
           0.35, label="Output", alpha=0.7, color="#e63946")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("LLM-", "") for l in labels],
                       fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("Tokens (×1000)")
    ax.set_title("Token Usage", fontweight="bold")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.bar(range(n), [cost_trackers[l]["avg_latency_s"] for l in labels],
           color=colors, alpha=0.7)
    ax.set_xticks(range(n))
    ax.set_xticklabels([l.replace("LLM-", "") for l in labels],
                       fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("Avg Latency (s/call)")
    ax.set_title("API Latency", fontweight="bold")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig13_llm_cost.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# =====================================================================
# KPI table
# =====================================================================

def print_full_table(eval_data, dt_h, control_mode):
    labels = list(eval_data.keys())

    all_kpis = {}
    for label, eps in eval_data.items():
        daily = []
        for ep in eps:
            daily.extend(compute_daily_kpis(ep, dt_h))
        all_kpis[label] = daily

    table_rows = [
        ("operating_cost_usd",     "Cost (USD/day)"),
        ("total_energy_kwh",       "Energy (kWh/day)"),
        ("hvac_energy_kwh",        "HVAC Energy (kWh/day)"),
        ("total_grid_import_kwh",  "Grid Import (kWh/day)"),
        ("discomfort_hours",       "Discomfort (h/day)"),
        ("peak_load_onpeak_kw",    "Peak Load (kW)"),
        ("peak_hour_import_kwh",   "Peak Import (kWh/day)"),
        ("grid_flexibility_index", "Flexibility Index"),
        ("pv_curtailment_kwh",     "Curtail (kWh/day)"),
        ("pv_curtailment_pct",     "Curtail (%)"),
        ("self_consumption",       "Self-Consumption (%)"),
        ("self_sufficiency",       "Self-Sufficiency (%)"),
    ]

    # Baseline means
    bl = labels[0]
    blm = {k: np.mean([x[k] for x in all_kpis[bl]]) for k, _ in table_rows}

    cw = 24
    hdr = f"{'KPI':<28s}"
    for l in labels:
        hdr += f" | {l:>{cw}s}"
    sep = "=" * len(hdr)

    n_daily = {l: len(v) for l, v in all_kpis.items()}
    print(f"\n{sep}")
    print(f"RL vs LLM COMPARISON  ({control_mode} mode, daily KPIs)")
    print(f"Daily samples: {n_daily}")
    print(sep)
    print(hdr)
    print("-" * len(hdr))

    for kk, kn in table_rows:
        row = f"{kn:<28s}"
        for label in labels:
            vals = np.array([x[kk] for x in all_kpis[label]])
            m, s = vals.mean(), vals.std()
            if label == bl:
                cell = f"{m:>8.2f} +/- {s:>5.2f}       "
            else:
                pct = ((m - blm[kk]) / abs(blm[kk]) * 100) if blm[kk] != 0 else 0
                cell = f"{m:>8.2f} +/- {s:>5.2f} ({pct:+.1f}%)"
            row += f" | {cell:>{cw}s}"
        print(row)
    print(sep)

    # Reward summary
    print(f"\nEPISODIC REWARD  ({control_mode} mode)")
    print("-" * 70)
    bl_r = np.mean([ep["reward"].sum() for ep in eval_data[bl]])
    for label in labels:
        rr = [ep["reward"].sum() for ep in eval_data[label]]
        m, s = np.mean(rr), np.std(rr)
        group = GROUP_LABELS.get(label, "")
        if label == bl:
            print(f"  [{group:4s}] {label:<20s}  {m:+8.2f} +/- {s:.2f}")
        else:
            imp = (m - bl_r) / abs(bl_r) * 100 if bl_r != 0 else 0
            print(f"  [{group:4s}] {label:<20s}  {m:+8.2f} +/- {s:.2f}  "
                  f"({imp:+.1f}% vs baseline)")
    print("-" * 70)

    return all_kpis  # return for analysis


# =====================================================================
# NEW: Delta comparison charts (LLM vs Baseline, LLM vs best RL)
# =====================================================================

def plot_delta_comparisons(eval_data, dt_h, out_dir):
    """
    Generate two grouped-bar charts:
      Fig 14: LLM variants % change vs Baseline
      Fig 15: LLM variants % change vs Best RL
    Returns the analysis_data dict for the summary.
    """
    labels = list(eval_data.keys())

    all_kpis = {}
    for label, eps in eval_data.items():
        daily = []
        for ep in eps:
            daily.extend(compute_daily_kpis(ep, dt_h))
        all_kpis[label] = daily

    # KPIs to compare (subset most useful for slides)
    compare_kpis = [
        ("operating_cost_usd",     "Cost"),
        ("hvac_energy_kwh",        "HVAC\nEnergy"),
        ("total_grid_import_kwh",  "Grid\nImport"),
        ("discomfort_hours",       "Discomfort"),
        ("peak_load_onpeak_kw",    "Peak\nDemand"),
        ("grid_flexibility_index", "Flexibility"),
        ("pv_curtailment_kwh",     "PV\nCurtail"),
    ]

    # Identify groups
    bl_label = "Baseline"
    rl_labels = [l for l in labels if GROUP_LABELS.get(l) == "RL"]
    llm_labels = [l for l in labels if l.startswith("LLM")]

    # Compute means
    means = {}
    for label in labels:
        means[label] = {
            kk: np.mean([x[kk] for x in all_kpis[label]])
            for kk, _ in compare_kpis
        }

    # Best RL per KPI (lowest cost/energy/discomfort, highest flexibility)
    higher_is_better = {"grid_flexibility_index"}
    best_rl = {}
    if rl_labels:
        for kk, _ in compare_kpis:
            if kk in higher_is_better:
                best_rl_label = max(rl_labels, key=lambda l: means[l][kk])
            else:
                best_rl_label = min(rl_labels, key=lambda l: means[l][kk])
            best_rl[kk] = (best_rl_label, means[best_rl_label][kk])

    # ── Fig 14: LLM vs Baseline ──
    _plot_grouped_delta(
        llm_labels, compare_kpis, means, bl_label,
        {kk: means[bl_label][kk] for kk, _ in compare_kpis},
        higher_is_better, "vs Baseline",
        os.path.join(out_dir, "fig14_llm_vs_baseline.png"),
    )

    # ── Fig 15: LLM vs Best RL ──
    if best_rl:
        ref_vals = {kk: best_rl[kk][1] for kk, _ in compare_kpis}
        ref_name = "Best RL"
        _plot_grouped_delta(
            llm_labels, compare_kpis, means, ref_name, ref_vals,
            higher_is_better, "vs Best RL",
            os.path.join(out_dir, "fig15_llm_vs_best_rl.png"),
        )

    # Build analysis data
    analysis = {
        "baseline_means": means.get(bl_label, {}),
        "rl_means": {l: means[l] for l in rl_labels},
        "llm_means": {l: means[l] for l in llm_labels},
        "best_rl_per_kpi": {kk: best_rl[kk][0] for kk, _ in compare_kpis}
            if best_rl else {},
        "kpi_names": {kk: kn for kk, kn in compare_kpis},
    }
    return analysis


def _plot_grouped_delta(llm_labels, compare_kpis, means, ref_name,
                        ref_vals, higher_is_better, subtitle, save_path):
    """Grouped bar chart: % change of each LLM variant vs a reference."""
    n_kpis = len(compare_kpis)
    n_llm = len(llm_labels)
    if n_llm == 0:
        return

    x = np.arange(n_kpis)
    total_w = 0.7
    bar_w = total_w / n_llm

    fig, ax = plt.subplots(figsize=(8, 3.5))

    for i, llm in enumerate(llm_labels):
        deltas = []
        for kk, _ in compare_kpis:
            ref = ref_vals[kk]
            val = means[llm][kk]
            if ref != 0:
                pct = (val - ref) / abs(ref) * 100
            else:
                pct = 0
            # For "higher is better" KPIs, positive % = good
            # For others, negative % = good — flip sign for display
            if kk not in higher_is_better:
                pct = -pct  # now positive = improvement
            deltas.append(pct)

        offset = (i - (n_llm - 1) / 2) * bar_w
        color = COLORS.get(llm, "#333")
        short = llm.replace("LLM-", "")
        bars = ax.bar(x + offset, deltas, bar_w, label=short,
                      color=color, alpha=0.75, edgecolor="white",
                      linewidth=0.3)

        # Value labels on bars
        for bar, d in zip(bars, deltas):
            va = "bottom" if d >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(), f"{d:+.1f}%",
                    ha="center", va=va, fontsize=5.5, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([kn for _, kn in compare_kpis], fontsize=7)
    ax.set_ylabel("Improvement (%)")
    ax.set_title(f"LLM Controller Performance {subtitle}",
                 fontweight="bold", fontsize=10)
    ax.legend(loc="best", frameon=False, fontsize=7, ncol=min(n_llm, 3))
    ax.grid(True, alpha=0.12, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}")


# =====================================================================
# NEW: Slide-ready analysis summary
# =====================================================================

def write_analysis_summary(eval_data, dt_h, cost_trackers, out_dir,
                           control_mode):
    """
    Write a Markdown analysis comparing LLM vs Baseline and LLM vs RL,
    plus a JSON summary with all the numbers for easy slide creation.
    """
    labels = list(eval_data.keys())
    bl = "Baseline"
    rl_labels = [l for l in labels if GROUP_LABELS.get(l) == "RL"]
    llm_labels = [l for l in labels if l.startswith("LLM")]

    # Compute KPI means
    all_kpis = {}
    for label, eps in eval_data.items():
        daily = []
        for ep in eps:
            daily.extend(compute_daily_kpis(ep, dt_h))
        all_kpis[label] = daily

    kpi_list = [
        ("operating_cost_usd",     "Operating Cost (USD/day)"),
        ("hvac_energy_kwh",        "HVAC Energy (kWh/day)"),
        ("total_grid_import_kwh",  "Grid Import (kWh/day)"),
        ("discomfort_hours",       "Thermal Discomfort (h/day)"),
        ("peak_load_onpeak_kw",    "Peak Demand (kW)"),
        ("grid_flexibility_index", "Grid Flexibility Index"),
        ("pv_curtailment_kwh",     "PV Curtailment (kWh/day)"),
    ]

    means = {}
    stds = {}
    for label in labels:
        means[label] = {}
        stds[label] = {}
        for kk, _ in kpi_list:
            vals = [x[kk] for x in all_kpis[label]]
            means[label][kk] = np.mean(vals)
            stds[label][kk] = np.std(vals)

    # Episodic rewards
    rewards = {}
    for label in labels:
        rr = [ep["reward"].sum() for ep in eval_data[label]]
        rewards[label] = {"mean": np.mean(rr), "std": np.std(rr)}

    # ── Markdown report ──
    lines = []
    lines.append("# BESTOpt v7 — LLM vs RL Analysis Summary")
    lines.append(f"\nControl mode: **{control_mode}** | Episodes: "
                 f"{len(list(eval_data.values())[0])}")
    lines.append(f"\nControllers evaluated: {', '.join(labels)}")

    # Section 1: LLM vs Baseline
    lines.append("\n## 1. LLM vs Baseline\n")
    lines.append(f"| KPI | Baseline | " +
                 " | ".join(l.replace('LLM-', '') for l in llm_labels) + " |")
    lines.append(f"|-----|----------| " +
                 " | ".join("---" for _ in llm_labels) + " |")
    for kk, kn in kpi_list:
        bm = means[bl][kk]
        row = f"| {kn} | {bm:.2f} |"
        for llm in llm_labels:
            m = means[llm][kk]
            pct = (m - bm) / abs(bm) * 100 if bm != 0 else 0
            row += f" {m:.2f} ({pct:+.1f}%) |"
        lines.append(row)

    # Section 2: LLM vs Best RL
    if rl_labels:
        lines.append("\n## 2. LLM vs Best RL\n")
        lines.append("Best RL per KPI:")
        higher_better = {"grid_flexibility_index"}
        for kk, kn in kpi_list:
            if kk in higher_better:
                best = max(rl_labels, key=lambda l: means[l][kk])
            else:
                best = min(rl_labels, key=lambda l: means[l][kk])
            lines.append(f"- {kn}: **{best}** ({means[best][kk]:.2f})")

        lines.append("")
        lines.append(f"| KPI | Best RL | " +
                     " | ".join(l.replace('LLM-', '') for l in llm_labels) +
                     " |")
        lines.append(f"|-----|--------| " +
                     " | ".join("---" for _ in llm_labels) + " |")
        for kk, kn in kpi_list:
            if kk in higher_better:
                best = max(rl_labels, key=lambda l: means[l][kk])
            else:
                best = min(rl_labels, key=lambda l: means[l][kk])
            bm = means[best][kk]
            row = f"| {kn} | {bm:.2f} ({best}) |"
            for llm in llm_labels:
                m = means[llm][kk]
                pct = (m - bm) / abs(bm) * 100 if bm != 0 else 0
                row += f" {m:.2f} ({pct:+.1f}%) |"
            lines.append(row)

    # Section 3: Reward comparison
    lines.append("\n## 3. Episodic Reward (5-day)\n")
    lines.append("| Controller | Group | Mean Reward | Std | vs Baseline |")
    lines.append("|------------|-------|-------------|-----|-------------|")
    bl_r = rewards[bl]["mean"]
    for label in labels:
        grp = GROUP_LABELS.get(label, "")
        m = rewards[label]["mean"]
        s = rewards[label]["std"]
        imp = (m - bl_r) / abs(bl_r) * 100 if bl_r != 0 else 0
        lines.append(f"| {label} | {grp} | {m:+.2f} | {s:.2f} | "
                     f"{imp:+.1f}% |")

    # Section 4: Cost (if available)
    if cost_trackers:
        lines.append("\n## 4. LLM API Cost\n")
        lines.append("| Variant | Calls | Cost (USD) | Avg Latency (s) |")
        lines.append("|---------|-------|------------|------------------|")
        tc = 0
        for l, t in cost_trackers.items():
            lines.append(f"| {l} | {t['total_calls']} | "
                         f"${t['total_cost_usd']:.4f} | "
                         f"{t['avg_latency_s']:.2f} |")
            tc += t["total_cost_usd"]
        lines.append(f"| **TOTAL** | | **${tc:.4f}** | |")

    # Section 5: Key takeaways (template)
    lines.append("\n## 5. Key Takeaways (for slides)\n")
    lines.append("- **LLM vs Baseline:** [fill after reviewing deltas in "
                 "fig14_llm_vs_baseline.png]")
    lines.append("- **LLM vs RL:** [fill after reviewing deltas in "
                 "fig15_llm_vs_best_rl.png]")
    lines.append("- **Best LLM variant:** [check which variant wins most "
                 "KPIs]")
    lines.append("- **Cost-performance tradeoff:** [compare API cost vs "
                 "KPI improvement]")
    lines.append("- **Comfort vs Energy tradeoff:** [compare discomfort "
                 "vs HVAC energy across methods]")

    md_text = "\n".join(lines)
    md_path = os.path.join(out_dir, "analysis_summary.md")
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"  -> {md_path}")

    # ── JSON for programmatic use in slides ──
    json_data = {
        "control_mode": control_mode,
        "controllers": labels,
        "kpi_means": {l: {kk: float(means[l][kk]) for kk, _ in kpi_list}
                      for l in labels},
        "kpi_stds": {l: {kk: float(stds[l][kk]) for kk, _ in kpi_list}
                     for l in labels},
        "rewards": {l: {"mean": float(rewards[l]["mean"]),
                        "std": float(rewards[l]["std"])} for l in labels},
        "cost_trackers": cost_trackers if cost_trackers else {},
    }
    json_path = os.path.join(out_dir, "analysis_data.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  -> {json_path}")

    return md_text


# =====================================================================
# Main
# =====================================================================

def main():
    control_mode = CONTROL_MODE
    n_rl_episodes = RL_EPISODES
    seed = SEED

    logging.getLogger("BESTOptEnvironment").setLevel(logging.WARNING)
    logging.getLogger("ThermalDynamics").setLevel(logging.WARNING)
    logging.getLogger("bestopt").setLevel(logging.WARNING)

    out_dir = os.path.join(
        PROJECT_ROOT_PATH, f"comparison_v7_{control_mode}"
    )
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"BESTOpt v7 — RL vs LLM Full Comparison")
    print(f"  Mode:        {control_mode}")
    print(f"  RL episodes: {n_rl_episodes}")
    print(f"  Output:      {out_dir}")
    print("=" * 70)

    # ── 1. Load LLM results ──
    print(f"\n[1] Loading LLM results ...")
    eval_data = load_llm_results(control_mode)
    if not eval_data:
        print("  FATAL: No LLM results found. Run run_llm_v7.py first.")
        return

    llm_n_eps = len(list(eval_data.values())[0])
    cost_trackers = load_llm_cost_trackers(control_mode)

    # ── 2. Run RL models ──
    if HAS_SB3:
        print(f"\n[2] Running RL models ({n_rl_episodes} episodes each) ...")
        rl_data = run_rl_episodes(control_mode, n_rl_episodes, seed)
        # Insert RL data after Baseline, before LLM
        ordered_data = {}
        if "Baseline" in eval_data:
            ordered_data["Baseline"] = eval_data["Baseline"]
        for algo in RL_ALGO_NAMES:
            if algo in rl_data:
                ordered_data[algo] = rl_data[algo]
        for label, eps in eval_data.items():
            if label != "Baseline":
                ordered_data[label] = eps
        eval_data = ordered_data
    else:
        print(f"\n[2] Skipping RL (stable-baselines3 not installed)")

    # ── 3. Get dt_h ──
    cm = ConfigurationManager(CONFIG_PATH)
    env = BESTOptGymEnv(config=cm.config, reward_weights=REWARD_WEIGHTS,
                        num_days=NUM_DAYS, control_mode=control_mode)
    dt_h = env.env.res / 3600.0
    env.close()

    print(f"\n  Controllers: {list(eval_data.keys())}")
    print(f"  Episodes per controller: "
          f"{', '.join(f'{l}={len(v)}' for l,v in eval_data.items())}")

    # ── 4. Generate plots ──
    print(f"\n[3] Generating KPI boxplots (one per KPI) ...")
    plot_kpi_individual_boxplots(eval_data, dt_h, out_dir)

    print(f"\n[4] Generating reward boxplot ...")
    plot_reward_boxplot(eval_data, out_dir)

    print(f"\n[5] Generating trajectory overlay (hourly mean) ...")
    plot_trajectory_overlay(eval_data, dt_h, out_dir, control_mode)

    print(f"\n[6] Generating trajectory multi-panel (per controller) ...")
    plot_trajectory_multipanel(eval_data, dt_h, out_dir, control_mode)

    print(f"\n[7] Generating reward component breakdown ...")
    plot_reward_components(eval_data, dt_h, out_dir)

    if cost_trackers:
        print(f"\n[8] Generating LLM cost figure ...")
        plot_llm_cost(cost_trackers, out_dir)

    # ── 5. NEW: Delta comparison charts ──
    print(f"\n[9] Generating LLM vs Baseline / RL delta charts ...")
    plot_delta_comparisons(eval_data, dt_h, out_dir)

    # ── 6. Tables ──
    print(f"\n[10] Quantitative comparison ...")
    print_full_table(eval_data, dt_h, control_mode)

    if cost_trackers:
        print(f"\nLLM API COST")
        print("-" * 60)
        tc = 0
        for l, t in cost_trackers.items():
            print(f"  {l:<20s}  calls={t['total_calls']:4d}  "
                  f"cost=${t['total_cost_usd']:.4f}  "
                  f"latency={t['avg_latency_s']:.2f}s avg")
            tc += t["total_cost_usd"]
        print(f"  {'TOTAL':<20s}  cost=${tc:.4f}")
        print("-" * 60)

    # ── 7. NEW: Analysis summary for slides ──
    print(f"\n[11] Writing analysis summary ...")
    write_analysis_summary(eval_data, dt_h, cost_trackers, out_dir,
                           control_mode)

    print(f"\n  All outputs -> {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()