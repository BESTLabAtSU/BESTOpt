"""
Analysis & Visualisation for Demand Response Flexibility Study
==============================================================
5 Cases: Baseline, Pre-cooling, HP Retrofit, EV, PV+Battery

COMFORT: evaluated against BASE setpoints (not precooling-adjusted),
         so all cases are compared fairly.

FLEXIBILITY KPIs:
  - Peak load reduction (max grid import, kW)
  - Peak-hour energy shift (total grid import during 17:00–21:00, kWh)
  - Total energy change (total grid import over horizon, kWh)

Figure 1  – One-day single-building deep-dive (5 subplots stacked)
Figure 2a – Grid import line plot (mean across 20 buildings)
Figure 2b – Grid import boxplot (daily kWh per building)
Figure 2c – Grid import time-series boxplot (distribution at each step)
Figure 3  – Two-week KPI box plots + radar
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ===========================
# STYLE
# ===========================

CASE_COLORS = {
    "baseline":    "#5B8DB8",
    "precooling":  "#E07B54",
    "hp_retrofit": "#6AAB73",
    "ev":          "#D4A84B",
    "pv_battery":  "#9B7EB8",
}
CASE_LABELS = {
    "baseline":    "Baseline",
    "precooling":  "Pre-cooling",
    "hp_retrofit": "HP Retrofit",
    "ev":          "EV",
    "pv_battery":  "PV-Battery",
}
CASE_ORDER = ["baseline", "precooling", "hp_retrofit", "ev", "pv_battery"]

RESOLUTION_H = 0.25  # 15 min in hours
GLOBAL_FONTSIZE = 7

# Peak hours from PriceModule: steps 68–84 → 17:00–21:00
PEAK_HOUR_START = 17
PEAK_HOUR_END = 21

SUBPLOT_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


def _style_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=GLOBAL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=GLOBAL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=GLOBAL_FONTSIZE, fontweight="bold", pad=0.5)
    if grid:
        ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=GLOBAL_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_visible(True)


# ===========================
# FIGURE 1 — Single-building one-day deep dive (5 subplots)
# ===========================

def plot_single_building_oneday(
    results_1day: dict,
    building_id: str = "SFH_1",
    case_name: str = "baseline",
    save_path: str = None,
):
    """
    5 rows × 1 column:
      (a) Zone temperature with occupancy-based setpoints (no comfort band)
      (b) Supply air temperature (line only)
      (c) Supply air flow rate (line only)
      (d) Stacked load disaggregation (HVAC, Lighting, Other; +EV if present)
      (e) Grid import line plot (mean across all buildings, all cases)
    """
    data = results_1day[case_name][building_id]
    n = len(data["timestamps"])
    t = np.arange(n) * RESOLUTION_H

    fig, axes = plt.subplots(5, 1, figsize=(4.4, 4.8), dpi=300, sharex=True)

    # ---------- (a) Zone temperature ----------
    ax_zt = axes[0]
    # Actual setpoints from simulation (occupancy-based with setback)
    cool_sp = np.array(data["cooling_setpoint"])
    heat_sp = np.array(data["heating_setpoint"])
    ax_zt.plot(t, data["zone_temperature"], color="#2166ac", lw=1.2,
               label="Zone temp")
    ax_zt.plot(t, cool_sp, "--", color="gray", lw=0.8, alpha=0.8,
               label="Setpoint")
    ax_zt.plot(t, heat_sp, "--", color="gray", lw=0.8, alpha=0.8)
    _style_ax(ax_zt, ylabel="Temp. (°C)",
              title=f"{SUBPLOT_LABELS[0]} Zone Temperature")
    # Legend outside box, below title
    ax_zt.legend(
        fontsize=GLOBAL_FONTSIZE - 1, frameon=False,
        loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2,
        borderaxespad=0,
    )
    ax_zt.set_ylim([14, 30])

    # ---------- (b) Supply air temperature ----------
    ax_sa = axes[1]
    ax_sa.plot(t, data["supply_air_temp_actual"], color="#b2182b", lw=1.2)
    _style_ax(ax_sa, ylabel="Temp. (°C)",
              title=f"{SUBPLOT_LABELS[1]} Supply Air Temperature")

    # ---------- (c) Supply air flow rate ----------
    ax_sf = axes[2]
    ax_sf.plot(t, data["supply_air_flow_actual"], color="#1b7837", lw=1.2)
    _style_ax(ax_sf, ylabel="Flow rate (m³/s)",
              title=f"{SUBPLOT_LABELS[2]} Supply Air Flow Rate")

    # ---------- (d) Load disaggregation ----------
    ax_ld = axes[3]

    hvac     = np.array(data["hvac_power_kw"])
    lighting = np.array(data["lighting_kw"])
    other    = (np.array(data["cooking_kw"])
                + np.array(data["pc_kw"])
                + np.array(data["tv_kw"]))
    ev       = np.array(data["ev_charging_kw"])

    # Conditionally include EV only if there is any EV charging
    has_ev = np.any(ev > 0.001)
    if has_ev:
        stack_labels = ["HVAC", "Lighting", "Other", "EV"]
        stack_data   = [hvac, lighting, other, ev]
        stack_colors = ["#4393c3", "#fdd49e", "#b2abd2", "#5aae61"]
    else:
        stack_labels = ["HVAC", "Lighting", "Other"]
        stack_data   = [hvac, lighting, other]
        stack_colors = ["#4393c3", "#fdd49e", "#b2abd2"]

    ax_ld.stackplot(t, *stack_data, labels=stack_labels,
                    colors=stack_colors, alpha=0.85)
    _style_ax(ax_ld, ylabel="Power (kW)",
              title=f"{SUBPLOT_LABELS[3]} Load Disaggregation")
    # Legend outside box, below title
    ax_ld.legend(
        fontsize=GLOBAL_FONTSIZE - 1, frameon=False,
        loc="upper left", bbox_to_anchor=(0.0, 1.02),
        ncol=len(stack_labels), borderaxespad=0,
    )
    ax_ld.set_ylim(bottom=0)

    # ---------- (e) Grid import line plot (mean across buildings) ----------
    ax_gi = axes[4]

    for cn in CASE_ORDER:
        if cn not in results_1day:
            continue
        case_data = results_1day[cn]
        bids = sorted(case_data.keys())
        mat = np.array([case_data[bid]["grid_import_kw"] for bid in bids])
        n_steps = mat.shape[1]
        t_gi = np.arange(n_steps) * RESOLUTION_H
        mean_val = np.mean(mat, axis=0)
        ax_gi.plot(t_gi, mean_val, color=CASE_COLORS[cn], lw=1.2,
                   label=CASE_LABELS[cn])

    ax_gi.axhline(0, color="grey", lw=0.5, ls="--")
    # Peak-hour shading
    ax_gi.axvspan(PEAK_HOUR_START, PEAK_HOUR_END, color="red", alpha=0.06)
    ylim = ax_gi.get_ylim()
    ax_gi.text((PEAK_HOUR_START + PEAK_HOUR_END) / 2,
               ylim[1] - 0.3 * (ylim[1] - ylim[0]),
               "Peak\nhours", ha="center", fontsize=GLOBAL_FONTSIZE - 1,
               color="red", alpha=0.6)

    _style_ax(ax_gi, xlabel="Time (h)", ylabel="Grid Import (kW)",
              title=f"{SUBPLOT_LABELS[4]} Grid Import (mean across buildings)")
    # Legend outside box, below title
    ax_gi.legend(
        fontsize=GLOBAL_FONTSIZE - 1, frameon=False,
        loc="lower center", bbox_to_anchor=(0.5, 0.3),
        ncol=2, borderaxespad=0, columnspacing=1.0,
    )

    for ax in axes:
        ax.set_xlim([0, 24])
        ax.set_xticks(np.arange(0, 25, 4))

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    return fig


# ===========================
# FIGURE 2a — Grid import line plot (mean across buildings)
# ===========================

def plot_grid_import_line(
    results_1day: dict,
    save_path: str = None,
):
    fig, ax = plt.subplots(figsize=(4.4, 2), dpi=300)

    n_steps = None
    for case_name in CASE_ORDER:
        if case_name not in results_1day:
            continue
        case_data = results_1day[case_name]
        bids = sorted(case_data.keys())
        mat = np.array([case_data[bid]["grid_import_kw"] for bid in bids])
        if n_steps is None:
            n_steps = mat.shape[1]
        t = np.arange(n_steps) * RESOLUTION_H

        mean_val = np.mean(mat, axis=0)
        ax.plot(t, mean_val, color=CASE_COLORS[case_name], lw=1.2,
                label=CASE_LABELS[case_name])

    ax.axhline(0, color="grey", lw=0.5, ls="--")
    # Peak-hour shading
    ax.axvspan(PEAK_HOUR_START, PEAK_HOUR_END, color="red", alpha=0.06)
    ylim = ax.get_ylim()
    ax.text((PEAK_HOUR_START + PEAK_HOUR_END) / 2,
            ylim[1] - 0.3 * (ylim[1] - ylim[0]),
            "Peak\nhours", ha="center", fontsize=GLOBAL_FONTSIZE - 1,
            color="red", alpha=0.6)

    _style_ax(ax, xlabel="Time (h)", ylabel="Grid Import (kW)")
    # Legend outside box, 2 rows (3+2), below title area
    ax.legend(
        fontsize=GLOBAL_FONTSIZE - 1, frameon=False,
        loc="lower left", bbox_to_anchor=(0.0, 1.02),
        ncol=3, borderaxespad=0, columnspacing=1.0,
    )
    ax.set_xlim([0, 24])
    ax.set_xticks(np.arange(0, 25, 2))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    return fig


# ===========================
# FIGURE 2b — Grid import boxplot (daily kWh)
# ===========================

def plot_grid_import_boxplot(
    results_1day: dict,
    save_path: str = None,
):
    fig, ax = plt.subplots(figsize=(4.4, 2.4), dpi=300)

    box_data, box_labels, box_colors = [], [], []

    for case_name in CASE_ORDER:
        if case_name not in results_1day:
            continue
        case_data = results_1day[case_name]
        bids = sorted(case_data.keys())
        mat = np.array([case_data[bid]["grid_import_kw"] for bid in bids])
        daily_kwh = np.sum(mat, axis=1) * RESOLUTION_H
        box_data.append(daily_kwh)
        box_labels.append(CASE_LABELS[case_name])
        box_colors.append(CASE_COLORS[case_name])

    bp = ax.boxplot(
        box_data, patch_artist=True, widths=0.5, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=3),
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for ml in bp["medians"]:
        ml.set_color("black")
        ml.set_linewidth(1.2)

    ax.set_xticklabels(box_labels, fontsize=GLOBAL_FONTSIZE)
    _style_ax(ax, ylabel="Daily Grid Import (kWh)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    return fig


# ===========================
# FIGURE 2c — Grid import time-series boxplot
# ===========================

def plot_grid_import_ts_boxplot(
    results_1day: dict,
    save_path: str = None,
    box_interval: int = 4,
):
    """Single-panel time-series boxplot, cases distinguished by color."""
    cases_present = [c for c in CASE_ORDER if c in results_1day]
    n_cases = len(cases_present)

    fig, ax = plt.subplots(figsize=(4.4, 2.8), dpi=300)

    # Width of each individual box and total group width
    group_width = RESOLUTION_H * box_interval * 0.8
    single_width = group_width / n_cases

    for i, case_name in enumerate(cases_present):
        case_data = results_1day[case_name]
        bids = sorted(case_data.keys())
        mat = np.array([case_data[bid]["grid_import_kw"] for bid in bids])
        n_steps = mat.shape[1]
        t_all = np.arange(n_steps) * RESOLUTION_H

        step_indices = np.arange(0, n_steps, box_interval)
        base_positions = t_all[step_indices]
        # Offset each case within the group
        offset = (i - (n_cases - 1) / 2) * single_width
        positions = base_positions + offset
        box_data = [mat[:, si] for si in step_indices]

        color = CASE_COLORS[case_name]
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=single_width * 0.85,
            patch_artist=True,
            showfliers=False,
            showmeans=False,
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for ml in bp["medians"]:
            ml.set_color("black")
            ml.set_linewidth(0.6)
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color(color)
                line.set_alpha(0.5)
                line.set_linewidth(0.5)

    # Peak-hour shading
    ax.axvspan(PEAK_HOUR_START, PEAK_HOUR_END, color="red", alpha=0.06)
    ax.axhline(0, color="grey", lw=0.4, ls="--")

    # Legend via proxy patches
    legend_handles = [
        Patch(facecolor=CASE_COLORS[c], alpha=0.5, label=CASE_LABELS[c])
        for c in cases_present
    ]
    ax.legend(handles=legend_handles, fontsize=GLOBAL_FONTSIZE - 1,
              frameon=False, loc="upper right", ncol=2)

    _style_ax(ax, xlabel="Time (h)", ylabel="Grid Import (kW)")
    ax.set_xlim([0, 24])
    ax.set_xticks(np.arange(0, 25, 2))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    return fig


# ===========================
# KPI COMPUTATION — fair comfort + flexibility metrics
# ===========================


# ===========================
# FIGURE 3 — KPI comparison (box plots + radar)
# ===========================

def plot_kpi_comparison(
    results: dict,
    save_path_bar: str = None,
    save_path_radar: str = None,
    n_days: int = 14,
):
    all_kpis = {}
    for case_name in CASE_ORDER:
        if case_name not in results:
            continue
        all_kpis[case_name] = _compute_kpis(results[case_name])

    # --- Add Grid Flexibility Index to each building ---
    for case_name, case_kpis in all_kpis.items():
        for bid, kpi in case_kpis.items():
            total_gi = kpi["total_grid_import_kwh"]
            peak_gi  = kpi["peak_hour_import_kwh"]
            kpi["grid_flexibility_index"] = (
                1.0 - peak_gi / total_gi if total_gi > 0 else 1.0
            )

    KPI_DEFS = [
        ("total_demand_kwh",       "Energy Consumption",   "kWh",  False, True),
        ("operating_cost_usd",     "Cost",                 "USD",  False, True),
        ("discomfort_hours",       "Discomfort Hours",     "h",    False, True),
        ("peak_demand_kw",         "Peak Load",            "kW",   False, False),
        ("peak_hour_import_kwh",   "On-Peak Energy",       "kWh",  False, True),
        ("grid_flexibility_index", "Grid Flexibility Index","–",   True,  False),
    ]
    # Tuple: (key, label, unit, higher_better, daily_divide)

    # ===== Panel A: box plots (2×3) =====
    fig_bar, axes = plt.subplots(2, 3, figsize=(4.4, 3.6), dpi=300)
    axes = axes.flatten()

    for idx, (kpi_key, kpi_label, unit, higher_better, daily) in enumerate(KPI_DEFS):
        ax = axes[idx]
        box_data, box_colors = [], []
        for case_name in CASE_ORDER:
            if case_name not in all_kpis:
                continue
            vals = [all_kpis[case_name][bid][kpi_key]
                    for bid in all_kpis[case_name]]
            if daily and n_days > 1:
                vals = [v / n_days for v in vals]
            box_data.append(vals)
            box_colors.append(CASE_COLORS[case_name])

        bp = ax.boxplot(
            box_data, patch_artist=True, widths=0.55, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=2),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for ml in bp["medians"]:
            ml.set_color("black")
            ml.set_linewidth(1.0)

        # X-axis: just numbers 1–5
        ax.set_xticklabels(
            [str(i + 1) for i in range(len(box_data))],
            fontsize=GLOBAL_FONTSIZE,
        )

        daily_tag = "/day" if daily and n_days > 1 else ""
        unit_str = f" ({unit}{daily_tag})" if unit and unit != "–" else ""
        _style_ax(ax, ylabel=f"{kpi_label}{unit_str}",
                  title=f"{SUBPLOT_LABELS[idx]} {kpi_label}")

    fig_bar.tight_layout(h_pad=0.5)
    if save_path_bar:
        fig_bar.savefig(save_path_bar, dpi=300, bbox_inches="tight")
        plt.close(fig_bar)
        print(f"  ✓ Saved: {save_path_bar}")

    # ===== Panel B: radar chart (same title updates) =====
    radar_kpis   = [k for k, _, _, _, _ in KPI_DEFS]
    radar_labels = [l for _, l, _, _, _ in KPI_DEFS]

    baseline_means = {}
    if "baseline" in all_kpis:
        for kpi_key in radar_kpis:
            baseline_means[kpi_key] = np.mean(
                [all_kpis["baseline"][bid][kpi_key]
                 for bid in all_kpis["baseline"]])

    n_kpis = len(radar_kpis)
    angles = np.linspace(0, 2 * np.pi, n_kpis, endpoint=False).tolist()
    angles += angles[:1]

    fig_radar, ax_r = plt.subplots(
        figsize=(4.4, 4.4), dpi=300, subplot_kw=dict(polar=True))

    for case_name in CASE_ORDER:
        if case_name not in all_kpis:
            continue
        values = []
        for kpi_key in radar_kpis:
            case_mean = np.mean([all_kpis[case_name][bid][kpi_key]
                                 for bid in all_kpis[case_name]])
            bm = baseline_means.get(kpi_key, 1)
            values.append(case_mean / bm if bm != 0 else 1)
        values += values[:1]
        color = CASE_COLORS[case_name]
        ax_r.plot(angles, values, "o-", lw=1.2, color=color,
                  label=CASE_LABELS[case_name], markersize=3)
        ax_r.fill(angles, values, color=color, alpha=0.08)

    ax_r.set_thetagrids(np.degrees(angles[:-1]), radar_labels,
                        fontsize=GLOBAL_FONTSIZE)
    ax_r.set_ylim(0, None)
    ax_r.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.08),
        fontsize=GLOBAL_FONTSIZE - 1, frameon=False,
        ncol=3, columnspacing=1.0,
    )
    ax_r.tick_params(labelsize=GLOBAL_FONTSIZE)
    ref_circle = [1.0] * (n_kpis + 1)
    ax_r.plot(angles, ref_circle, "k--", lw=0.8, alpha=0.4)

    fig_radar.tight_layout()
    if save_path_radar:
        fig_radar.savefig(save_path_radar, dpi=300, bbox_inches="tight")
        plt.close(fig_radar)
        print(f"  ✓ Saved: {save_path_radar}")

    return fig_bar, fig_radar, all_kpis


# ===========================
# SUMMARY TABLE
# ===========================

def _compute_kpis(case_data: dict, dt_h: float = RESOLUTION_H):
    kpis = {}
    for bid, d in case_data.items():
        total_demand_kwh = np.sum(d["total_demand_kw"]) * dt_h
        hvac_energy_kwh  = np.sum(d["hvac_power_kw"]) * dt_h
        pv_gen_kwh       = np.sum(d["pv_generation_kw"]) * dt_h
        grid_import_kwh  = np.sum(d["grid_import_kw"]) * dt_h
        curtail_kwh      = np.sum(d["curtailment_kw"]) * dt_h

        prices      = np.array(d["electricity_price"])
        grid_import = np.array(d["grid_import_kw"])
        cost_dollar = np.sum(grid_import * prices * dt_h) / 100.0

        # COMFORT: against BASE setpoints (fair across all cases)
        z         = np.array(d["zone_temperature"])
        base_cool = np.array(d["base_cooling_setpoint"])
        base_heat = np.array(d["base_heating_setpoint"])
        discomfort_steps = np.sum(
            (z > base_cool + 0.5) | (z < base_heat - 0.5))
        discomfort_hours = discomfort_steps * dt_h

        # FLEXIBILITY
        gi = np.array(d["grid_import_kw"])
        peak_demand = float(np.max(gi)) if len(gi) > 0 else 0.0

        is_peak = np.array(d["is_peak"], dtype=bool)
        peak_hour_import_kwh = float(np.sum(gi[is_peak]) * dt_h) \
            if is_peak.any() else 0.0
        total_grid_import_kwh = float(np.sum(gi) * dt_h)

        # Grid Flexibility Index
        grid_flexibility_index = (
            1.0 - peak_hour_import_kwh / total_grid_import_kwh
            if total_grid_import_kwh > 0 else 1.0
        )

        pv_used = pv_gen_kwh - curtail_kwh
        self_consumption = (pv_used / pv_gen_kwh) if pv_gen_kwh > 0 else 0.0

        kpis[bid] = {
            "total_demand_kwh": total_demand_kwh,
            "hvac_energy_kwh": hvac_energy_kwh,
            "operating_cost_usd": cost_dollar,
            "discomfort_hours": discomfort_hours,
            "peak_demand_kw": peak_demand,
            "peak_hour_import_kwh": peak_hour_import_kwh,
            "total_grid_import_kwh": total_grid_import_kwh,
            "grid_flexibility_index": grid_flexibility_index,
            "self_consumption": self_consumption,
            "pv_gen_kwh": pv_gen_kwh,
            "curtailment_kwh": curtail_kwh,
        }
    return kpis


def print_kpi_summary(all_kpis: dict, n_days: int = 1):
    kpi_keys = [
        ("total_demand_kwh",       "Energy Consumption",  True),
        ("operating_cost_usd",     "Cost ($)",            True),
        ("discomfort_hours",       "Discomfort (h)",      True),
        ("peak_demand_kw",         "Peak Load (kW)",      False),
        ("peak_hour_import_kwh",   "On-Peak Energy (kWh)",True),
        ("total_grid_import_kwh",  "Grid Import (kWh)",   True),
        ("grid_flexibility_index", "Grid Flex. Index",    False),
    ]
    # (key, label, daily_divide)

    header = f"{'KPI':<24s}"
    for cn in CASE_ORDER:
        if cn in all_kpis:
            header += f" | {CASE_LABELS[cn]:>22s}"
    print("=" * len(header))
    daily_tag = " (daily mean ± std)" if n_days > 1 else " (mean ± std)"
    print(f"KPI SUMMARY{daily_tag} across 20 buildings")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for key, label, daily in kpi_keys:
        row = f"{label:<24s}"
        baseline_mean = None
        for cn in CASE_ORDER:
            if cn not in all_kpis:
                continue
            vals = [all_kpis[cn][bid][key] for bid in all_kpis[cn]]
            if daily and n_days > 1:
                vals = [v / n_days for v in vals]
            m, s = np.mean(vals), np.std(vals)
            if cn == "baseline":
                baseline_mean = m
                row += f" | {m:>8.2f} ± {s:>5.2f}      "
            else:
                pct = ((m - baseline_mean) / baseline_mean * 100
                       if baseline_mean else 0)
                sign = "+" if pct > 0 else ""
                row += f" | {m:>8.2f} ± {s:>5.2f} ({sign}{pct:.1f}%)"
        print(row)
    print("=" * len(header))


def print_flexibility_summary(all_kpis: dict, n_days: int = 1):
    flex_keys = [
        ("peak_demand_kw",         "Peak Load (kW)",       False),
        ("peak_hour_import_kwh",   "On-Peak Energy (kWh)", True),
        ("total_grid_import_kwh",  "Grid Import (kWh)",    True),
        ("grid_flexibility_index", "Grid Flex. Index",     False),
    ]

    print("\n" + "=" * 90)
    daily_tag = " (daily values)" if n_days > 1 else ""
    print(f"FLEXIBILITY SUMMARY{daily_tag} — Change vs Baseline (mean across 20 buildings)")
    print("=" * 90)

    if "baseline" not in all_kpis:
        print("  ⚠ Baseline not found.")
        return

    header = f"{'Metric':<24s} | {'Baseline':>10s}"
    for cn in CASE_ORDER:
        if cn in all_kpis and cn != "baseline":
            header += f" | {CASE_LABELS[cn]:>16s}"
    print(header)
    print("-" * len(header))

    for key, label, daily in flex_keys:
        bl_vals = [all_kpis["baseline"][bid][key]
                   for bid in all_kpis["baseline"]]
        if daily and n_days > 1:
            bl_vals = [v / n_days for v in bl_vals]
        bl_mean = np.mean(bl_vals)

        row = f"{label:<24s} | {bl_mean:>10.2f}"
        for cn in CASE_ORDER:
            if cn not in all_kpis or cn == "baseline":
                continue
            vals = [all_kpis[cn][bid][key] for bid in all_kpis[cn]]
            if daily and n_days > 1:
                vals = [v / n_days for v in vals]
            cm = np.mean(vals)
            delta = cm - bl_mean
            pct = (delta / bl_mean * 100) if bl_mean != 0 else 0
            sign = "+" if delta > 0 else ""
            row += f" | {sign}{delta:>7.2f} ({sign}{pct:.1f}%)"
        print(row)

    print("=" * 90)

# ===========================
# MAIN
# ===========================

def main():
    PROJECT_ROOT_PATH_LOCAL = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_PATH_LOCAL = os.path.dirname(
        os.path.dirname(PROJECT_ROOT_PATH_LOCAL))

    results_dir = os.path.join(
        PROJECT_ROOT_PATH_LOCAL, "examples",
        "DR_Flexibility_Study", "results")
    os.makedirs(results_dir, exist_ok=True)

    pkl_1day = os.path.join(results_dir, "results_1day.pkl")

    if not os.path.exists(pkl_1day):
        print(f"✗ {pkl_1day} not found. Run run_study.py first.")
        return

    with open(pkl_1day, "rb") as f:
        results_1day = pickle.load(f)

    # --- Figure 1: single-building deep-dive (now includes grid import as subplot e) ---
    print("\nFigure 1: Single-building one-day deep dive (5 subplots)")
    for case_name in CASE_ORDER:
        if case_name in results_1day:
            plot_single_building_oneday(
                results_1day,
                building_id="SFH_1",
                case_name=case_name,
                save_path=os.path.join(
                    results_dir, f"fig1_deepdive_{case_name}.png"),
            )

    # --- Figure 2a: grid import line plot (standalone) ---
    print("\nFigure 2a: Grid import line plot (mean)")
    plot_grid_import_line(
        results_1day,
        save_path=os.path.join(results_dir, "fig2a_grid_import_line.png"),
    )

    # --- Figure 2b: grid import boxplot ---
    print("\nFigure 2b: Grid import boxplot")
    plot_grid_import_boxplot(
        results_1day,
        save_path=os.path.join(results_dir, "fig2b_grid_import_boxplot.png"),
    )

    # --- Figure 2c: grid import time-series boxplot ---
    print("\nFigure 2c: Grid import time-series boxplot")
    plot_grid_import_ts_boxplot(
        results_1day,
        save_path=os.path.join(results_dir,
                               "fig2c_grid_import_ts_boxplot.png"),
        box_interval=4,  # one box per hour
    )

    # --- 1-day KPIs + flexibility summary ---
    print("\n1-Day KPI Summary")
    all_kpis_1day = {}
    for case_name in CASE_ORDER:
        if case_name in results_1day:
            all_kpis_1day[case_name] = _compute_kpis(
                results_1day[case_name])
    print_kpi_summary(all_kpis_1day)
    print_flexibility_summary(all_kpis_1day)

    # --- Figure 3: two-week KPIs ---
    pkl_2weeks = os.path.join(results_dir, "results_2weeks.pkl")
    if os.path.exists(pkl_2weeks):
        with open(pkl_2weeks, "rb") as f:
            results_2weeks = pickle.load(f)
        print("\nFigure 3: Two-week KPI comparison")
        fig_bar, fig_radar, all_kpis_2w = plot_kpi_comparison(
            results_2weeks,
            save_path_bar=os.path.join(results_dir,
                                       "fig3a_kpi_boxplots.png"),
            save_path_radar=os.path.join(results_dir,
                                         "fig3b_kpi_radar.png"),
        )
        print('-----------2weeks-------')
        print_kpi_summary(all_kpis_2w)
        print_flexibility_summary(all_kpis_2w)

    try:
        plt.show()
    except Exception:
        print("Headless mode — check saved PNGs.")


if __name__ == "__main__":
    main()