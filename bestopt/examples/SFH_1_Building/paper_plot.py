import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams


def calculate_analytical_metrics(timesteps, data_dict, dt_hours=0.25):
    """
    Calculate comprehensive analytical metrics from simulation data.

    Parameters:
    -----------
    timesteps : array-like
        Time steps array
    data_dict : dict
        Dictionary containing all simulation data
    dt_hours : float
        Time step duration in hours (default 0.25 for 15-min intervals)

    Returns:
    --------
    metrics : dict
        Dictionary containing all calculated metrics
    """
    metrics = {}
    n_steps = len(timesteps)

    # ===== 1. ENERGY METRICS =====
    # Convert to numpy arrays
    total_load = np.asarray(data_dict.get('total_building_load', np.zeros(n_steps)))
    pv_gen = np.asarray(data_dict.get('pv_generation', np.zeros(n_steps)))
    net_grid = np.asarray(data_dict.get('net_grid', np.zeros(n_steps)))
    hvac_power = np.asarray(data_dict.get('hvac_power_kw', np.zeros(n_steps)))
    ev_charging = np.asarray(data_dict.get('ev_charging', np.zeros(n_steps)))
    battery_power = np.asarray(data_dict.get('battery_power', np.zeros(n_steps)))
    curtailment = np.asarray(data_dict.get('curtailment', np.zeros(n_steps)))

    # Total energy consumption (kWh)
    metrics['total_energy_kwh'] = np.sum(total_load) * dt_hours
    metrics['hvac_energy_kwh'] = np.sum(hvac_power) * dt_hours
    metrics['ev_energy_kwh'] = np.sum(np.maximum(ev_charging, 0)) * dt_hours

    # PV generation (kWh)
    metrics['pv_generation_kwh'] = np.sum(pv_gen) * dt_hours

    # Grid exchange (kWh)
    grid_import = np.maximum(net_grid, 0)
    grid_export = np.maximum(-net_grid, 0)
    metrics['grid_import_kwh'] = np.sum(grid_import) * dt_hours
    metrics['grid_export_kwh'] = np.sum(grid_export) * dt_hours
    metrics['net_grid_kwh'] = metrics['grid_import_kwh'] - metrics['grid_export_kwh']

    # Self-consumption and self-sufficiency
    pv_self_consumed = metrics['pv_generation_kwh'] - metrics['grid_export_kwh']
    if metrics['pv_generation_kwh'] > 0:
        metrics['self_consumption_ratio'] = pv_self_consumed / metrics['pv_generation_kwh']
    else:
        metrics['self_consumption_ratio'] = 0.0

    if metrics['total_energy_kwh'] > 0:
        metrics['self_sufficiency_ratio'] = pv_self_consumed / metrics['total_energy_kwh']
    else:
        metrics['self_sufficiency_ratio'] = 0.0

    # Curtailment
    metrics['curtailment_kwh'] = np.sum(curtailment) * dt_hours

    # ===== 2. COMFORT METRICS =====
    zone_temp = np.asarray(data_dict.get('zone_temperature', np.zeros(n_steps)))
    cooling_sp = np.asarray(data_dict.get('cooling_setpoint', np.ones(n_steps) * 26))
    heating_sp = np.asarray(data_dict.get('heating_setpoint', np.ones(n_steps) * 20))

    # Temperature violations
    cooling_violations = np.maximum(zone_temp - cooling_sp, 0)
    heating_violations = np.maximum(heating_sp - zone_temp, 0)

    metrics['cooling_violation_hours'] = np.sum(cooling_violations > 0) * dt_hours
    metrics['heating_violation_hours'] = np.sum(heating_violations > 0) * dt_hours
    metrics['total_violation_hours'] = metrics['cooling_violation_hours'] + metrics['heating_violation_hours']

    # Comfort band adherence
    in_comfort_band = (zone_temp >= heating_sp) & (zone_temp <= cooling_sp)
    metrics['comfort_adherence_pct'] = np.mean(in_comfort_band) * 100

    # Temperature deviation metrics
    metrics['max_cooling_deviation_c'] = np.max(cooling_violations)
    metrics['max_heating_deviation_c'] = np.max(heating_violations)
    metrics['avg_temp_c'] = np.mean(zone_temp)
    metrics['temp_std_c'] = np.std(zone_temp)

    # Predicted Mean Vote approximation (simplified)
    # PMV = 0 is ideal, negative is cold, positive is warm
    comfort_midpoint = (cooling_sp + heating_sp) / 2
    metrics['avg_pmv_deviation'] = np.mean(np.abs(zone_temp - comfort_midpoint))

    # ===== 3. FLEXIBILITY METRICS =====
    is_peak = np.asarray(data_dict.get('is_peak', np.zeros(n_steps, dtype=bool)))

    # Peak vs off-peak consumption
    if np.any(is_peak):
        metrics['peak_energy_kwh'] = np.sum(total_load[is_peak]) * dt_hours
        metrics['peak_hours'] = np.sum(is_peak) * dt_hours
    else:
        metrics['peak_energy_kwh'] = 0.0
        metrics['peak_hours'] = 0.0

    if np.any(~is_peak):
        metrics['offpeak_energy_kwh'] = np.sum(total_load[~is_peak]) * dt_hours
        metrics['offpeak_hours'] = np.sum(~is_peak) * dt_hours
    else:
        metrics['offpeak_energy_kwh'] = 0.0
        metrics['offpeak_hours'] = 0.0

    # Peak demand
    metrics['peak_demand_kw'] = np.max(total_load)
    metrics['avg_demand_kw'] = np.mean(total_load)
    metrics['load_factor'] = metrics['avg_demand_kw'] / metrics['peak_demand_kw'] if metrics[
                                                                                         'peak_demand_kw'] > 0 else 0

    # Peak grid import
    metrics['peak_grid_import_kw'] = np.max(grid_import)
    if np.any(is_peak):
        metrics['peak_hour_grid_import_kw'] = np.max(grid_import[is_peak])
    else:
        metrics['peak_hour_grid_import_kw'] = 0.0

    # Battery utilization
    bat_soc = np.asarray(data_dict.get('battery_soc', np.zeros(n_steps)))
    metrics['battery_cycles'] = np.sum(np.abs(np.diff(bat_soc))) / 2  # Approximate cycles
    metrics['battery_soc_min'] = np.min(bat_soc) * 100
    metrics['battery_soc_max'] = np.max(bat_soc) * 100
    metrics['battery_soc_avg'] = np.mean(bat_soc) * 100

    # EV flexibility
    ev_tesla_soc = np.asarray(data_dict.get('ev_tesla_soc', np.zeros(n_steps)))
    ev_nissan_soc = np.asarray(data_dict.get('ev_nissan_soc', np.zeros(n_steps)))
    metrics['ev_tesla_final_soc'] = ev_tesla_soc[-1] * 100 if len(ev_tesla_soc) > 0 else 0
    metrics['ev_nissan_final_soc'] = ev_nissan_soc[-1] * 100 if len(ev_nissan_soc) > 0 else 0

    # ===== 4. ECONOMIC METRICS =====
    elec_price = np.asarray(data_dict.get('electricity_price', np.ones(n_steps) * 0.15))

    # Energy costs
    metrics['total_energy_cost'] = np.sum(grid_import * elec_price) * dt_hours

    # Revenue from export (assuming feed-in tariff = 50% of retail price)
    fit_ratio = 0.5
    metrics['export_revenue'] = np.sum(grid_export * elec_price * fit_ratio) * dt_hours
    metrics['net_energy_cost'] = metrics['total_energy_cost'] - metrics['export_revenue']

    # Peak hour costs
    if np.any(is_peak):
        metrics['peak_energy_cost'] = np.sum(grid_import[is_peak] * elec_price[is_peak]) * dt_hours
    else:
        metrics['peak_energy_cost'] = 0.0

    # Average price paid
    if metrics['grid_import_kwh'] > 0:
        metrics['avg_price_paid'] = metrics['total_energy_cost'] / metrics['grid_import_kwh']
    else:
        metrics['avg_price_paid'] = 0.0

    # ===== 5. HVAC PERFORMANCE METRICS =====
    hvac_thermal = np.asarray(data_dict.get('hvac_thermal_load', np.zeros(n_steps))) / 1000  # Convert to kW
    hvac_elec = np.asarray(data_dict.get('hvac_power', np.zeros(n_steps))) / 1000

    # COP calculation (only when HVAC is active)
    active_mask = hvac_elec > 0.01
    if np.any(active_mask):
        cop_values = hvac_thermal[active_mask] / hvac_elec[active_mask]
        metrics['hvac_avg_cop'] = np.mean(cop_values)
        metrics['hvac_max_cop'] = np.max(cop_values)
    else:
        metrics['hvac_avg_cop'] = 0.0
        metrics['hvac_max_cop'] = 0.0

    metrics['hvac_runtime_hours'] = np.sum(hvac_elec > 0.01) * dt_hours

    return metrics


def print_metrics_summary(metrics):
    """
    Print a formatted summary of all metrics.
    """
    print("\n" + "=" * 70)
    print("SIMULATION ANALYTICAL RESULTS")
    print("=" * 70)

    print("\n1. ENERGY METRICS")
    print("-" * 40)
    print(f"  Total Building Load:     {metrics['total_energy_kwh']:.2f} kWh")
    print(
        f"    - HVAC:                {metrics['hvac_energy_kwh']:.2f} kWh ({metrics['hvac_energy_kwh'] / metrics['total_energy_kwh'] * 100:.1f}%)")
    print(
        f"    - EV Charging:         {metrics['ev_energy_kwh']:.2f} kWh ({metrics['ev_energy_kwh'] / metrics['total_energy_kwh'] * 100:.1f}%)")
    print(f"  PV Generation:           {metrics['pv_generation_kwh']:.2f} kWh")
    print(f"  Grid Import:             {metrics['grid_import_kwh']:.2f} kWh")
    print(f"  Grid Export:             {metrics['grid_export_kwh']:.2f} kWh")
    print(f"  Net Grid Exchange:       {metrics['net_grid_kwh']:.2f} kWh")
    print(f"  Self-Consumption Ratio:  {metrics['self_consumption_ratio'] * 100:.1f}%")
    print(f"  Self-Sufficiency Ratio:  {metrics['self_sufficiency_ratio'] * 100:.1f}%")
    print(f"  Curtailment:             {metrics['curtailment_kwh']:.2f} kWh")

    print("\n2. COMFORT METRICS")
    print("-" * 40)
    print(f"  Comfort Band Adherence:  {metrics['comfort_adherence_pct']:.1f}%")
    print(f"  Cooling Violation Hours: {metrics['cooling_violation_hours']:.2f} h")
    print(f"  Heating Violation Hours: {metrics['heating_violation_hours']:.2f} h")
    print(f"  Max Cooling Deviation:   {metrics['max_cooling_deviation_c']:.2f} °C")
    print(f"  Max Heating Deviation:   {metrics['max_heating_deviation_c']:.2f} °C")
    print(f"  Average Temperature:     {metrics['avg_temp_c']:.2f} °C")
    print(f"  Temperature Std Dev:     {metrics['temp_std_c']:.2f} °C")

    print("\n3. FLEXIBILITY METRICS")
    print("-" * 40)
    print(f"  Peak Hours:              {metrics['peak_hours']:.1f} h")
    print(f"  Peak Energy Consumption: {metrics['peak_energy_kwh']:.2f} kWh")
    print(f"  Off-Peak Energy:         {metrics['offpeak_energy_kwh']:.2f} kWh")
    print(f"  Peak Demand:             {metrics['peak_demand_kw']:.2f} kW")
    print(f"  Average Demand:          {metrics['avg_demand_kw']:.2f} kW")
    print(f"  Load Factor:             {metrics['load_factor'] * 100:.1f}%")
    print(f"  Peak Grid Import:        {metrics['peak_grid_import_kw']:.2f} kW")
    print(f"  Battery SOC Range:       {metrics['battery_soc_min']:.1f}% - {metrics['battery_soc_max']:.1f}%")
    print(f"  Battery Cycles:          {metrics['battery_cycles']:.2f}")
    print(f"  EV Tesla Final SOC:      {metrics['ev_tesla_final_soc']:.1f}%")
    print(f"  EV Nissan Final SOC:     {metrics['ev_nissan_final_soc']:.1f}%")

    print("\n4. ECONOMIC METRICS")
    print("-" * 40)
    print(f"  Total Energy Cost:       ${metrics['total_energy_cost']:.2f}")
    print(f"  Export Revenue:          ${metrics['export_revenue']:.2f}")
    print(f"  Net Energy Cost:         ${metrics['net_energy_cost']:.2f}")
    print(f"  Peak Hour Cost:          ${metrics['peak_energy_cost']:.2f}")
    print(f"  Average Price Paid:      ${metrics['avg_price_paid']:.3f}/kWh")

    print("\n5. HVAC PERFORMANCE")
    print("-" * 40)
    print(f"  Average COP:             {metrics['hvac_avg_cop']:.2f}")
    print(f"  Maximum COP:             {metrics['hvac_max_cop']:.2f}")
    print(f"  Runtime Hours:           {metrics['hvac_runtime_hours']:.2f} h")

    print("\n" + "=" * 70)


def create_metrics_table_figure(metrics, save_path='metrics_table.pdf'):
    """
    Create a figure with a table of key metrics for paper inclusion.
    """
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.axis('off')

    # Define table data
    categories = [
        ('Energy', [
            ('Total Load', f"{metrics['total_energy_kwh']:.2f} kWh"),
            ('PV Generation', f"{metrics['pv_generation_kwh']:.2f} kWh"),
            ('Grid Import', f"{metrics['grid_import_kwh']:.2f} kWh"),
            ('Self-Consumption', f"{metrics['self_consumption_ratio'] * 100:.1f}%"),
            ('Self-Sufficiency', f"{metrics['self_sufficiency_ratio'] * 100:.1f}%"),
        ]),
        ('Comfort', [
            ('Comfort Adherence', f"{metrics['comfort_adherence_pct']:.1f}%"),
            ('Violation Hours', f"{metrics['total_violation_hours']:.2f} h"),
            ('Avg. Temperature', f"{metrics['avg_temp_c']:.2f} °C"),
        ]),
        ('Flexibility', [
            ('Peak Demand', f"{metrics['peak_demand_kw']:.2f} kW"),
            ('Load Factor', f"{metrics['load_factor'] * 100:.1f}%"),
            ('Battery Cycles', f"{metrics['battery_cycles']:.2f}"),
        ]),
        ('Economic', [
            ('Net Cost', f"${metrics['net_energy_cost']:.2f}"),
            ('Avg. Price', f"${metrics['avg_price_paid']:.3f}/kWh"),
        ]),
    ]

    # Create table data
    cell_text = []
    row_colors = []
    color_map = {
        'Energy': '#e6f3ff',
        'Comfort': '#fff2e6',
        'Flexibility': '#e6ffe6',
        'Economic': '#ffe6e6',
    }

    for category, items in categories:
        for i, (name, value) in enumerate(items):
            if i == 0:
                cell_text.append([category, name, value])
            else:
                cell_text.append(['', name, value])
            row_colors.append(color_map[category])

    # Create table
    table = ax.table(
        cellText=cell_text,
        colLabels=['Category', 'Metric', 'Value'],
        loc='center',
        cellLoc='left',
        colWidths=[0.25, 0.45, 0.30],
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color rows
    for i, color in enumerate(row_colors):
        for j in range(3):
            table[(i + 1, j)].set_facecolor(color)

    # Style header
    for j in range(3):
        table[(0, j)].set_facecolor('#4472c4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    plt.title('Simulation Performance Metrics', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    return fig


def create_paper_figure(timesteps, data_dict, save_path='simulation_results.pdf'):
    """
    Create a static figure for paper with 4 subplots showing one day simulation results.

    Parameters:
    -----------
    timesteps : array-like
        Time steps (0 to 95 for 15-minute intervals over 24 hours)
    data_dict : dict
        Dictionary containing all the data arrays with keys:
        - zone_temperature, cooling_setpoint, heating_setpoint
        - hvac_thermal_load, hvac_power
        - battery_soc, ev_tesla_soc, ev_nissan_soc
        - lighting, cooking, pc, tv, hvac_power_kw
        - total_building_load, grid_import, battery_power
        - pv_generation, net_grid
        - is_peak (boolean array for peak hours)
    save_path : str
        Path to save the PDF figure
    """

    # Set global font sizes
    rcParams['font.size'] = 8
    rcParams['axes.labelsize'] = 8
    rcParams['axes.titlesize'] = 9
    rcParams['xtick.labelsize'] = 7
    rcParams['ytick.labelsize'] = 7
    rcParams['legend.fontsize'] = 7
    rcParams['figure.titlesize'] = 9

    # Create figure with 4 subplots with height ratios 1:1:1:1.5
    fig, axes = plt.subplots(4, 1, figsize=(7, 9), dpi=300,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1.5]})
    fig.subplots_adjust(hspace=0.2, top=0.96, bottom=0.07, left=0.12, right=0.92)

    # Convert timesteps to hours for x-axis
    hours = np.asarray(timesteps) * 0.25  # Assuming 15-minute intervals

    # Extract peak hours for shading
    peak_regions = []
    if 'is_peak' in data_dict:
        is_peak = data_dict['is_peak']
        peak_start = None
        for i in range(len(is_peak)):
            if is_peak[i] and peak_start is None:
                peak_start = hours[i]
            elif not is_peak[i] and peak_start is not None:
                peak_regions.append((peak_start, hours[i - 1] if i > 0 else hours[i]))
                peak_start = None
        if peak_start is not None:
            peak_regions.append((peak_start, hours[-1]))

    # Color palette for consistency
    colors = {
        'temperature': '#000000',  # Black for zone temperature
        'setpoint': '#888888',  # Gray for setpoints
        'thermal_load': '#1f77b4',  # Blue
        'hvac_power': '#d62728',  # Red
        'soc': '#2ca02c',  # Green for SOC
        'net_load': '#ff7f0e',  # Orange for net load
        'hvac': '#9467bd',  # Purple for HVAC
        'lighting': '#FFD700',  # Gold for lighting
        'other': '#17becf',  # Cyan for other loads
        'pv': '#2ca02c',  # Green for PV generation
        'total': '#000000',  # Black for total load
    }

    # ===== Subplot 1: Zone Temperature and Setpoints =====
    ax1 = axes[0]

    # Add peak hour shading
    for start, end in peak_regions:
        ax1.axvspan(start, end, alpha=0.15, color='red', zorder=0)

    ax1.plot(hours, data_dict['zone_temperature'], '-', linewidth=1.5,
             color=colors['temperature'], label='Zone Temperature', zorder=3)
    ax1.plot(hours, data_dict['cooling_setpoint'], '--', linewidth=1.0,
             color=colors['setpoint'], label='Cooling Setpoint', zorder=2)
    ax1.plot(hours, data_dict['heating_setpoint'], '--', linewidth=1.0,
             color=colors['setpoint'], label='Heating Setpoint', zorder=2)

    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlim([0, 24])
    ax1.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
    ax1.legend(loc='upper left', frameon=False, ncol=1)
    ax1.set_title('(a) Zone Temperature Control', fontweight='bold', loc='left')

    # ===== Subplot 2: HVAC Thermal Load and Power =====
    ax2 = axes[1]

    # Add peak hour shading
    for start, end in peak_regions:
        ax2.axvspan(start, end, alpha=0.15, color='red', zorder=0)

    ax2.plot(hours, np.asarray(data_dict['hvac_thermal_load']) / 1000, '-',
             linewidth=1.5, color=colors['thermal_load'], label='Thermal Load', zorder=3)
    ax2.plot(hours, np.asarray(data_dict['hvac_power']) / 1000, '-',
             linewidth=1.5, color=colors['hvac_power'], label='Electrical Power', zorder=3)

    ax2.set_ylabel('Power (kW)')
    ax2.set_xlim([0, 24])
    ax2.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
    ax2.legend(loc='upper left', frameon=False)
    ax2.set_title('(b) HVAC Performance', fontweight='bold', loc='left')

    # ===== Subplot 3: SOC and Net Load =====
    ax3 = axes[2]

    # Add peak hour shading (only add label once)
    for idx, (start, end) in enumerate(peak_regions):
        ax3.axvspan(start, end, alpha=0.15, color='red', zorder=0)

    # All SOCs
    soc_bat = np.asarray(data_dict['battery_soc']) * 100.0
    soc_tesla = np.asarray(data_dict['ev_tesla_soc']) * 100.0
    soc_nissan = np.asarray(data_dict['ev_nissan_soc']) * 100.0

    # Net grid power
    net_grid = np.asarray(data_dict.get('net_grid', np.zeros_like(hours)))

    # Plot SOCs
    ax3.plot(hours, soc_bat, '-', linewidth=1.5, color='#2ca02c',
             label='Battery SOC', zorder=3)
    ax3.plot(hours, soc_tesla, '-', linewidth=1.5, color='#1f77b4',
             label='EV Tesla SOC', zorder=3)
    ax3.plot(hours, soc_nissan, '-', linewidth=1.5, color='#17becf',
             label='EV Nissan SOC', zorder=3)

    ax3.set_ylabel('SOC (%)')
    ax3.set_xlim([0, 24])
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, linewidth=0.5, zorder=0)

    # Create legend with peak hours patch
    peak_patch = mpatches.Patch(color='red', alpha=0.15, label='Peak Hours')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend([peak_patch] + handles, ['Peak Hours'] + labels,
               loc='lower left', frameon=False, ncol=2)

    ax3.set_title('(c) Energy Storage SOC', fontweight='bold', loc='left')

    ax4 = axes[3]

    # Add peak hour shading
    for start, end in peak_regions:
        ax4.axvspan(start, end, alpha=0.15, color='red', zorder=0)

    # Prepare load components
    lighting = np.asarray(data_dict.get('lighting', np.zeros_like(hours)))
    cooking = np.asarray(data_dict.get('cooking', np.zeros_like(hours)))
    pc = np.asarray(data_dict.get('pc', np.zeros_like(hours)))
    tv = np.asarray(data_dict.get('tv', np.zeros_like(hours)))
    hvac = np.asarray(data_dict.get('hvac_power_kw', np.zeros_like(hours)))

    # EV charging load (already in kW from power_balance)
    ev_charging = np.asarray(data_dict.get('ev_charging', np.zeros_like(hours)))
    ev_charging = np.maximum(ev_charging, 0)  # Only positive values (charging)

    # Aggregate other loads (PC, cooking, TV)
    other_loads = cooking + pc + tv

    # Total building load (now including EV)
    total_building = np.asarray(data_dict.get('total_building_load',
                                              lighting + other_loads + hvac))
    total_with_ev = total_building + ev_charging

    # PV generation (to be plotted as negative)
    pv_gen = np.asarray(data_dict.get('pv_generation', np.zeros_like(hours)))
    pv_gen = np.nan_to_num(np.maximum(pv_gen, 0.0))

    # Plot stacked areas for loads (positive region)
    # Layer 1: HVAC (bottom)
    ax4.fill_between(hours, 0, hvac, alpha=0.7, color=colors['hvac'],
                     label='HVAC', zorder=2)

    # Layer 2: Lighting
    stack1 = hvac
    ax4.fill_between(hours, stack1, stack1 + lighting, alpha=0.7,
                     color=colors['lighting'], label='Lighting', zorder=2)

    # Layer 3: Other loads (cooking, PC, TV)
    stack2 = stack1 + lighting
    ax4.fill_between(hours, stack2, stack2 + other_loads,
                     alpha=0.7, color=colors['other'], label='Other', zorder=2)

    # Layer 4: EV Charging (top of stack)
    stack3 = stack2 + other_loads
    ax4.fill_between(hours, stack3, stack3 + ev_charging,
                     alpha=0.7, color='#e377c2', label='EV Charging', zorder=2)  # Pink/magenta

    # Plot total load line (including EV)
    ax4.plot(hours, total_with_ev, '-', linewidth=1.8, color=colors['total'],
             label='Total Load', zorder=4)

    # Plot PV generation as negative (below zero)
    ax4.fill_between(hours, 0, -pv_gen, alpha=0.7, color=colors['pv'],
                     label='PV Generation', zorder=2)
    ax4.plot(hours, -pv_gen, '-', linewidth=1.2, color=colors['pv'],
             alpha=0.8, zorder=3)

    # Add horizontal line at y=0
    ax4.axhline(y=0, color='black', linewidth=0.8, linestyle='-', zorder=1)

    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Power (kW)')
    ax4.set_xlim([0, 24])

    # Set y-axis limits to accommodate both positive loads and negative PV
    y_max = max(total_with_ev.max(), (stack3 + ev_charging).max()) * 1.1
    y_min = -pv_gen.max() * 1.15 if pv_gen.max() > 0 else -0.5
    ax4.set_ylim([y_min, y_max])

    ax4.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
    ax4.legend(loc='upper left', ncol=3, frameon=False, columnspacing=1)
    ax4.set_title('(d) Disaggregated Load Profile', fontweight='bold', loc='left')

    # Set x-axis ticks for all subplots
    for ax in axes:
        ax.set_xticks(np.arange(0, 25, 3))
        ax.tick_params(axis='both', which='major')

    # Only show x-axis label on bottom plot
    for ax in axes[:-1]:
        ax.set_xticklabels([])

    # Save figure
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def create_paper_figure_with_annotations(timesteps, data_dict, metrics, save_path='simulation_results_annotated.pdf'):
    """
    Create a paper figure with key metrics annotated on the plots.
    """
    # Set global font sizes
    rcParams['font.size'] = 8
    rcParams['axes.labelsize'] = 8
    rcParams['axes.titlesize'] = 9
    rcParams['xtick.labelsize'] = 7
    rcParams['ytick.labelsize'] = 7
    rcParams['legend.fontsize'] = 7
    rcParams['figure.titlesize'] = 9

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(7, 9.5), dpi=300,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1.5]})
    fig.subplots_adjust(hspace=0.25, top=0.96, bottom=0.07, left=0.12, right=0.88)

    hours = np.asarray(timesteps) * 0.25

    # Extract peak hours for shading
    peak_regions = []
    if 'is_peak' in data_dict:
        is_peak = data_dict['is_peak']
        peak_start = None
        for i in range(len(is_peak)):
            if is_peak[i] and peak_start is None:
                peak_start = hours[i]
            elif not is_peak[i] and peak_start is not None:
                peak_regions.append((peak_start, hours[i - 1] if i > 0 else hours[i]))
                peak_start = None
        if peak_start is not None:
            peak_regions.append((peak_start, hours[-1]))

    colors = {
        'temperature': '#000000',
        'setpoint': '#888888',
        'thermal_load': '#1f77b4',
        'hvac_power': '#d62728',
        'hvac': '#9467bd',
        'lighting': '#FFD700',
        'other': '#17becf',
        'pv': '#2ca02c',
        'total': '#000000',
    }

    # ===== Subplot 1: Temperature with comfort metrics =====
    ax1 = axes[0]
    for start, end in peak_regions:
        ax1.axvspan(start, end, alpha=0.15, color='red', zorder=0)

    ax1.plot(hours, data_dict['zone_temperature'], '-', linewidth=1.5,
             color=colors['temperature'], label='Zone Temperature', zorder=3)
    ax1.plot(hours, data_dict['cooling_setpoint'], '--', linewidth=1.0,
             color=colors['setpoint'], label='Cooling Setpoint', zorder=2)
    ax1.plot(hours, data_dict['heating_setpoint'], '--', linewidth=1.0,
             color=colors['setpoint'], label='Heating Setpoint', zorder=2)

    # Add annotation box with comfort metrics
    textstr = f"Comfort: {metrics['comfort_adherence_pct']:.1f}%\nViolations: {metrics['total_violation_hours']:.1f}h"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax1.text(0.98, 0.95, textstr, transform=ax1.transAxes, fontsize=7,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlim([0, 24])
    ax1.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
    ax1.legend(loc='upper left', frameon=False, ncol=1)
    ax1.set_title('(a) Zone Temperature Control', fontweight='bold', loc='left')

    # ===== Subplot 2: HVAC with COP annotation =====
    ax2 = axes[1]
    for start, end in peak_regions:
        ax2.axvspan(start, end, alpha=0.15, color='red', zorder=0)

    ax2.plot(hours, np.asarray(data_dict['hvac_thermal_load']) / 1000, '-',
             linewidth=1.5, color=colors['thermal_load'], label='Thermal Load', zorder=3)
    ax2.plot(hours, np.asarray(data_dict['hvac_power']) / 1000, '-',
             linewidth=1.5, color=colors['hvac_power'], label='Electrical Power', zorder=3)

    # Add HVAC metrics
    textstr = f"Avg COP: {metrics['hvac_avg_cop']:.2f}\nEnergy: {metrics['hvac_energy_kwh']:.1f} kWh"
    ax2.text(0.98, 0.95, textstr, transform=ax2.transAxes, fontsize=7,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    ax2.set_ylabel('Power (kW)')
    ax2.set_xlim([0, 24])
    ax2.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
    ax2.legend(loc='upper left', frameon=False)
    ax2.set_title('(b) HVAC Performance', fontweight='bold', loc='left')

    # ===== Subplot 3: SOC with flexibility metrics =====
    ax3 = axes[2]
    for idx, (start, end) in enumerate(peak_regions):
        ax3.axvspan(start, end, alpha=0.15, color='red', zorder=0)

    soc_bat = np.asarray(data_dict['battery_soc']) * 100.0
    soc_tesla = np.asarray(data_dict['ev_tesla_soc']) * 100.0
    soc_nissan = np.asarray(data_dict['ev_nissan_soc']) * 100.0

    ax3.plot(hours, soc_bat, '-', linewidth=1.5, color='#2ca02c',
             label='Battery SOC', zorder=3)
    ax3.plot(hours, soc_tesla, '-', linewidth=1.5, color='#1f77b4',
             label='EV Tesla SOC', zorder=3)
    ax3.plot(hours, soc_nissan, '-', linewidth=1.5, color='#17becf',
             label='EV Nissan SOC', zorder=3)

    # Add storage metrics
    textstr = f"Battery Cycles: {metrics['battery_cycles']:.2f}\nEV Final: {metrics['ev_tesla_final_soc']:.0f}%/{metrics['ev_nissan_final_soc']:.0f}%"
    ax3.text(0.98, 0.05, textstr, transform=ax3.transAxes, fontsize=7,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)

    ax3.set_ylabel('SOC (%)')
    ax3.set_xlim([0, 24])
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, linewidth=0.5, zorder=0)

    peak_patch = mpatches.Patch(color='red', alpha=0.15, label='Peak Hours')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend([peak_patch] + handles, ['Peak Hours'] + labels,
               loc='lower left', frameon=False, ncol=2)
    ax3.set_title('(c) Energy Storage SOC', fontweight='bold', loc='left')

    # ===== Subplot 4: Load profile with energy metrics =====
    ax4 = axes[3]
    for start, end in peak_regions:
        ax4.axvspan(start, end, alpha=0.15, color='red', zorder=0)

    lighting = np.asarray(data_dict.get('lighting', np.zeros_like(hours)))
    cooking = np.asarray(data_dict.get('cooking', np.zeros_like(hours)))
    pc = np.asarray(data_dict.get('pc', np.zeros_like(hours)))
    tv = np.asarray(data_dict.get('tv', np.zeros_like(hours)))
    hvac = np.asarray(data_dict.get('hvac_power_kw', np.zeros_like(hours)))
    ev_charging = np.asarray(data_dict.get('ev_charging', np.zeros_like(hours)))
    ev_charging = np.maximum(ev_charging, 0)
    other_loads = cooking + pc + tv
    total_building = np.asarray(data_dict.get('total_building_load', lighting + other_loads + hvac))
    total_with_ev = total_building + ev_charging
    pv_gen = np.asarray(data_dict.get('pv_generation', np.zeros_like(hours)))
    pv_gen = np.nan_to_num(np.maximum(pv_gen, 0.0))

    ax4.fill_between(hours, 0, hvac, alpha=0.7, color=colors['hvac'], label='HVAC', zorder=2)
    stack1 = hvac
    ax4.fill_between(hours, stack1, stack1 + lighting, alpha=0.7, color=colors['lighting'], label='Lighting', zorder=2)
    stack2 = stack1 + lighting
    ax4.fill_between(hours, stack2, stack2 + other_loads, alpha=0.7, color=colors['other'], label='Other', zorder=2)
    stack3 = stack2 + other_loads
    ax4.fill_between(hours, stack3, stack3 + ev_charging, alpha=0.7, color='#e377c2', label='EV Charging', zorder=2)
    ax4.plot(hours, total_with_ev, '-', linewidth=1.8, color=colors['total'], label='Total Load', zorder=4)
    ax4.fill_between(hours, 0, -pv_gen, alpha=0.7, color=colors['pv'], label='PV Generation', zorder=2)
    ax4.plot(hours, -pv_gen, '-', linewidth=1.2, color=colors['pv'], alpha=0.8, zorder=3)
    ax4.axhline(y=0, color='black', linewidth=0.8, linestyle='-', zorder=1)

    # Add energy metrics annotation
    textstr = (f"Total: {metrics['total_energy_kwh']:.1f} kWh | PV: {metrics['pv_generation_kwh']:.1f} kWh\n"
               f"Self-cons: {metrics['self_consumption_ratio'] * 100:.0f}% | Net cost: ${metrics['net_energy_cost']:.2f}")
    ax4.text(0.98, 0.95, textstr, transform=ax4.transAxes, fontsize=7,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Power (kW)')
    ax4.set_xlim([0, 24])
    y_max = max(total_with_ev.max(), (stack3 + ev_charging).max()) * 1.1
    y_min = -pv_gen.max() * 1.15 if pv_gen.max() > 0 else -0.5
    ax4.set_ylim([y_min, y_max])
    ax4.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
    ax4.legend(loc='upper left', ncol=3, frameon=False, columnspacing=1)
    ax4.set_title('(d) Disaggregated Load Profile', fontweight='bold', loc='left')

    for ax in axes:
        ax.set_xticks(np.arange(0, 25, 3))
        ax.tick_params(axis='both', which='major')
    for ax in axes[:-1]:
        ax.set_xticklabels([])

    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def export_metrics_to_latex(metrics, save_path='metrics_table.tex'):
    """
    Export metrics to LaTeX table format for direct inclusion in paper.
    """
    latex_content = r"""
\begin{table}[htbp]
\centering
\caption{Simulation Performance Metrics for One-Day Operation}
\label{tab:simulation_metrics}
\begin{tabular}{llr}
\toprule
\textbf{Category} & \textbf{Metric} & \textbf{Value} \\
\midrule
\multirow{5}{*}{Energy} 
    & Total Building Load & """ + f"{metrics['total_energy_kwh']:.2f}" + r""" kWh \\
    & PV Generation & """ + f"{metrics['pv_generation_kwh']:.2f}" + r""" kWh \\
    & Grid Import & """ + f"{metrics['grid_import_kwh']:.2f}" + r""" kWh \\
    & Self-Consumption Ratio & """ + f"{metrics['self_consumption_ratio'] * 100:.1f}" + r"""\% \\
    & Self-Sufficiency Ratio & """ + f"{metrics['self_sufficiency_ratio'] * 100:.1f}" + r"""\% \\
\midrule
\multirow{3}{*}{Comfort}
    & Comfort Band Adherence & """ + f"{metrics['comfort_adherence_pct']:.1f}" + r"""\% \\
    & Temperature Violation Hours & """ + f"{metrics['total_violation_hours']:.2f}" + r""" h \\
    & Average Zone Temperature & """ + f"{metrics['avg_temp_c']:.2f}" + r"""~°C \\
\midrule
\multirow{4}{*}{Flexibility}
    & Peak Demand & """ + f"{metrics['peak_demand_kw']:.2f}" + r""" kW \\
    & Load Factor & """ + f"{metrics['load_factor'] * 100:.1f}" + r"""\% \\
    & Battery Equivalent Cycles & """ + f"{metrics['battery_cycles']:.2f}" + r""" \\
    & Peak Hour Grid Import & """ + f"{metrics['peak_hour_grid_import_kw']:.2f}" + r""" kW \\
\midrule
\multirow{3}{*}{Economic}
    & Net Energy Cost & \$""" + f"{metrics['net_energy_cost']:.2f}" + r""" \\
    & Average Price Paid & \$""" + f"{metrics['avg_price_paid']:.3f}" + r"""/kWh \\
    & Peak Hour Cost & \$""" + f"{metrics['peak_energy_cost']:.2f}" + r""" \\
\midrule
\multirow{2}{*}{HVAC}
    & Average COP & """ + f"{metrics['hvac_avg_cop']:.2f}" + r""" \\
    & Runtime Hours & """ + f"{metrics['hvac_runtime_hours']:.2f}" + r""" h \\
\bottomrule
\end{tabular}
\end{table}
"""

    with open(save_path, 'w') as f:
        f.write(latex_content)

    print(f"LaTeX table saved to {save_path}")
    return latex_content


def run_simulation_and_plot(env, total_steps=96):
    """
    Run simulation for one day and create the paper figure with analytical results.

    Parameters:
    -----------
    env : BESTOptEnvironment
        The environment object
    total_steps : int
        Number of simulation steps (default 96 for one day with 15-min intervals)
    """

    # Initialize data storage
    data_dict = {
        'zone_temperature': [],
        'cooling_setpoint': [],
        'heating_setpoint': [],
        'hvac_thermal_load': [],
        'hvac_power': [],
        'battery_soc': [],
        'ev_tesla_soc': [],
        'ev_nissan_soc': [],
        'lighting': [],
        'cooking': [],
        'pc': [],
        'tv': [],
        'hvac_power_kw': [],
        'total_building_load': [],
        'battery_power': [],
        'grid_import': [],
        'net_grid': [],
        'pv_generation': [],
        'curtailment': [],
        'ev_charging': [],
        'is_peak': [],
        'electricity_price': [],
    }

    timesteps = []

    # Run simulation
    for timestep in range(min(total_steps, env.total_step)):
        # Step the environment
        observations, done, info = env.step()

        # Get building and system IDs
        cluster_id = 'residential_cluster_1'
        building_id = 'SFH_1'
        hvac_system_id = env.building_system_map[building_id].get('thermal')
        der_system_id = env.building_system_map[building_id].get('electrical')

        # Extract thermal data
        zone_temperature = env.cluster_states[cluster_id].thermal.systems['SFH_1_building'].components[
            'zone0'].temperature
        cooling_setpoint = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].cooling_setpoint_c
        heating_setpoint = env.cluster_actions[cluster_id].thermal.system_actions['hvac_system_1'].heating_setpoint_c

        hvac_system = env.system_modules[hvac_system_id]
        der_system = env.system_modules[der_system_id]

        HVAC_power = hvac_system.FCU_power_total_W
        HVAC_thermal_load = hvac_system.Q_zone_actual_W

        # Extract SOC data
        bat_soc = der_system.battery_states['bat_1'].soc
        ev_tesla_soc = der_system.ev_states['ev_tesla'].soc
        ev_nissan_soc = der_system.ev_states['ev_nissan'].soc

        # Get loads in kW
        HVAC_power_kw = HVAC_power / 1000
        cooking = env.electrical_zone_modules['SFH_1.zone0'].cooking_power / 1000
        pc = env.electrical_zone_modules['SFH_1.zone0'].pc_power / 1000
        tv = env.electrical_zone_modules['SFH_1.zone0'].tv_power / 1000
        lighting = env.electrical_zone_modules['SFH_1.zone0'].lighting_power / 1000

        building_total_load = HVAC_power_kw + env.cluster_states['residential_cluster_1'].electrical.systems[
            'SFH_1_building'].components['electrical'].building_power_w / 1000

        # Get peak signal and price
        is_peak = env.disturbance.prices.peaksignal
        electricity_price = env.disturbance.prices.electricity_price

        # Get DER system action (simplified structure)
        der_action = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1']

        # Extract from simplified DERSystemAction
        battery_power = der_action.battery_power.get('bat_1', 0.0)
        grid_import = der_action.grid_import
        curtailment = der_action.curtailment

        # Extract from PowerBalance
        power_balance = der_action.power_balance
        pv_generation = power_balance.pv_generation
        net_grid = power_balance.net_grid
        ev_charging = power_balance.ev_charging

        # Store data
        timesteps.append(timestep)
        data_dict['zone_temperature'].append(zone_temperature)
        data_dict['cooling_setpoint'].append(cooling_setpoint)
        data_dict['heating_setpoint'].append(heating_setpoint)
        data_dict['hvac_thermal_load'].append(HVAC_thermal_load)
        data_dict['hvac_power'].append(HVAC_power)
        data_dict['battery_soc'].append(bat_soc)
        data_dict['ev_tesla_soc'].append(ev_tesla_soc)
        data_dict['ev_nissan_soc'].append(ev_nissan_soc)
        data_dict['lighting'].append(lighting)
        data_dict['cooking'].append(cooking)
        data_dict['pc'].append(pc)
        data_dict['tv'].append(tv)
        data_dict['hvac_power_kw'].append(HVAC_power_kw)
        data_dict['total_building_load'].append(building_total_load)
        data_dict['is_peak'].append(is_peak)
        data_dict['electricity_price'].append(electricity_price)
        data_dict['battery_power'].append(battery_power)
        data_dict['grid_import'].append(grid_import)
        data_dict['pv_generation'].append(pv_generation)
        data_dict['net_grid'].append(net_grid)
        data_dict['curtailment'].append(curtailment)
        data_dict['ev_charging'].append(ev_charging)

        if timestep % 10 == 0:
            print(f"Step {timestep}/{total_steps}: Temp={zone_temperature:.1f}°C, "
                  f"Net Grid={net_grid:.2f}kW, PV={pv_generation:.2f}kW")

        if done:
            break

    # Convert lists to numpy arrays
    for key in data_dict:
        data_dict[key] = np.array(data_dict[key])

    timesteps = np.array(timesteps)

    # ===== NEW: Calculate analytical metrics =====
    print("\nCalculating analytical metrics...")
    metrics = calculate_analytical_metrics(timesteps, data_dict)

    # Print metrics summary
    print_metrics_summary(metrics)

    # Create the original figure
    fig = create_paper_figure(timesteps, data_dict, save_path='one_day_simulation.pdf')

    # ===== NEW: Create annotated figure with metrics =====
    fig_annotated = create_paper_figure_with_annotations(
        timesteps, data_dict, metrics,
        save_path='one_day_simulation_annotated.pdf'
    )

    # ===== NEW: Create metrics table figure =====
    create_metrics_table_figure(metrics, save_path='metrics_table.pdf')

    # ===== NEW: Export LaTeX table =====
    export_metrics_to_latex(metrics, save_path='metrics_table.tex')

    print(f"\nSimulation completed: {len(timesteps)} steps")
    print("\nGenerated files:")
    print("  - one_day_simulation.pdf (original figure)")
    print("  - one_day_simulation_annotated.pdf (figure with metric annotations)")
    print("  - metrics_table.pdf (standalone metrics table)")
    print("  - metrics_table.tex (LaTeX table for paper)")

    return fig, data_dict, metrics


# Example usage
if __name__ == "__main__":
    from bestopt.env.core.config_manager import ConfigurationManager
    from bestopt.env.core.environment import BESTOptEnvironment
    import os

    PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

    config_path = os.path.join(PROJECT_ROOT_PATH, "examples", "SFH_1_Building", "config_setup.json")
    cm = ConfigurationManager(config_path)
    env = BESTOptEnvironment(cm.config)
    fig, data_dict, metrics = run_simulation_and_plot(env, total_steps=96)