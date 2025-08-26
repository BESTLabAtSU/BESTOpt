# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from micro_grid import (
    GlobalConfig,
    Config,
    Optimal_control,
    Rule_control,  # Updated class name
)
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

data = pd.read_csv('example_data/Example_MG.csv')

# Configuration
# Global variables
global_config = GlobalConfig(
    T_amb=data['temp_amb'].values,  # Temperature array
    Sol=data['solar'].values,  # Solar irradiance array
    TOU=data['TOU'].values,  # Time-of-use pricing array
    Res=15,  # 15-minute resolution
    cpus=16  # Use 16 CPU cores
)

# Create configuration object
config = Config(global_config)


def generate_random_building_config(building_id, base_load):
    """Generate random configuration for a building"""

    # Define component probability (adjust as needed)
    pv_prob = 0.7  # 70% chance of having PV
    battery_prob = 0.6  # 60% chance of having battery
    ev_prob = 0.5  # 50% chance of having EV

    # Randomly decide which components to include
    has_pv = random.random() < pv_prob
    has_battery = random.random() < battery_prob
    has_ev = random.random() < ev_prob

    config_dict = {
        'building_id': building_id,
        'has_pv': has_pv,
        'has_battery': has_battery,
        'has_ev': has_ev,
        'load': base_load * np.random.uniform(1, 1),  # Scale load by 70%-130%
    }

    # PV parameters (if included)
    if has_pv:
        pv_params = {
            'capacity_kw': np.random.uniform(4, 8)  # 2-8 kW PV capacity
        }
        config_dict['pv_params'] = pv_params
        config_dict['pv_capacity'] = pv_params['capacity_kw']
    else:
        config_dict['pv_params'] = None
        config_dict['pv_capacity'] = 0

    # Battery parameters (if included)
    if has_battery:
        battery_capacity = np.random.uniform(5, 10)  # 3-10 kWh
        battery_params = {
            'capacity_kwh': battery_capacity,
            'c_rate': np.random.uniform(0.2, 0.5),  # 0.2-0.5C rate
            'charge_efficiency': np.random.uniform(0.90, 0.95),
            'discharge_efficiency': np.random.uniform(0.90, 0.95),
            'min_soc': np.random.uniform(0.05, 0.15),  # 5-15% min SOC
            'max_soc': np.random.uniform(0.85, 0.95),  # 85-95% max SOC
            'initial_soc': np.random.uniform(0.4, 0.6)  # 40-60% initial SOC
        }
        config_dict['battery_params'] = battery_params
        config_dict['battery_capacity'] = battery_capacity
    else:
        config_dict['battery_params'] = None
        config_dict['battery_capacity'] = 0

    # EV parameters (if included)
    if has_ev:
        # Random departure time (6-10 AM) and arrival time (4-8 PM)
        departure_hour = np.random.uniform(6, 10)
        arrival_hour = np.random.uniform(16, 20)

        ev_capacity = np.random.uniform(40, 80)  # 30-80 kWh EV battery
        ev_params = {
            'capacity_kwh': ev_capacity,
            'c_rate': np.random.uniform(0.15, 0.3),  # 0.15-0.3C rate
            'charge_efficiency': np.random.uniform(0.90, 0.95),
            'discharge_efficiency': np.random.uniform(0.90, 0.95),
            'min_soc': np.random.uniform(0.05, 0.15),
            'max_soc': 1.0,
            'departure_time': int(departure_hour * 4),  # Convert to 15-min steps
            'arrival_time': int(arrival_hour * 4),
            'arrival_soc': np.random.uniform(0.3, 0.7),  # 30-70% arrival SOC
            'required_departure_soc': np.random.uniform(0.8, 0.95)  # 80-95% departure SOC
        }
        config_dict['ev_params'] = ev_params
        config_dict['ev_capacity'] = ev_capacity
        config_dict['departure_time'] = departure_hour
        config_dict['arrival_time'] = arrival_hour
    else:
        config_dict['ev_params'] = None
        config_dict['ev_capacity'] = 0
        config_dict['departure_time'] = None
        config_dict['arrival_time'] = None

    return config_dict


def print_building_configurations(building_configs):
    """Print summary of all building configurations"""
    print("=" * 80)
    print("BUILDING CONFIGURATIONS SUMMARY")
    print("=" * 80)

    total_pv = 0
    total_battery = 0
    total_ev = 0
    pv_count = 0
    battery_count = 0
    ev_count = 0

    for i, config_dict in enumerate(building_configs):
        print(f"\nBuilding {i + 1}:")
        print(f"  Load Scale Factor: {config_dict['load'][0] / data['load'].values[0] * 1000:.2f}")

        if config_dict['has_pv']:
            print(f"  PV: {config_dict['pv_capacity']:.1f} kW")
            total_pv += config_dict['pv_capacity']
            pv_count += 1
        else:
            print("  PV: None")

        if config_dict['has_battery']:
            print(
                f"  Battery: {config_dict['battery_capacity']:.1f} kWh (C-rate: {config_dict['battery_params']['c_rate']:.2f})")
            total_battery += config_dict['battery_capacity']
            battery_count += 1
        else:
            print("  Battery: None")

        if config_dict['has_ev']:
            print(
                f"  EV: {config_dict['ev_capacity']:.1f} kWh (Depart: {config_dict['departure_time']:.1f}h, Arrive: {config_dict['arrival_time']:.1f}h)")
            total_ev += config_dict['ev_capacity']
            ev_count += 1
        else:
            print("  EV: None")

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total Buildings: 20")
    print(f"Buildings with PV: {pv_count}/20 ({pv_count / 20 * 100:.1f}%) - Total: {total_pv:.1f} kW")
    print(
        f"Buildings with Battery: {battery_count}/20 ({battery_count / 20 * 100:.1f}%) - Total: {total_battery:.1f} kWh")
    print(f"Buildings with EV: {ev_count}/20 ({ev_count / 20 * 100:.1f}%) - Total: {total_ev:.1f} kWh")
    print("=" * 80)


# Generate 20 random building configurations
print("Generating 20 random building configurations...")
building_configs = []  # Convert to kW
datapath = "./MG_dataset"
for i in range(20):
    df = pd.read_csv(f"{datapath}/bldg{i}.csv")
    base_load = df["total_building_load"].values / 1000
    config_dict = generate_random_building_config(i + 1, base_load)
    building_configs.append(config_dict)

# Print all configurations
print_building_configurations(building_configs)

# Create building objects and run simulations
print("\nCreating buildings and running simulations...")
optimal_results = []
rule_results = []
building_objects = []

for i, config_dict in enumerate(building_configs):
    print(f"Processing Building {i + 1}/20...")

    # Create building configuration
    building = config.create_building(
        load=config_dict['load'],
        pv_params=config_dict['pv_params'],
        battery_params=config_dict['battery_params'],
        ev_params=config_dict['ev_params']
    )
    building_objects.append(building)

    # Run optimal control
    opt_controller = Optimal_control(building, global_config)
    opt_result = opt_controller.run()
    opt_result['building_id'] = i + 1
    optimal_results.append(opt_result)

    # Run rule-based control
    rule_controller = Rule_control(building, global_config)
    rule_result = rule_controller.run()
    rule_result['building_id'] = i + 1
    rule_results.append(rule_result)

print("All simulations completed!")

# Calculate overall results
print("\nCalculating overall results...")

# Combine all results
total_opt_cost = sum([result['GridPurchase'].sum() * result['TOU'].sum() / len(result) for result in optimal_results])
total_rule_cost = sum([result['GridPurchase'].sum() * result['TOU'].sum() / len(result) for result in rule_results])

total_opt_energy = sum([result['GridPurchase'].sum() for result in optimal_results])
total_rule_energy = sum([result['GridPurchase'].sum() for result in rule_results])

print(f"\nOVERALL RESULTS:")
print(f"=" * 50)
print(f"Optimal Control:")
print(f"  Total Grid Purchase: {total_opt_energy:.2f} kWh")
print(f"  Total Cost: ${total_opt_cost:.2f}")
print(f"\nRule-Based Control:")
print(f"  Total Grid Purchase: {total_rule_energy:.2f} kWh")
print(f"  Total Cost: ${total_rule_cost:.2f}")
print(f"\nSavings with Optimal Control:")
print(
    f"  Energy Savings: {total_rule_energy - total_opt_energy:.2f} kWh ({(total_rule_energy - total_opt_energy) / total_rule_energy * 100:.1f}%)")
print(
    f"  Cost Savings: ${total_rule_cost - total_opt_cost:.2f} ({(total_rule_cost - total_opt_cost) / total_rule_cost * 100:.1f}%)")

# Store results for further analysis
results_summary = {
    'building_configs': building_configs,
    'optimal_results': optimal_results,
    'rule_results': rule_results,
    'building_objects': building_objects
}

# Calculate baseline (no MG components) for comparison
print("\nCalculating baseline (no MG components)...")
baseline_results = []

for i, config_dict in enumerate(building_configs):
    # Create building with only load (no PV, battery, or EV)
    baseline_building = config.create_building(
        load=config_dict['load'],
        pv_params=None,
        battery_params=None,
        ev_params=None
    )

    # For baseline, grid purchase = building load (no other sources)
    baseline_result = pd.DataFrame({
        'GridPurchase': config_dict['load'],
        'TOU': data['TOU'].values,
        'building_id': i + 1
    })
    baseline_results.append(baseline_result)

# Plotting section
print("\nGenerating plots...")

# Aggregate grid purchase for all scenarios
time_steps = len(data)
total_baseline_grid = np.zeros(time_steps)
total_rule_grid = np.zeros(time_steps)
total_optimal_grid = np.zeros(time_steps)

# Sum grid purchases across all buildings for each time step
for i in range(20):
    total_baseline_grid += baseline_results[i]['GridPurchase'].values
    total_rule_grid += rule_results[i]['GridPurchase'].values
    total_optimal_grid += optimal_results[i]['GridPurchase'].values

# Create time index (assuming data represents one day with 15-min resolution)
time_index = pd.date_range(start='2024-01-01', periods=time_steps, freq='15min')

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Grid Load Comparison
ax1.plot(time_index, total_baseline_grid, label='Baseline (No MG)', linewidth=2, color='red', alpha=0.8)
ax1.plot(time_index, total_rule_grid, label='Rule-Based Control', linewidth=2, color='orange', alpha=0.8)
ax1.plot(time_index, total_optimal_grid, label='Optimal Control', linewidth=2, color='green', alpha=0.8)

ax1.set_ylabel('Total Grid Load (kW)', fontsize=12)
ax1.set_title('Grid Load Comparison: 20 Buildings with Different Control Strategies', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Format x-axis to show hours
ax1.set_xlabel('Time of Day', fontsize=12)
hours = [i for i in range(0, 24, 4)]
hour_indices = [i * 4 for i in hours]  # Convert to 15-min indices
ax1.set_xticks([time_index[i] for i in hour_indices])
ax1.set_xticklabels([f"{h:02d}:00" for h in hours])

# Plot 2: TOU Price overlay
ax2_twin = ax2.twinx()
ax2.plot(time_index, total_baseline_grid - total_rule_grid, label='Baseline vs Rule-Based Savings',
         linewidth=2, color='blue', alpha=0.7)
ax2.plot(time_index, total_baseline_grid - total_optimal_grid, label='Baseline vs Optimal Savings',
         linewidth=2, color='darkgreen', alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Plot TOU price on secondary y-axis
ax2_twin.plot(time_index, data['TOU'].values, label='TOU Price',
              linewidth=1.5, color='purple', alpha=0.6, linestyle=':')

ax2.set_ylabel('Grid Load Savings (kW)', fontsize=12)
ax2_twin.set_ylabel('TOU Price ($/kWh)', fontsize=12, color='purple')
ax2.set_xlabel('Time of Day', fontsize=12)
ax2.set_title('Grid Load Savings and TOU Price Profile', fontsize=14, fontweight='bold')

# Format x-axis
ax2.set_xticks([time_index[i] for i in hour_indices])
ax2.set_xticklabels([f"{h:02d}:00" for h in hours])

# Legends
ax2.legend(loc='upper left', fontsize=10)
ax2_twin.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2_twin.tick_params(axis='y', labelcolor='purple')

plt.tight_layout()
plt.show()

# Create the plot
fig, (ax1) = plt.subplots(1, 1, figsize=(5, 3), dpi=500)

# Plot 1: Grid Load Comparison
ax1.plot(time_index, total_baseline_grid, label='Baseline (No MG)', linewidth=1, color='blue', alpha=0.8)
ax1.plot(time_index, total_rule_grid, label='Rule-Based Control', linewidth=1, color='green', alpha=0.8)
ax1.plot(time_index, total_optimal_grid, label='Optimal Control', linewidth=1, color='red', alpha=0.8)

ax1.set_ylabel('Grid Purchase (kW)', fontsize=8)
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

# Format x-axis to show hours
ax1.set_xlabel('Time of Day', fontsize=8)
hours = [i for i in range(0, 24, 4)]
hour_indices = [i * 4 for i in hours]  # Convert to 15-min indices
ax1.set_xticks([time_index[i] for i in hour_indices])
ax1.set_xticklabels([f"{h:02d}:00" for h in hours])

plt.tight_layout()
plt.show()


# Print detailed comparison
print(f"\nDETAILED COMPARISON RESULTS:")
print(f"=" * 60)

baseline_total = sum([result['GridPurchase'].sum() for result in baseline_results])
rule_total = sum([result['GridPurchase'].sum() for result in rule_results])
optimal_total = sum([result['GridPurchase'].sum() for result in optimal_results])

print(f"Total Grid Energy Consumption:")
print(f"  Baseline (No MG):     {baseline_total:.2f} kWh")
print(f"  Rule-Based Control:   {rule_total:.2f} kWh")
print(f"  Optimal Control:      {optimal_total:.2f} kWh")

print(f"\nEnergy Savings vs Baseline:")
print(
    f"  Rule-Based Savings:   {baseline_total - rule_total:.2f} kWh ({(baseline_total - rule_total) / baseline_total * 100:.1f}%)")
print(
    f"  Optimal Savings:      {baseline_total - optimal_total:.2f} kWh ({(baseline_total - optimal_total) / baseline_total * 100:.1f}%)")

print(f"\nRule-Based vs Optimal:")
print(
    f"  Additional Optimal Savings: {rule_total - optimal_total:.2f} kWh ({(rule_total - optimal_total) / rule_total * 100:.1f}%)")

# Calculate peak reduction
baseline_peak = np.max(total_baseline_grid)
rule_peak = np.max(total_rule_grid)
optimal_peak = np.max(total_optimal_grid)

print(f"\nPeak Load Reduction:")
print(f"  Baseline Peak:        {baseline_peak:.2f} kW")
print(
    f"  Rule-Based Peak:      {rule_peak:.2f} kW ({(baseline_peak - rule_peak) / baseline_peak * 100:.1f}% reduction)")
print(
    f"  Optimal Peak:         {optimal_peak:.2f} kW ({(baseline_peak - optimal_peak) / baseline_peak * 100:.1f}% reduction)")

# Calculate cost comparison
baseline_cost = sum([np.sum(result['GridPurchase'].values * result['TOU'].values) for result in baseline_results])
rule_cost = sum([np.sum(result['GridPurchase'].values * result['TOU'].values) for result in rule_results])
optimal_cost = sum([np.sum(result['GridPurchase'].values * result['TOU'].values) for result in optimal_results])

print(f"\nCost Comparison:")
print(f"  Baseline Cost:        ${baseline_cost:.2f}")
print(
    f"  Rule-Based Cost:      ${rule_cost:.2f} (${baseline_cost - rule_cost:.2f} saved, {(baseline_cost - rule_cost) / baseline_cost * 100:.1f}%)")
print(
    f"  Optimal Cost:         ${optimal_cost:.2f} (${baseline_cost - optimal_cost:.2f} saved, {(baseline_cost - optimal_cost) / baseline_cost * 100:.1f}%)")
print(f"=" * 60)










# # Import required packages
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from micro_grid import (
#     GlobalConfig,
#     Config,
#     Optimal_control,
#     Rule_control,
# )
#
# data = pd.read_csv('./example_data/Example_MG.csv')
#
# # Configuration
# # Global variables
# global_config = GlobalConfig(
#     T_amb=data['temp_amb'].values,    # Temperature array
#     Sol=data['solar'].values,         # Solar irradiance array
#     TOU=data['TOU'].values,           # Time-of-use pricing array
#     Res=15,                           # 15-minute resolution
#     cpus=16                           # Use 16 CPU cores
# )
#
# # Create configuration object
# config = Config(global_config)
#
# # Add building and MG components
# # Example: Bldg + PV + Battery
# # Define PV parameters
# pv_params = {
#     'capacity_kw': 4  # PV Capacity: 4 kW
# }
#
# # Define battery parameters
# battery_params = {
#     'capacity_kwh': 5.0,          # Battery Capacity: 5 kWh
#     'c_rate': 0.25,               # 0.25C charge/discharge rate (4 hours)
#     'charge_efficiency': 0.95,    # 95% charging efficiency
#     'discharge_efficiency': 0.95, # 95% discharging efficiency
#     'min_soc': 0.1,               # 10% minimum SOC
#     'max_soc': 0.9,               # 90% maximum SOC
#     'initial_soc': 0.5            # Start at 50% SOC
# }
#
# # Define ev parameters
# ev_params = {
#     'capacity_kwh': 50.0,         # Battery Capacity: 5 kWh
#     'c_rate': 0.2,                # 0.25C charge/discharge rate (4 hours)
#     'charge_efficiency': 0.95,    # 95% charging efficiency
#     'discharge_efficiency': 0.95, # 95% discharging efficiency
#     'min_soc': 0.1,               # 10% minimum SOC
#     'max_soc': 1.0,               # 90% maximum SOC
#     'departure_time': 8*4,
#     'arrival_time': 18*4,
#     'arrival_soc': 0.5,
#     'required_departure_soc': 0.95
# }
#
# # Create building configuration
# building = config.create_building(
#     load=data['load'].values/1000,      # Assign Building load (Need to be kW)
#     pv_params=pv_params,           # Assign PV parameters
#     battery_params=battery_params,  # Assign Battery parameters
#     ev_params=ev_params
# )
#
# # Use optimal control
# Optimal_controller = Optimal_control(building, global_config)
# print("Running optimal control...")
# opt_results = Optimal_controller.run()
# print(f"Optimal control completed!")
#
# # Use rule-based control
# Rule_controller = Rule_control(building, global_config)
# print("Running rule based control...")
# rb_results = Rule_controller.run()
# print(f"Rule based control completed!")
#
# fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)
# ax.plot(np.arange(len(rb_results)), rb_results["GridPurchase"], '-', linewidth=1, color='red', label="rule-based PV-Bat")
# ax.plot(np.arange(len(opt_results)), opt_results["GridPurchase"], '-', linewidth=1, color='blue', label="optimal PV-Bat")
# ax.plot(np.arange(len(opt_results)), opt_results["Load"], '-', linewidth=1, color='black', label="no PV-Bat")
# ax_ = ax.twinx()
# ax_.plot(np.arange(len(opt_results)), opt_results["TOU"], '--', linewidth=1, color='gray', label="Price")
# ax.grid(False)
# ax_.grid(False)
# ax.set_ylabel('Load(kW)', fontsize=7)
# ax_.set_ylabel('Price($/kWh)', fontsize=7)
# ax.tick_params(axis='both', which='both', labelsize=7)
# ax_.tick_params(axis='both', which='both', labelsize=7)
# ax.set_xlabel('Time Step', fontsize=7)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
#                ncol=3, fontsize=7, frameon=False)
# plt.show()
#
# fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)
# ax.plot(np.arange(len(rb_results)), rb_results["SoC_Bat"], '-', linewidth=1, color='red', label="rule-based PV-Bat")
# ax.plot(np.arange(len(opt_results)), opt_results["SoC_Bat"], '-', linewidth=1, color='blue', label="optimal PV-Bat")
# ax_ = ax.twinx()
# ax_.plot(np.arange(len(opt_results)), opt_results["TOU"], '--', linewidth=1, color='gray', label="Price")
# ax.grid(False)
# ax_.grid(False)
# ax.set_ylabel('SOC', fontsize=7)
# ax_.set_ylabel('Price($/kWh)', fontsize=7)
# ax.tick_params(axis='both', which='both', labelsize=7)
# ax_.tick_params(axis='both', which='both', labelsize=7)
# ax.set_xlabel('Time Step', fontsize=7)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
#                ncol=2, fontsize=7, frameon=False)
# plt.show()
#
# fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)
# ax.plot(np.arange(len(rb_results)), rb_results["SoC_EV"], '-', linewidth=1, color='red', label="rule-based EV")
# ax.plot(np.arange(len(opt_results)), opt_results["SoC_EV"], '-', linewidth=1, color='blue', label="optimal EV")
# ax_ = ax.twinx()
# ax_.plot(np.arange(len(opt_results)), opt_results["TOU"], '--', linewidth=1, color='gray', label="Price")
# ax.grid(False)
# ax_.grid(False)
# ax.set_ylabel('SOC', fontsize=7)
# ax_.set_ylabel('Price($/kWh)', fontsize=7)
# ax.tick_params(axis='both', which='both', labelsize=7)
# ax_.tick_params(axis='both', which='both', labelsize=7)
# ax.set_xlabel('Time Step', fontsize=7)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
#                ncol=2, fontsize=7, frameon=False)
# plt.show()