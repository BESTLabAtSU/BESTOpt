import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tes import (
    GlobalConfig,
    Config,
    Optimal_control,
)

# Load example data
data = pd.read_csv('example_data/Example_MG.csv')
chiller_data = pd.read_csv('/old/df_cooling_with_weather.csv')
air_side_load = chiller_data["phvac_abs_sum"].values/1000
other_load = chiller_data["other_elec_power"].values/1000

# Configuration
# Global variables
global_config = GlobalConfig(
    T_amb=chiller_data['temp_amb'].values,    # Temperature array
    TOU=chiller_data['TOU($/kWh)'].values,    # Time-of-use pricing array
    Res=15,                           # 15-minute resolution
    cpus=16                           # Use 16 CPU cores
)

# Create configuration object
config = Config(global_config)

# Define battery parameters
tes_params = {
    'capacity_kwh': 80.0,         # Battery Capacity: 10 kWh
    'c_rate': 0.25,               # 0.25C charge/discharge rate (4 hours)
    'charge_efficiency': 0.95,    # 95% charging efficiency
    'discharge_efficiency': 0.95, # 95% discharging efficiency
    'min_soc': 0.0,               # 10% minimum SOC
    'max_soc': 1.0,               # 90% maximum SOC
    'initial_soc': 0.5            # Start at 50% SOC
}


# TES opt
building = config.create_building(
    thermalload=air_side_load.round(2),
    pcm_params=tes_params,
)

Optimal_controller = Optimal_control(building, global_config)
print("Running optimal control...")
opt_results = Optimal_controller.run()
print(f"Optimal control completed!")
base_chiller = opt_results["baseLoad"]
fig, ax = plt.subplots(1, 1, figsize=(5, 2), dpi=300)
ax.plot(np.arange(len(opt_results)), opt_results["Opt_ele"], '-', linewidth=1, color='red', label="TES")
ax.plot(np.arange(len(opt_results)), opt_results["baseLoad"], '-', linewidth=1, color='black', label="no TES")
ax_ = ax.twinx()
ax_.plot(np.arange(len(opt_results)), opt_results["TOU"], '--', linewidth=1, color='gray', label="Price")
ax.grid(False)
ax_.grid(False)
ax.set_ylabel('Chiller_Ele(kW)', fontsize=7)
ax_.set_ylabel('Price($/kWh)', fontsize=7)
ax.tick_params(axis='both', which='both', labelsize=7)
ax_.tick_params(axis='both', which='both', labelsize=7)
ax.set_xlabel('Time Step', fontsize=7)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
               ncol=3, fontsize=7, frameon=False)
plt.tight_layout()
plt.show()
################################### TES done

from micro_grid import (
    GlobalConfig,
    Config,
    Optimal_control,
)

# Configuration
# Global variables
global_config = GlobalConfig(
    T_amb=chiller_data['temp_amb'].values,    # Temperature array
    Sol=chiller_data['q_sol'].values,         # Solar irradiance array
    TOU=chiller_data['TOU($/kWh)'].values,           # Time-of-use pricing array
    Res=15,                           # 15-minute resolution
    cpus=16                           # Use 16 CPU cores
)

# Create configuration object
config = Config(global_config)

# Add building and MG components
# Example: Bldg + PV + Battery
# Define PV parameters
pv_params = {
    'capacity_kw': 20  # PV Capacity: 5 kW
}

# Define battery parameters
battery_params = {
    'capacity_kwh': 100.0,         # Battery Capacity: 10 kWh
    'c_rate': 0.25,               # 0.25C charge/discharge rate (4 hours)
    'charge_efficiency': 0.95,    # 95% charging efficiency
    'discharge_efficiency': 0.95, # 95% discharging efficiency
    'min_soc': 0.1,               # 10% minimum SOC
    'max_soc': 0.9,               # 90% maximum SOC
    'initial_soc': 0.1            # Start at 50% SOC
}

# Create building configuration
building = config.create_building(
    load=opt_results["Opt_ele"].values+other_load.round(2),  # Assign Building load (Need to be kW)
    pv_params=pv_params,            # Assign PV parameters
    battery_params=battery_params,  # Assign Battery parameters

)
#%%
# Use optimal control
Optimal_controller = Optimal_control(building, global_config)
print("Running optimal control...")
opt_results = Optimal_controller.run()
print(f"Optimal control completed!")


fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=300)
ax.plot(np.arange(len(opt_results)), opt_results["GridPurchase"], '-', linewidth=1, color='red', label="TES+PV+Battery")
ax.plot(np.arange(len(opt_results)), (other_load.round(2)+base_chiller), '-', linewidth=1, color='black', label="no DERs")
ax_ = ax.twinx()
ax_.plot(np.arange(len(opt_results)), opt_results["TOU"], '--', linewidth=1, color='gray', label="Price")
ax.grid(False)
ax_.grid(False)
ax.set_ylabel('Total_Ele(kW)', fontsize=7)
ax_.set_ylabel('Price($/kWh)', fontsize=7)
ax.tick_params(axis='both', which='both', labelsize=7)
ax_.tick_params(axis='both', which='both', labelsize=7)
ax.set_xlabel('Time Step', fontsize=7)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
               ncol=3, fontsize=7, frameon=False)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=300)
ax.plot(np.arange(len(opt_results)), opt_results["SoC_Bat"], '-', linewidth=1, color='blue', label="Battery SOC")
ax_ = ax.twinx()
ax_.plot(np.arange(len(opt_results)), opt_results["TOU"], '--', linewidth=1, color='gray', label="Price")
ax.grid(False)
ax_.grid(False)
ax.set_ylabel('Battery SOC', fontsize=7)
ax_.set_ylabel('Price($/kWh)', fontsize=7)
ax.tick_params(axis='both', which='both', labelsize=7)
ax_.tick_params(axis='both', which='both', labelsize=7)
ax.set_xlabel('Time Step', fontsize=7)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
               ncol=3, fontsize=7, frameon=False)
plt.tight_layout()
plt.show()

baseline_total_load = other_load.round(2) + base_chiller
optimized_grid_purchase = opt_results["GridPurchase"]
tou_prices = opt_results["TOU"]
resolution_hours = global_config.Res / 60

def calculate_energy_cost(power_profile_kW, tou_prices_per_kWh, resolution_hours):
    """
    Calculates the total energy cost for a given power profile.
    Args:
        power_profile_kW (np.array): Array of power consumption in kW.
        tou_prices_per_kWh (np.array): Array of Time-of-Use prices in $/kWh.
        resolution_hours (float): Time resolution in hours (e.g., 0.25 for 15 minutes).
    Returns:
        float: Total energy cost.
    """
    return np.sum(power_profile_kW * tou_prices_per_kWh * resolution_hours)
# Calculate costs
baseline_cost = calculate_energy_cost(baseline_total_load, tou_prices, resolution_hours)
optimized_cost = calculate_energy_cost(optimized_grid_purchase, tou_prices, resolution_hours)
# Calculate savings
money_saved = baseline_cost - optimized_cost
percentage_money_saved = (money_saved / baseline_cost) * 100 if baseline_cost else 0
print("\n--- Money Saving Analysis ---")
print(f"Baseline Energy Cost (no DERs): ${baseline_cost:.2f}")
print(f"Optimized Energy Cost (TES+PV+Battery): ${optimized_cost:.2f}")
print(f"Money Saved: ${money_saved:.2f}")
print(f"Percentage Money Saved: {percentage_money_saved:.2f}%")
#-----------------------------------------------------------------------------------------------------------------------
### :zap: Flexibility Benefits
def calculate_peak_reduction(baseline_load_kW, optimized_load_kW):
    """
    Calculates the peak demand reduction.
    Args:
        baseline_load_kW (np.array): Array of baseline power consumption in kW.
        optimized_load_kW (np.array): Array of optimized power consumption in kW (grid purchase).
    Returns:
        tuple: (baseline_peak, optimized_peak, peak_reduction_kW, percentage_peak_reduction).
    """
    baseline_peak = np.max(baseline_load_kW)
    optimized_peak = np.max(optimized_load_kW)
    peak_reduction_kW = baseline_peak - optimized_peak
    percentage_peak_reduction = (peak_reduction_kW / baseline_peak) * 100 if baseline_peak else 0
    return baseline_peak, optimized_peak, peak_reduction_kW, percentage_peak_reduction
def calculate_load_shifting_metric(baseline_load_kW, optimized_load_kW, tou_prices_per_kWh):
    """
    Estimates load shifting by comparing energy consumption during peak vs off-peak hours.
    A more sophisticated metric would involve cross-correlation or dynamic time warping,
    but this provides a basic indication.
    Args:
        baseline_load_kW (np.array): Array of baseline power consumption in kW.
        optimized_load_kW (np.array): Array of optimized power consumption in kW (grid purchase).
        tou_prices_per_kWh (np.array): Array of Time-of-Use prices in $/kWh.
    Returns:
        dict: A dictionary containing metrics related to load shifting.
    """
    # Identify peak and off-peak hours based on TOU prices
    # For simplicity, let's define peak as prices > average price
    avg_price = np.mean(tou_prices_per_kWh)
    peak_hours_indices = np.where(tou_prices_per_kWh > avg_price)[0]
    off_peak_hours_indices = np.where(tou_prices_per_kWh <= avg_price)[0]
    baseline_peak_hour_consumption = np.sum(baseline_load_kW[peak_hours_indices]) * resolution_hours
    optimized_peak_hour_consumption = np.sum(optimized_load_kW[peak_hours_indices]) * resolution_hours
    baseline_off_peak_hour_consumption = np.sum(baseline_load_kW[off_peak_hours_indices]) * resolution_hours
    optimized_off_peak_hour_consumption = np.sum(optimized_load_kW[off_peak_hours_indices]) * resolution_hours
    load_shifted_from_peak = baseline_peak_hour_consumption - optimized_peak_hour_consumption
    load_shifted_to_off_peak = optimized_off_peak_hour_consumption - baseline_off_peak_hour_consumption
    return {
        "baseline_peak_hour_consumption_kWh": baseline_peak_hour_consumption,
        "optimized_peak_hour_consumption_kWh": optimized_peak_hour_consumption,
        "load_reduction_during_peak_kWh": load_shifted_from_peak,
        "baseline_off_peak_hour_consumption_kWh": baseline_off_peak_hour_consumption,
        "optimized_off_peak_hour_consumption_kWh": optimized_off_peak_hour_consumption,
        "load_increase_during_off_peak_kWh": load_shifted_to_off_peak
    }
# Calculate peak reduction
baseline_peak, optimized_peak, peak_reduction_kW, percentage_peak_reduction = \
    calculate_peak_reduction(baseline_total_load, optimized_grid_purchase)
print("\n--- Peak Reduction Analysis ---")
print(f"Baseline Peak Demand: {baseline_peak:.2f} kW")
print(f"Optimized Peak Demand: {optimized_peak:.2f} kW")
print(f"Peak Demand Reduction: {peak_reduction_kW:.2f} kW")
print(f"Percentage Peak Demand Reduction: {percentage_peak_reduction:.2f}%")
# Calculate load shifting metrics
load_shifting_metrics = calculate_load_shifting_metric(baseline_total_load, optimized_grid_purchase, tou_prices)
print("\n--- Load Shifting Analysis ---")
print(f"Baseline Peak Hour Consumption: {load_shifting_metrics['baseline_peak_hour_consumption_kWh']:.2f} kWh")
print(f"Optimized Peak Hour Consumption: {load_shifting_metrics['optimized_peak_hour_consumption_kWh']:.2f} kWh")
print(f"Load Reduction During Peak Hours: {load_shifting_metrics['load_reduction_during_peak_kWh']:.2f} kWh")
print(f"Baseline Off-Peak Hour Consumption: {load_shifting_metrics['baseline_off_peak_hour_consumption_kWh']:.2f} kWh")
print(f"Optimized Off-Peak Hour Consumption: {load_shifting_metrics['optimized_off_peak_hour_consumption_kWh']:.2f} kWh")
print(f"Load Increase During Off-Peak Hours (Shifted): {load_shifting_metrics['load_increase_during_off_peak_kWh']:.2f} kWh")
#-----------------------------------------------------------------------------------------------------------------------
