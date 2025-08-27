"""
This is a complete example demonstrating the building energy physics-informed ML environment.

This example shows how to:
1. Configure and initialize all system modules
2. Set up the simulation environment
3. Run a complete simulation loop
4. Implement different control strategies
5. Analyze results and calculate KPIs
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any
import logging

# Import the modules (adjust paths as needed for your project structure)
from ..env.core.environment import Environment
from ..env.core.data_structure import (
    SimulationConfig, Action, ThermalAction, ElectricalAction,
    HVACMode, BatteryMode
)
from ..env.modules.building.dynamics import ThermalDynamicsModule
from ..env.modules.ders.battery import BatteryModule
from ..env.modules.ders.pv import PVModule
from ..env.disturbances.weather import WeatherModule






































class BasicController:
    """
    Simple rule-based controller for demonstration.

    This controller implements basic comfort-based HVAC control
    and simple battery management.
    """

    def __init__(self):
        self.comfort_temp_low = 20.0  # °C
        self.comfort_temp_high = 24.0  # °C
        self.battery_charge_threshold = 30.0  # % SOC
        self.battery_discharge_threshold = 80.0  # % SOC

    def get_action(self, observation, info=None) -> Action:
        """
        Generate control action based on current observation.

        Args:
            observation: Current system observation
            info: Additional information from environment

        Returns:
            Control action
        """
        action = Action()

        # HVAC Control
        zone_temp = observation.zone_temp

        if zone_temp < self.comfort_temp_low:
            action.thermal.hvac_mode = HVACMode.HEATING
            action.thermal.heating_setpoint = self.comfort_temp_low + 1.0
            action.thermal.cooling_setpoint = self.comfort_temp_high
        elif zone_temp > self.comfort_temp_high:
            action.thermal.hvac_mode = HVACMode.COOLING
            action.thermal.heating_setpoint = self.comfort_temp_low
            action.thermal.cooling_setpoint = self.comfort_temp_high - 1.0
        else:
            action.thermal.hvac_mode = HVACMode.AUTO
            action.thermal.heating_setpoint = self.comfort_temp_low
            action.thermal.cooling_setpoint = self.comfort_temp_high

        # Battery Control (simple time-of-use strategy)
        battery_soc = observation.battery_soc
        pv_power = observation.pv_power
        grid_power = observation.grid_power

        # Charge battery during high PV production or low electricity prices
        if pv_power > 2.0 and battery_soc < 90:  # Excess PV power
            action.electrical.battery_power_setpoint = min(5.0, pv_power - 1.0)
            action.electrical.battery_mode = BatteryMode.CHARGE

        # Discharge battery during peak hours (simplified: 4-9 PM)
        elif 16 <= observation.time_of_day <= 21 and battery_soc > self.battery_discharge_threshold:
            action.electrical.battery_power_setpoint = -3.0
            action.electrical.battery_mode = BatteryMode.DISCHARGE

        # Charge battery when cheap electricity (simplified: 11 PM - 6 AM)
        elif (23 <= observation.time_of_day or observation.time_of_day <= 6) and battery_soc < 60:
            action.electrical.battery_power_setpoint = 2.0
            action.electrical.battery_mode = BatteryMode.CHARGE

        else:
            action.electrical.battery_power_setpoint = 0.0
            action.electrical.battery_mode = BatteryMode.IDLE

        return action







def create_simulation_config() -> SimulationConfig:
    """Create simulation configuration."""
    config = SimulationConfig()

    # Simulation parameters
    config.timestep = 15.0  # 15 minutes
    config.simulation_days = 7  # One week simulation
    config.start_time = 0.0  # Start at beginning of year

    # Building parameters
    config.building_area = 200.0  # m²
    config.building_volume = 600.0  # m³
    config.building_thermal_mass = 50000.0  # J/K
    config.building_ua_value = 300.0  # W/K

    # HVAC parameters
    config.hvac_capacity = 12.0  # kW
    config.hvac_cop_cooling = 3.0
    config.hvac_cop_heating = 2.5

    # Battery parameters
    config.battery_capacity = 20.0  # kWh
    config.battery_max_power = 5.0  # kW
    config.battery_efficiency = 0.95
    config.battery_soc_min = 10.0  # %
    config.battery_soc_max = 90.0  # %

    # PV parameters
    config.pv_capacity = 8.0  # kW
    config.pv_tilt = 30.0  # degrees
    config.pv_azimuth = 180.0  # degrees (south-facing)

    return config


def create_modules() -> Dict[str, Any]:
    """Create and configure all system modules."""

    # Thermal dynamics module
    thermal_config = {
        'building_area': 200.0,  # m²
        'ceiling_height': 3.0,  # m
        'ua_walls': 150.0,  # W/K
        'ua_windows': 100.0,  # W/K
        'ua_roof': 50.0,  # W/K
        'infiltration_rate': 0.5,  # ACH
        'zone_thermal_mass': 180000.0,  # J/K
        'hvac_capacity': 12.0,  # kW
        'enable_history': True
    }
    thermal_module = ThermalDynamicsModule(thermal_config)

    # Battery module
    battery_config = {
        'capacity_kwh': 20.0,
        'max_power_charge_kw': 5.0,
        'max_power_discharge_kw': 5.0,
        'charge_efficiency': 0.95,
        'discharge_efficiency': 0.95,
        'soc_min': 10.0,
        'soc_max': 90.0,
        'initial_soc': 50.0,
        'chemistry': 'lithium_ion',
        'enable_history': True
    }
    battery_module = BatteryModule(battery_config)

    # PV module
    pv_config = {
        'dc_capacity_kw': 8.0,
        'panel_area_m2': 40.0,
        'num_panels': 20,
        'technology': 'silicon_crystalline',
        'tilt_degrees': 30.0,
        'azimuth_degrees': 180.0,
        'latitude': 37.7749,  # San Francisco
        'longitude': -122.4194,
        'inverter_capacity_kw': 7.6,
        'inverter_efficiency_max': 0.96,
        'enable_history': True
    }
    pv_module = PVModule(pv_config)

    # Weather module
    weather_config = {
        'source': 'synthetic',
        'station': {
            'name': 'San Francisco',
            'latitude': 37.7749,
            'longitude': -122.4194,
            'elevation': 56.0,
            'timezone_offset': -8.0
        },
        'temp_annual_avg': 15.0,  # °C
        'temp_daily_range': 8.0,  # °C
        'temp_annual_range': 10.0,  # °C
        'humidity_avg': 65.0,  # %
        'wind_speed_avg': 4.0,  # m/s
        'clearness_index': 0.65,
        'enable_history': True,
        'random_seed': 42
    }
    weather_module = WeatherModule(weather_config)

    return {
        'thermal': thermal_module,
        'battery': battery_module,
        'pv': pv_module,
        'weather': weather_module
    }


def run_simulation():
    """Run a complete building energy simulation."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting building energy simulation...")

    # Create configuration
    config = create_simulation_config()

    # Create modules
    modules = create_modules()

    # Create environment
    env = Environment(config)

    # Register modules with execution order
    env.register_module('weather', modules['weather'], execution_order=1)
    env.register_module('pv', modules['pv'], execution_order=2)
    env.register_module('thermal', modules['thermal'], execution_order=3)
    env.register_module('battery', modules['battery'], execution_order=4)

    # Initialize environment
    env.initialize()

    # Create controller
    controller = BasicController()

    # Storage for results
    results = {
        'time': [],
        'zone_temp': [],
        'outdoor_temp': [],
        'pv_power': [],
        'battery_soc': [],
        'battery_power': [],
        'grid_power': [],
        'hvac_power': [],
        'comfort_violation': [],
        'electricity_cost': []
    }

    logger.info("Running simulation loop...")

    # Simulation loop
    step = 0
    while True:
        # Get observation
        obs = env.get_observation()

        # Get control action
        action = controller.get_action(obs)

        # Step environment
        observation, reward, done, info = env.step(action)

        # Store results
        results['time'].append(step * config.timestep)
        results['zone_temp'].append(observation.zone_temp)
        results['outdoor_temp'].append(info['module_outputs']['weather']['weather']['outdoor_temperature'])
        results['pv_power'].append(observation.pv_power)
        results['battery_soc'].append(observation.battery_soc)
        results['battery_power'].append(info['module_outputs']['battery']['battery_power'])
        results['grid_power'].append(info['module_outputs']['weather'].get('grid_power', 0.0))  # Placeholder
        results['hvac_power'].append(info['module_outputs']['thermal']['hvac_power'])
        results['comfort_violation'].append(info['kpis']['comfort_violation'])
        results['electricity_cost'].append(info['kpis']['electricity_cost'])

        # Log progress
        if step % 96 == 0:  # Every day (96 timesteps = 24 hours)
            day = step // 96 + 1
            logger.info(f"Day {day}: Zone temp: {observation.zone_temp:.1f}°C, "
                        f"Battery SOC: {observation.battery_soc:.1f}%, "
                        f"PV power: {observation.pv_power:.1f} kW")

        step += 1

        if done:
            break

    logger.info("Simulation completed!")

    # Get performance summary
    performance = env.get_performance_summary()
    logger.info("Performance Summary:")
    for module, stats in performance.items():
        if isinstance(stats, dict) and 'mean_time' in stats:
            logger.info(f"  {module}: avg {stats['mean_time'] * 1000:.1f}ms, "
                        f"max {stats['max_time'] * 1000:.1f}ms")

    return results, performance


def analyze_results(results: Dict[str, List], performance: Dict[str, Any]):
    """Analyze and visualize simulation results."""

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS ANALYSIS")
    print("=" * 60)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"Simulation duration: {len(df)} timesteps ({len(df) * 0.25:.1f} hours)")
    print(f"Average zone temperature: {df['zone_temp'].mean():.1f}°C")
    print(f"Temperature range: {df['zone_temp'].min():.1f}°C to {df['zone_temp'].max():.1f}°C")
    print(f"Average outdoor temperature: {df['outdoor_temp'].mean():.1f}°C")

    # Energy statistics
    total_pv_energy = df['pv_power'].sum() * 0.25  # kWh
    total_hvac_energy = df['hvac_power'].sum() * 0.25  # kWh
    avg_battery_soc = df['battery_soc'].mean()

    print(f"\nEnergy Statistics:")
    print(f"Total PV generation: {total_pv_energy:.1f} kWh")
    print(f"Total HVAC consumption: {total_hvac_energy:.1f} kWh")
    print(f"Average battery SOC: {avg_battery_soc:.1f}%")
    print(f"Battery SOC range: {df['battery_soc'].min():.1f}% to {df['battery_soc'].max():.1f}%")

    # Comfort statistics
    comfort_violations = df['comfort_violation'].sum()
    violation_hours = (df['comfort_violation'] > 0).sum() * 0.25

    print(f"\nComfort Statistics:")
    print(f"Total comfort violation: {comfort_violations:.1f} °C·hours")
    print(f"Hours with comfort violations: {violation_hours:.1f} hours")
    print(f"Comfort compliance: {(1 - violation_hours / (len(df) * 0.25)) * 100:.1f}%")

    # Economic statistics
    total_cost = df['electricity_cost'].sum()
    avg_cost_per_hour = df['electricity_cost'].mean()

    print(f"\nEconomic Statistics:")
    print(f"Total electricity cost: ${total_cost:.2f}")
    print(f"Average cost per hour: ${avg_cost_per_hour:.3f}")
    print(f"Daily average cost: ${avg_cost_per_hour * 24:.2f}")

    # Create visualizations
    create_plots(df)

    # Performance analysis
    if performance and 'kpis' in performance:
        print(f"\nKPI Summary:")
        kpi_stats = performance['kpis']
        for kpi, stats in kpi_stats.items():
            print(f"  {kpi}: mean={stats['mean']:.3f}, max={stats['max']:.3f}")


def create_plots(df: pd.DataFrame):
    """Create visualization plots for the simulation results."""

    try:
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Building Energy Simulation Results', fontsize=16, fontweight='bold')

        # Plot 1: Temperature profiles
        ax1 = axes[0, 0]
        ax1.plot(df['time'], df['zone_temp'], label='Zone Temperature', color='red', linewidth=2)
        ax1.plot(df['time'], df['outdoor_temp'], label='Outdoor Temperature', color='blue', alpha=0.7)
        ax1.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Comfort Min')
        ax1.axhline(y=24, color='orange', linestyle='--', alpha=0.7, label='Comfort Max')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature Profiles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Power flows
        ax2 = axes[0, 1]
        ax2.plot(df['time'], df['pv_power'], label='PV Power', color='gold', linewidth=2)
        ax2.plot(df['time'], df['hvac_power'], label='HVAC Power', color='purple', linewidth=2)
        ax2.plot(df['time'], df['battery_power'], label='Battery Power', color='green', linewidth=2)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Power (kW)')
        ax2.set_title('Power Flows')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Battery SOC
        ax3 = axes[1, 0]
        ax3.plot(df['time'], df['battery_soc'], label='Battery SOC', color='green', linewidth=2)
        ax3.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Min SOC')
        ax3.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Max SOC')
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('SOC (%)')
        ax3.set_title('Battery State of Charge')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Comfort violations
        ax4 = axes[1, 1]
        ax4.plot(df['time'], df['comfort_violation'], color='red', linewidth=2)
        ax4.fill_between(df['time'], 0, df['comfort_violation'], alpha=0.3, color='red')
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Comfort Violation (°C)')
        ax4.set_title('Comfort Violations')
        ax4.grid(True, alpha=0.3)

        # Plot 5: Cumulative electricity cost
        ax5 = axes[2, 0]
        cumulative_cost = df['electricity_cost'].cumsum()
        ax5.plot(df['time'], cumulative_cost, color='darkgreen', linewidth=2)
        ax5.set_xlabel('Time (hours)')
        ax5.set_ylabel('Cumulative Cost ($)')
        ax5.set_title('Cumulative Electricity Cost')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Daily energy balance
        ax6 = axes[2, 1]
        # Calculate daily totals
        df['day'] = (df['time'] / 24).astype(int)
        daily_pv = df.groupby('day')['pv_power'].sum() * 0.25  # Convert to kWh
        daily_hvac = df.groupby('day')['hvac_power'].sum() * 0.25  # Convert to kWh

        x = range(len(daily_pv))
        width = 0.35
        ax6.bar([i - width / 2 for i in x], daily_pv, width, label='PV Generation', color='gold', alpha=0.8)
        ax6.bar([i + width / 2 for i in x], daily_hvac, width, label='HVAC Consumption', color='purple', alpha=0.8)
        ax6.set_xlabel('Day')
        ax6.set_ylabel('Energy (kWh)')
        ax6.set_title('Daily Energy Balance')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nPlots saved as 'simulation_results.png'")

    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Error creating plots: {e}")


def advanced_analysis(results: Dict[str, List]):
    """Perform advanced analysis on simulation results."""

    df = pd.DataFrame(results)

    print(f"\n" + "=" * 60)
    print("ADVANCED ANALYSIS")
    print("=" * 60)

    # Energy efficiency metrics
    pv_generation_total = df['pv_power'].sum() * 0.25  # kWh
    hvac_consumption_total = df['hvac_power'].sum() * 0.25  # kWh

    if pv_generation_total > 0:
        pv_utilization = min(100, (hvac_consumption_total / pv_generation_total) * 100)
        print(f"PV utilization rate: {pv_utilization:.1f}%")

    # Battery performance
    battery_cycles = calculate_battery_cycles(df['battery_power'].tolist())
    print(f"Battery equivalent cycles: {battery_cycles:.2f}")

    # Peak demand analysis
    max_hvac_power = df['hvac_power'].max()
    avg_hvac_power = df['hvac_power'].mean()
    peak_to_avg_ratio = max_hvac_power / avg_hvac_power if avg_hvac_power > 0 else 0

    print(f"Peak HVAC demand: {max_hvac_power:.1f} kW")
    print(f"Average HVAC demand: {avg_hvac_power:.1f} kW")
    print(f"Peak-to-average ratio: {peak_to_avg_ratio:.1f}")

    # Time-of-use analysis
    df['hour'] = (df['time'] % 24).astype(int)
    hourly_avg_power = df.groupby('hour')['hvac_power'].mean()
    peak_hours = hourly_avg_power.nlargest(4).index.tolist()  # Top 4 peak hours

    print(f"Peak demand hours: {peak_hours}")

    # Thermal comfort analysis
    comfort_violations_by_hour = df.groupby('hour')['comfort_violation'].sum()
    worst_comfort_hours = comfort_violations_by_hour.nlargest(3).index.tolist()

    print(f"Hours with most comfort violations: {worst_comfort_hours}")

    # Cost analysis by time period
    df['time_period'] = df['hour'].apply(classify_time_period)
    cost_by_period = df.groupby('time_period')['electricity_cost'].sum()

    print(f"Costs by time period:")
    for period, cost in cost_by_period.items():
        print(f"  {period}: ${cost:.2f}")


def classify_time_period(hour: int) -> str:
    """Classify hour into time period for cost analysis."""
    if 22 <= hour or hour <= 6:
        return "off_peak"
    elif 16 <= hour <= 20:
        return "peak"
    else:
        return "mid_peak"


def calculate_battery_cycles(power_history: List[float], capacity_kwh: float = 20.0) -> float:
    """
    Calculate equivalent battery cycles from power history.

    Simplified cycle counting - in practice, use rainflow counting.
    """
    total_energy_throughput = sum(abs(p) * 0.25 for p in power_history)  # kWh
    equivalent_cycles = total_energy_throughput / (2 * capacity_kwh)  # Full cycles
    return equivalent_cycles


def demonstrate_different_controllers():
    """Demonstrate different control strategies."""

    print(f"\n" + "=" * 60)
    print("CONTROLLER COMPARISON")
    print("=" * 60)

    # This would run multiple simulations with different controllers
    # For brevity, we'll just describe the concept

    controllers = [
        "Rule-based (demonstrated above)",
        "Model Predictive Control (MPC)",
        "Reinforcement Learning Agent",
        "Optimal Control (Dynamic Programming)"
    ]

    print("Different control strategies that could be implemented:")
    for i, controller in enumerate(controllers, 1):
        print(f"  {i}. {controller}")

    print(f"\nEach controller would optimize different objectives:")
    print("  - Rule-based: Simple comfort maintenance")
    print("  - MPC: Predictive optimization with forecasts")
    print("  - RL: Learning-based adaptation over time")
    print("  - Optimal: Mathematical optimization with known system")


if __name__ == "__main__":
    """Main execution function."""

    print("Building Energy Physics-Informed ML Environment Demo")
    print("=" * 60)

    try:
        # Run the main simulation
        results, performance = run_simulation()

        # Analyze results
        analyze_results(results, performance)

        # Advanced analysis
        advanced_analysis(results)

        # Show controller comparison concept
        demonstrate_different_controllers()

        print(f"\n" + "=" * 60)
        print("SIMULATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print(f"\nKey Achievements:")
        print("  ✓ Successfully integrated thermal, electrical, and weather models")
        print("  ✓ Demonstrated physics-informed simulation environment")
        print("  ✓ Implemented rule-based control strategy")
        print("  ✓ Generated comprehensive performance analysis")
        print("  ✓ Created visualization of simulation results")

        print(f"\nNext Steps:")
        print("  • Implement MPC or RL controllers")
        print("  • Add more detailed component models")
        print("  • Integrate with real weather data")
        print("  • Extend to multi-building scenarios")
        print("  • Add more sophisticated KPI calculations")

    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback

        traceback.print_exc()