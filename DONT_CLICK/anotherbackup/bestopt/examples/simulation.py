import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Optional
import logging
import json

from ..env.core.environment import Environment
from ..config.cfg import (configuration, register_module, build_module)
from ..env.modules.building.dynamics import ThermalDynamicsModule
from ..env.modules.hvac.hvac_system import HVACModule
from ..env.modules.ders.battery import BatteryModule
from ..env.modules.ders.ev import EVModule
from ..env.modules.ders.pv import PVModule
from ..env.modules.ders.tes import ThermalEnergyStorageModule
from ..env.disturbances.weather import WeatherModule
from ..env.disturbances.price_signals import PriceSignalModule
from ..env.disturbances.occupancy import OccupancyModule
from ..env.controllers.rule_based import RuleBasedController
from env.evaluation.kpi_tracker import KPITracker
from env.evaluation.comfort_models import ComfortEvaluator
from env.utils.visualizer import SimulationVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimulationExample:
    """Main simulation class that orchestrates the entire simulation process."""

    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the simulation example.

        Args:
            config_overrides: Optional configuration overrides
        """
        self.logger = logging.getLogger(f"{__name__}.SimulationExample")

        # Create configuration
        self.config = self._create_configuration(config_overrides)

        # Initialize components
        self.env = None
        self.controller = None
        self.kpi_tracker = None
        self.visualizer = None

        # Storage for results
        self.results = {
            'states': [],
            'actions': [],
            'observations': [],
            'rewards': [],
            'kpis': [],
            'comfort': [],
            'energy': []
        }

    def _create_configuration(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create simulation configuration with optional overrides."""

        # Default configuration
        default_config = {
            'env': {
                'timestep': 15.0,  # 15 minutes
                'simulation_days': 7,
                'start_time': 0.0
            },
            'building': {
                'building_area': 200.0,  # m²
                'building_volume': 600.0,  # m³
                'building_thermal_mass': 50000.0,  # kJ/K
                'building_ua_value': 300.0,  # W/K
                'number_of_zones': 1,
                'zone_names': ['main']
            },
            'hvac': {
                'hvac_capacity': 10.0,  # kW
                'hvac_cop_cooling': 3.0,
                'hvac_cop_heating': 2.5,
                'min_supply_temp': 12.0,  # °C
                'max_supply_temp': 35.0,  # °C
                'design_airflow': 0.5  # m³/s
            },
            'battery': {
                'battery_capacity': 20.0,  # kWh
                'battery_c_rate': 0.25,
                'battery_charge_efficiency': 0.95,
                'battery_discharge_efficiency': 0.95,
                'battery_soc_min': 0.1,
                'battery_soc_max': 0.9,
                'battery_initial_soc': 0.5
            },
            'ev': {
                'ev_capacity': 60.0,  # kWh
                'ev_c_rate': 0.25,
                'ev_charge_efficiency': 0.90,
                'ev_discharge_efficiency': 0.90,
                'ev_soc_min': 0.10,
                'ev_soc_max': 0.90,
                'ev_initial_soc': 0.5,
                'ev_max_charge_power': 7.4,  # kW
                'ev_v2g_enabled': True
            },
            'pv': {
                'pv_capacity': 8.0,  # kW
                'pv_tilt': 30.0,  # degrees
                'pv_azimuth': 180.0,  # degrees (south-facing)
                'pv_efficiency': 0.20,
                'pv_degradation_rate': 0.005  # per year
            },
            'tes': {
                'tes_capacity': 50.0,  # kWh thermal
                'tes_charge_efficiency': 0.90,
                'tes_discharge_efficiency': 0.85,
                'tes_loss_coefficient': 0.01,  # per hour
                'tes_max_charge_rate': 10.0,  # kW
                'tes_max_discharge_rate': 10.0  # kW
            }
        }

        # Apply overrides if provided
        if overrides:
            for key, value in overrides.items():
                if key in default_config:
                    default_config[key].update(value)

        return configuration(**default_config)

    def setup_modules(self):
        """Register and initialize all simulation modules."""

        self.logger.info("Setting up simulation modules...")

        # Register all modules with the configuration system
        register_module('building', 'building', ThermalDynamicsModule)
        register_module('hvac', 'hvac', HVACModule)
        register_module('battery', 'battery', BatteryModule)
        register_module('ev', 'ev', EVModule)
        register_module('pv', 'pv', PVModule)
        register_module('tes', 'tes', ThermalEnergyStorageModule, expects_dict=True)

        # Build module instances
        modules = build_module(
            self.config,
            enabled=['building', 'hvac', 'battery', 'ev', 'pv', 'tes']
        )

        # Add disturbances modules (these might not use the config system)
        modules['weather'] = WeatherModule()
        modules['price_signals'] = PriceSignalModule()
        modules['occupancy'] = OccupancyModule()

        self.logger.info(f"Registered {len(modules)} modules")
        return modules

    def setup_environment(self):
        """Initialize the simulation environment."""

        self.logger.info("Setting up environment...")

        # Get modules
        modules = self.setup_modules()

        # Create environment directly with config dictionary
        # The Environment class should accept either a dict or SimulationConfig
        self.env = Environment(config=self.config, modules=modules)

        # Set execution order for modules
        execution_order = {
            'weather': 0,
            'price_signals': 1,
            'occupancy': 2,
            'pv': 3,
            'building': 4,
            'hvac': 5,
            'battery': 6,
            'ev': 7,
            'tes': 8
        }

        for name, order in execution_order.items():
            if name in modules:
                self.env.register_module(name, modules[name], order)

        # Initialize environment
        self.env.initialize()

        self.logger.info("Environment setup complete")

    def setup_controller(self, controller_type: str = 'rule_based'):
        """
        Setup the controller for the simulation.

        Args:
            controller_type: Type of controller ('rule_based', 'mpc', 'rl')
        """

        self.logger.info(f"Setting up {controller_type} controller...")

        if controller_type == 'rule_based':
            self.controller = RuleBasedController(self.config)
        elif controller_type == 'mpc':
            self.controller = MPCController(self.config, horizon=24)
        elif controller_type == 'rl':
            # Would load a trained RL agent here
            raise NotImplementedError("RL controller not implemented in this example")
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

        self.logger.info("Controller setup complete")

    def setup_evaluation(self):
        """Setup KPI tracking and evaluation tools."""

        self.logger.info("Setting up evaluation tools...")

        self.kpi_tracker = KPITracker()
        self.comfort_evaluator = ComfortEvaluator()
        self.visualizer = SimulationVisualizer()

        self.logger.info("Evaluation tools setup complete")

    def run_simulation(self, controller_type: str = 'rule_based'):
        """
        Run the complete simulation.

        Args:
            controller_type: Type of controller to use
        """

        self.logger.info("Starting simulation...")

        # Setup components
        self.setup_environment()
        self.setup_controller(controller_type)
        self.setup_evaluation()

        # Reset environment
        observation = self.env.reset()

        # Calculate total timesteps
        timestep_hours = self.config['env']['timestep'] / 60.0
        total_hours = self.config['env']['simulation_days'] * 24
        total_timesteps = int(total_hours / timestep_hours)

        self.logger.info(f"Running {total_timesteps} timesteps...")

        # Simulation loop
        for step in range(total_timesteps):

            # Get control action from controller
            action = self.controller.get_action(observation, self.env.current_state)

            # Execute environment step
            observation, reward, done, info = self.env.step(action)

            # Store results
            self._store_results(
                state=self.env.current_state,
                action=action,
                observation=observation,
                reward=reward,
                info=info
            )

            # Update KPI tracker
            self.kpi_tracker.update(
                state=self.env.current_state,
                action=action,
                disturbance=self.env.current_disturbance
            )

            # Log progress every hour
            if step % int(60 / self.config['env']['timestep']) == 0:
                hours_simulated = step * timestep_hours
                self.logger.info(f"Simulated {hours_simulated:.1f} hours ({step}/{total_timesteps} steps)")
                self._log_current_status()

            if done:
                break

        self.logger.info("Simulation complete!")

        # Finalize results
        self._finalize_results()

        return self.results

    def _store_results(self, state: State, action: Action,
                       observation, reward: float, info: Dict):
        """Store simulation results for analysis."""

        self.results['states'].append(state.to_dict())
        self.results['actions'].append(action.to_dict())
        self.results['observations'].append(observation)
        self.results['rewards'].append(reward)

        if 'kpis' in info:
            self.results['kpis'].append(info['kpis'])

        # Calculate comfort metrics
        comfort_metrics = self.comfort_evaluator.evaluate(
            state=state,
            disturbance=self.env.current_disturbance
        )
        self.results['comfort'].append(comfort_metrics)

        # Energy metrics
        energy_metrics = {
            'grid_power': state.electrical.grid_building,
            'pv_power': state.electrical.pv_building,
            'battery_power': state.electrical.battery_to_building,
            'ev_power': state.electrical.ev_to_building,
            'hvac_power': state.electrical.hvac_power,
            'total_load': (state.electrical.hvac_power +
                          state.electrical.lighting_power +
                          state.electrical.plug_loads_power)
        }
        self.results['energy'].append(energy_metrics)

    def _log_current_status(self):
        """Log current simulation status."""

        state = self.env.current_state
        disturbance = self.env.current_disturbance

        self.logger.debug(
            f"Status - "
            f"Temp: {state.thermal.zone_temperatures.get('main', 0):.1f}°C, "
            f"Battery SOC: {state.electrical.battery_soc:.1f}%, "
            f"EV SOC: {state.electrical.ev_soc:.1f}%, "
            f"Grid: {state.electrical.grid_building:.2f}kW, "
            f"Price: ${disturbance.prices.electricity_price:.3f}/kWh"
        )

    def _finalize_results(self):
        """Process and finalize simulation results."""

        # Calculate summary statistics
        self.results['summary'] = {
            'total_energy_consumed': sum(r['total_load'] for r in self.results['energy']) *
                                    self.config['env']['timestep'] / 60.0,
            'total_grid_energy': sum(max(0, r['grid_power']) for r in self.results['energy']) *
                               self.config['env']['timestep'] / 60.0,
            'total_pv_generated': sum(r['pv_power'] for r in self.results['energy']) *
                                self.config['env']['timestep'] / 60.0,
            'average_comfort': np.mean([r.get('pmv', 0) for r in self.results['comfort']]),
            'comfort_violations': sum(1 for r in self.results['comfort']
                                    if abs(r.get('pmv', 0)) > 0.5),
            'total_cost': sum(r.get('electricity_cost', 0) for r in self.results['kpis']),
            'peak_demand': max(r['grid_power'] for r in self.results['energy'])
        }

        self.logger.info(f"Summary: {self.results['summary']}")

    def analyze_results(self):
        """Analyze and visualize simulation results."""

        self.logger.info("Analyzing results...")

        # Create DataFrame for easier analysis
        df_energy = pd.DataFrame(self.results['energy'])
        df_states = pd.DataFrame([s['thermal']['zone_temperatures'].get('main', 22)
                                 for s in self.results['states']],
                                columns=['temperature'])

        # Add timestamp
        timestep_hours = self.config['env']['timestep'] / 60.0
        df_energy['hours'] = np.arange(len(df_energy)) * timestep_hours
        df_states['hours'] = df_energy['hours']

        # Create visualizations
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        # Temperature plot
        axes[0].plot(df_states['hours'], df_states['temperature'], label='Zone Temperature')
        axes[0].axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Comfort Min')
        axes[0].axhline(y=26, color='r', linestyle='--', alpha=0.5, label='Comfort Max')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].legend()
        axes[0].grid(True)

        # Power flows
        axes[1].plot(df_energy['hours'], df_energy['grid_power'], label='Grid')
        axes[1].plot(df_energy['hours'], df_energy['pv_power'], label='PV')
        axes[1].plot(df_energy['hours'], df_energy['battery_power'], label='Battery')
        axes[1].plot(df_energy['hours'], df_energy['total_load'], label='Total Load', alpha=0.7)
        axes[1].set_ylabel('Power (kW)')
        axes[1].legend()
        axes[1].grid(True)

        # Battery and EV SOC
        battery_socs = [s['electrical']['battery_soc'] for s in self.results['states']]
        ev_socs = [s['electrical']['ev_soc'] for s in self.results['states']]
        axes[2].plot(df_energy['hours'], battery_socs, label='Battery SOC')
        axes[2].plot(df_energy['hours'], ev_socs, label='EV SOC')
        axes[2].set_ylabel('SOC (%)')
        axes[2].legend()
        axes[2].grid(True)

        # Cumulative cost
        if self.results['kpis']:
            costs = [r.get('electricity_cost', 0) for r in self.results['kpis']]
            cumulative_cost = np.cumsum(costs)
            axes[3].plot(df_energy['hours'], cumulative_cost, label='Cumulative Cost')
            axes[3].set_ylabel('Cost ($)')
            axes[3].legend()
            axes[3].grid(True)

        axes[3].set_xlabel('Time (hours)')

        plt.tight_layout()
        plt.savefig('simulation_results.png')
        plt.show()

        self.logger.info("Analysis complete. Results saved to simulation_results.png")

        return df_energy, df_states

    def save_results(self, filename: str = 'simulation_results.json'):
        """Save simulation results to file."""

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        # Deep convert all results
        serializable_results = json.loads(
            json.dumps(self.results, default=convert_to_serializable)
        )

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {filename}")


def main():
    """Main function to run the simulation example."""

    # Example 1: Basic simulation with rule-based controller
    print("\n" + "="*60)
    print("Example 1: Rule-Based Controller Simulation")
    print("="*60 + "\n")

    sim = SimulationExample()
    results = sim.run_simulation(controller_type='rule_based')
    df_energy, df_states = sim.analyze_results()
    sim.save_results('rule_based_results.json')

    # Example 2: Simulation with custom configuration
    print("\n" + "="*60)
    print("Example 2: Custom Configuration Simulation")
    print("="*60 + "\n")

    custom_config = {
        'env': {
            'timestep': 30.0,  # 30 minutes
            'simulation_days': 3
        },
        'battery': {
            'battery_capacity': 30.0,  # Larger battery
            'battery_c_rate': 0.5  # Faster charging
        },
        'pv': {
            'pv_capacity': 12.0  # Larger PV system
        }
    }

    sim_custom = SimulationExample(config_overrides=custom_config)
    results_custom = sim_custom.run_simulation()
    sim_custom.save_results('custom_config_results.json')

    # Example 3: Comparative analysis
    print("\n" + "="*60)
    print("Example 3: Comparative Analysis")
    print("="*60 + "\n")

    print(f"Rule-based controller total cost: ${results['summary']['total_cost']:.2f}")
    print(f"Custom config total cost: ${results_custom['summary']['total_cost']:.2f}")
    print(f"Rule-based comfort violations: {results['summary']['comfort_violations']}")
    print(f"Custom config comfort violations: {results_custom['summary']['comfort_violations']}")

    # Example 4: Batch simulations for sensitivity analysis
    print("\n" + "="*60)
    print("Example 4: Sensitivity Analysis")
    print("="*60 + "\n")

    battery_sizes = [10.0, 20.0, 30.0, 40.0]
    sensitivity_results = []

    for battery_size in battery_sizes:
        config = {
            'battery': {'battery_capacity': battery_size}
        }
        sim_batch = SimulationExample(config_overrides=config)
        results_batch = sim_batch.run_simulation()

        sensitivity_results.append({
            'battery_size': battery_size,
            'total_cost': results_batch['summary']['total_cost'],
            'grid_energy': results_batch['summary']['total_grid_energy'],
            'comfort_violations': results_batch['summary']['comfort_violations']
        })

        print(f"Battery {battery_size}kWh - Cost: ${results_batch['summary']['total_cost']:.2f}")

    # Plot sensitivity analysis results
    df_sensitivity = pd.DataFrame(sensitivity_results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df_sensitivity['battery_size'], df_sensitivity['total_cost'], 'o-')
    axes[0].set_xlabel('Battery Size (kWh)')
    axes[0].set_ylabel('Total Cost ($)')
    axes[0].grid(True)

    axes[1].plot(df_sensitivity['battery_size'], df_sensitivity['grid_energy'], 'o-')
    axes[1].set_xlabel('Battery Size (kWh)')
    axes[1].set_ylabel('Grid Energy (kWh)')
    axes[1].grid(True)

    axes[2].plot(df_sensitivity['battery_size'], df_sensitivity['comfort_violations'], 'o-')
    axes[2].set_xlabel('Battery Size (kWh)')
    axes[2].set_ylabel('Comfort Violations')
    axes[2].grid(True)

    plt.suptitle('Sensitivity Analysis: Battery Size Impact')
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png')
    plt.show()

    print("\nSensitivity analysis saved to sensitivity_analysis.png")

    print("\n" + "="*60)
    print("Simulation Examples Complete!")
    print("="*60)


if __name__ == "__main__":
    main()