import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from bestopt.env.core.data_structure import (
    ClusterAction, HVACSystemAction, DERSystemAction, SystemType
)

warnings.filterwarnings('ignore')


class SimulationManager:
    """Manager class for building simulation operations - Simplified Power Flow Model"""

    def __init__(self, base_path: str = "./simulation_results"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.colors = {
            'hvac': '#3498DB',
            'lighting': '#2ECC71',
            'cooking': '#F39C12',
            'pc': '#9B59B6',
            'tv': '#1ABC9C',
            'ev': '#E67E22',
            'pv': '#FDB813',
            'battery': '#2980B9',
            'grid': '#2C3E50',
            'peak': '#FF6B6B',
            'curtail': '#95A5A6'
        }

        # Comparison color schemes
        self.sim_colors = {
            'sim1': '#2C3E50',  # Dark blue-gray
            'sim2': '#E74C3C',  # Red
        }

    def run_and_save_simulation(self,
                                env,
                                total_steps: int = 96,
                                simulation_name: str = None,
                                config_info: Dict = None) -> Tuple[Dict, str]:
        """Run simulation with simplified power flow model."""
        if simulation_name is None:
            simulation_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"Starting simulation: {simulation_name}")

        # Initialize data storage - SIMPLIFIED MODEL
        data_dict = {
            # Timestep
            'timesteps': [],

            # Thermal data
            'zone_temperature': [],
            'cooling_setpoint': [],
            'heating_setpoint': [],
            'hvac_thermal_load': [],
            'hvac_power': [],
            'supply_air_temp_setpoint': [],
            'supply_air_temp_actual': [],
            'supply_air_flow_setpoint': [],
            'supply_air_flow_actual': [],
            'chilled_water_flow_actual': [],
            'chiller_supply_water_temp': [],

            # Demand side breakdown [kW]
            'building_load': [],  # Total building electrical load
            'hvac_power_kw': [],
            'lighting': [],
            'cooking': [],
            'pc': [],
            'tv': [],
            'ev_charging': [],  # Total EV charging power
            'ev_tesla_charging': [],  # Individual EV charging
            'ev_nissan_charging': [],
            'total_demand': [],  # building_load + ev_charging

            # Generation side [kW]
            'pv_generation': [],

            # Battery (controllable) [kW]
            'battery_power': [],  # positive=charge, negative=discharge
            'battery_soc': [],

            # EV SOC (for tracking)
            'ev_tesla_soc': [],
            'ev_nissan_soc': [],

            # Grid and balance [kW]
            'net_grid': [],  # Net grid import (>=0 if no export)
            'curtailment': [],  # Curtailed PV

            # Peak signal
            'is_peak': [],
            'electricity_price': [],
        }

        # Run simulation
        for timestep in range(min(total_steps, env.total_step)):
            # # Initialize cluster action
            # cluster_action = ClusterAction(cluster_id="residential_cluster_1")
            #
            # # HVAC action
            # hvac_action = HVACSystemAction(
            #     system_id="hvac_system_1",
            #     system_type=SystemType.HVAC.value
            # )
            #
            # hvac_action.supply_temp_setpoint_c = 12
            # hvac_action.supply_airflow_setpoint_m3s = 0.6
            # cluster_action.thermal.system_actions["hvac_system_1"] = hvac_action
            #
            # # DER action (will be computed by controller)
            # der_action = DERSystemAction(
            #     system_id="der_system_1",
            #     system_type=SystemType.DER.value
            # )
            # cluster_action.electrical.system_actions["der_system_1"] = der_action
            #
            # # Step environment
            # observations, done, info = env.step(
            #     external_actions={"residential_cluster_1": cluster_action}
            # )

            observations, done, info = env.step()

            # Extract data
            cluster_id = 'residential_cluster_1'
            building_id = 'SFH_1'
            hvac_system_id = env.building_system_map[building_id].get('thermal')
            der_system_id = env.building_system_map[building_id].get('electrical')

            hvac_system = env.system_modules[hvac_system_id]
            der_system = env.system_modules[der_system_id]

            # Thermal data
            zone_temp = env.cluster_states[cluster_id].thermal.systems[
                'SFH_1_building'].components['zone0'].temperature
            cooling_sp = env.cluster_actions[cluster_id].thermal.system_actions[
                'hvac_system_1'].cooling_setpoint_c
            heating_sp = env.cluster_actions[cluster_id].thermal.system_actions[
                'hvac_system_1'].heating_setpoint_c

            HVAC_power = hvac_system.FCU_power_total_W
            HVAC_thermal = hvac_system.Q_zone_actual_W
            SAT_actual = hvac_system.SAT_actual_C
            SAF_actual = hvac_system.SA_flow_actual_m3s
            SAT_sp = env.cluster_actions[cluster_id].thermal.system_actions[
                'hvac_system_1'].supply_temp_setpoint_c
            SAF_sp = env.cluster_actions[cluster_id].thermal.system_actions[
                'hvac_system_1'].supply_airflow_setpoint_m3s
            CHW_flow = hvac_system.CHW_flow_actual_m3s
            CHW_temp = hvac_system.CHW_supply_temp_C

            # Electrical loads [kW]
            hvac_kw = HVAC_power / 1000
            cooking = env.electrical_zone_modules['SFH_1.zone0'].cooking_power / 1000
            pc = env.electrical_zone_modules['SFH_1.zone0'].pc_power / 1000
            tv = env.electrical_zone_modules['SFH_1.zone0'].tv_power / 1000
            lighting = env.electrical_zone_modules['SFH_1.zone0'].lighting_power / 1000

            building_load = hvac_kw + env.cluster_states[cluster_id].electrical.systems[
                'SFH_1_building'].components['electrical'].building_power_w / 1000

            # Get DER action from controller
            der_action = env.cluster_actions[cluster_id].electrical.system_actions['der_system_1']

            # EV charging (from action)
            ev_charging_dict = getattr(der_action, 'ev_charging', {})
            ev_tesla_charging = ev_charging_dict.get('ev_tesla', 0.0)
            ev_nissan_charging = ev_charging_dict.get('ev_nissan', 0.0)
            total_ev_charging = ev_tesla_charging + ev_nissan_charging

            # Total demand
            total_demand = building_load + total_ev_charging

            # PV generation
            pv_generation = der_system.pv_states['pv_1'].generation_w / 1000

            # Battery power (from action)
            battery_power_dict = getattr(der_action, 'battery_power', {})
            battery_power = sum(battery_power_dict.values())

            # SOC values
            bat_soc = der_system.battery_states['bat_1'].soc
            ev_tesla_soc = der_system.ev_states['ev_tesla'].soc
            ev_nissan_soc = der_system.ev_states['ev_nissan'].soc

            # Grid and curtailment
            net_grid = getattr(der_action, 'grid_import', 0.0)
            curtailment = getattr(der_action, 'curtailment', 0.0)

            # Peak signal
            is_peak = env.disturbance.prices.peaksignal
            elec_price = env.disturbance.prices.electricity_price

            # Store data
            data_dict['timesteps'].append(timestep)

            # Thermal
            data_dict['zone_temperature'].append(zone_temp)
            data_dict['cooling_setpoint'].append(cooling_sp)
            data_dict['heating_setpoint'].append(heating_sp)
            data_dict['hvac_thermal_load'].append(HVAC_thermal)
            data_dict['hvac_power'].append(HVAC_power)
            data_dict['supply_air_temp_setpoint'].append(SAT_sp)
            data_dict['supply_air_temp_actual'].append(SAT_actual)
            data_dict['supply_air_flow_setpoint'].append(SAF_sp)
            data_dict['supply_air_flow_actual'].append(SAF_actual)
            data_dict['chilled_water_flow_actual'].append(CHW_flow)
            data_dict['chiller_supply_water_temp'].append(CHW_temp)

            # Demand breakdown
            data_dict['building_load'].append(building_load)
            data_dict['hvac_power_kw'].append(hvac_kw)
            data_dict['lighting'].append(lighting)
            data_dict['cooking'].append(cooking)
            data_dict['pc'].append(pc)
            data_dict['tv'].append(tv)
            data_dict['ev_charging'].append(total_ev_charging)
            data_dict['ev_tesla_charging'].append(ev_tesla_charging)
            data_dict['ev_nissan_charging'].append(ev_nissan_charging)
            data_dict['total_demand'].append(total_demand)

            # Generation
            data_dict['pv_generation'].append(pv_generation)

            # Battery
            data_dict['battery_power'].append(battery_power)
            data_dict['battery_soc'].append(bat_soc)

            # EV SOC
            data_dict['ev_tesla_soc'].append(ev_tesla_soc)
            data_dict['ev_nissan_soc'].append(ev_nissan_soc)

            # Grid
            data_dict['net_grid'].append(net_grid)
            data_dict['curtailment'].append(curtailment)
            data_dict['is_peak'].append(is_peak)
            data_dict['electricity_price'].append(elec_price)

            if timestep % 20 == 0:
                print(f"Step {timestep}/{total_steps}: Temp={zone_temp:.1f}�C, "
                      f"Bat SOC={bat_soc:.1%}, Grid={net_grid:.1f}kW")

            if done:
                break

        # Convert to numpy arrays
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])

        # Save metadata
        metadata = {
            'simulation_name': simulation_name,
            'timestamp': datetime.now().isoformat(),
            'total_steps': len(data_dict['timesteps']),
            'config_info': config_info or {}
        }

        # Save data
        save_path = Path(self.base_path) / "operation" / f"{simulation_name}.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({'data': data_dict, 'metadata': metadata}, f)

        # Also save CSV
        df = pd.DataFrame(data_dict)
        df.to_csv(self.base_path / "operation" / f"{simulation_name}.csv", index=False)

        print(f"\nSimulation completed: {len(data_dict['timesteps'])} steps")
        print(f"Data saved to: {save_path}")

        return data_dict, str(save_path)

    def plot_electrical_results(self,
                                simulation_name: str = None,
                                data_dict: Dict = None,
                                save_figure: bool = True,
                                show_figure: bool = True) -> plt.Figure:
        """
        Create electrical system plots for journal publication.

        3 subplots (height ratio 2:1:2):
        1. Demand breakdown (stacked, EV on top) with PV generation (negative)
        2. Battery SOC
        3. Net grid load
        """
        if data_dict is None:
            data_dict, metadata = self.load_simulation_data(simulation_name)

        timesteps = data_dict['timesteps']
        time_hours = timesteps * 0.25
        peak_mask = data_dict['is_peak'].astype(bool)

        # Journal-quality settings
        FONT_SIZE = 7
        LEGEND_FONT_SIZE = 5.5

        # Create figure with 2:1:2 height ratio
        fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.0), dpi=300, facecolor='white',
                                 gridspec_kw={'height_ratios': [2, 1, 2]})

        # Set global font size
        plt.rcParams.update({'font.size': FONT_SIZE})

        # ===== Subplot 1: Demand & Generation (no peak shading) =====
        ax1 = axes[0]

        # Stack demand components - EV on TOP (last in stack)
        # Order: bottom to top in visual stack
        demand_components = [
            data_dict['hvac_power_kw'],
            data_dict['lighting'],
            data_dict['cooking'],
            data_dict['pc'],
            data_dict['tv'],
            data_dict['ev_charging']  # EV on top (plotted last = visually on top)
        ]
        demand_labels = ['HVAC', 'Lighting', 'Cooking', 'PC', 'TV', 'EV Charging']
        demand_colors = [self.colors['hvac'], self.colors['lighting'],
                         self.colors['cooking'], self.colors['pc'],
                         self.colors['tv'], self.colors['ev']]

        # Use baseline=0 to ensure proper stacking from zero
        ax1.stackplot(time_hours, demand_components,
                      labels=demand_labels, colors=demand_colors, alpha=0.8, baseline='zero')

        # Total demand line
        ax1.plot(time_hours, data_dict['total_demand'],
                 color='#2C3E50', linewidth=1, label='Total Demand', linestyle='-')

        # PV generation (negative, yellow shading below zero)
        pv_neg = -data_dict['pv_generation']
        ax1.fill_between(time_hours, 0, pv_neg,
                         color=self.colors['pv'], alpha=0.6, label='PV Generation')
        ax1.plot(time_hours, pv_neg, color='#E67E22', linewidth=0.8)

        ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax1.set_ylabel('Power (kW)', fontsize=FONT_SIZE)
        ax1.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, ncol=4, frameon=False,
                   handlelength=1, handletextpad=0.3, columnspacing=0.5)
        ax1.tick_params(axis='both', labelsize=FONT_SIZE)
        ax1.set_xticklabels([])

        # ===== Subplot 2: Battery SOC Only =====
        ax2 = axes[1]

        # Add peak shading FIRST
        if peak_mask.any():
            ax2.fill_between(time_hours, 0, 100, where=peak_mask,
                             alpha=0.15, color=self.colors['peak'],
                             label='Peak Period', zorder=0)

        ax2.plot(time_hours, data_dict['battery_soc'] * 100,
                 color=self.colors['battery'], linewidth=1.2, label='Battery SOC')

        ax2.set_ylabel('Battery SOC (%)', fontsize=FONT_SIZE)
        ax2.set_ylim([0, 100])
        ax2.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, frameon=False)
        ax2.tick_params(axis='both', labelsize=FONT_SIZE)
        ax2.set_xticklabels([])

        # ===== Subplot 3: Net Grid Load =====
        ax3 = axes[2]

        net_grid = data_dict['net_grid']
        curtailment = data_dict['curtailment']

        # Add peak shading FIRST
        if peak_mask.any():
            y_max_shade = max(np.max(net_grid) * 1.2, 1)
            y_min_shade = min(np.min(-curtailment) * 1.1, 0) if np.any(curtailment > 0) else 0
            ax3.fill_between(time_hours, y_min_shade, y_max_shade, where=peak_mask,
                             alpha=0.15, color=self.colors['peak'],
                             label='Peak Period', zorder=0)

        # Net grid import (positive values)
        ax3.fill_between(time_hours, 0, net_grid,
                         color=self.colors['grid'], alpha=0.6, label='Grid Import')
        ax3.plot(time_hours, net_grid, color='#1A252F', linewidth=1)

        # Curtailed PV (show as negative, below zero line)
        if np.any(curtailment > 0):
            ax3.fill_between(time_hours, 0, -curtailment,
                             color=self.colors['curtail'], alpha=0.5, label='Curtailed PV')
            ax3.plot(time_hours, -curtailment, color='#7F8C8D', linewidth=0.8)

        ax3.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax3.set_ylabel('Power (kW)', fontsize=FONT_SIZE)
        ax3.set_xlabel('Time (hours)', fontsize=FONT_SIZE)
        ax3.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, frameon=False)
        ax3.tick_params(axis='both', labelsize=FONT_SIZE)

        # ===== Synchronize Y-axis tick steps for ax1 and ax3 =====
        # Get data ranges
        ax1_min = min(np.min(pv_neg), 0) * 1.1
        ax1_max = np.max(data_dict['total_demand']) * 1.1
        ax3_min = min(-np.max(curtailment), 0) * 1.1 if np.any(curtailment > 0) else 0
        ax3_max = np.max(net_grid) * 1.2 if np.max(net_grid) > 0 else 1

        # Determine common tick step based on larger range
        range1 = ax1_max - ax1_min
        range3 = ax3_max - ax3_min
        max_range = max(range1, range3)

        # Choose appropriate step size
        if max_range <= 10:
            tick_step = 2
        elif max_range <= 20:
            tick_step = 4
        elif max_range <= 30:
            tick_step = 5
        else:
            tick_step = 10

        # Set ticks for ax1
        ax1_tick_min = np.floor(ax1_min / tick_step) * tick_step
        ax1_tick_max = np.ceil(ax1_max / tick_step) * tick_step
        ax1.set_ylim([ax1_tick_min, ax1_tick_max])
        ax1.set_yticks(np.arange(ax1_tick_min, ax1_tick_max + tick_step, tick_step))

        # Set ticks for ax3
        ax3_tick_min = np.floor(ax3_min / tick_step) * tick_step
        ax3_tick_max = np.ceil(ax3_max / tick_step) * tick_step
        ax3.set_ylim([ax3_tick_min, ax3_tick_max])
        ax3.set_yticks(np.arange(ax3_tick_min, ax3_tick_max + tick_step, tick_step))

        # Common styling for all axes
        for ax in axes:
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)

        plt.tight_layout(h_pad=0.5)

        if save_figure:
            fig_path = self.base_path / f"{simulation_name}_electrical.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            # Also save as PDF for journal submission
            pdf_path = self.base_path / f"{simulation_name}_electrical.pdf"
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {fig_path} and {pdf_path}")

        if show_figure:
            plt.show()

        return fig

    def plot_thermal_results(self,
                             simulation_name: str = None,
                             data_dict: Dict = None,
                             save_figure: bool = True,
                             show_figure: bool = True) -> plt.Figure:
        """Create thermal system plots matching electrical plot format."""
        if data_dict is None:
            data_dict, metadata = self.load_simulation_data(simulation_name)

        timesteps = data_dict['timesteps']
        time_hours = timesteps * 0.25
        peak_mask = data_dict['is_peak'].astype(bool)

        # Journal-quality settings (matching electrical plot)
        FONT_SIZE = 7
        LEGEND_FONT_SIZE = 5.5

        # Create figure with same width as electrical (3.5 inches), 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(3.5, 5.0), dpi=300, facecolor='white',
                                 gridspec_kw={'height_ratios': [1, 1, 1, 1]})

        plt.rcParams.update({'font.size': FONT_SIZE})

        # ===== Subplot 1: Zone Temperature =====
        ax1 = axes[0]
        if peak_mask.any():
            y_min = min(np.min(data_dict['heating_setpoint']), np.min(data_dict['zone_temperature'])) - 1
            y_max = max(np.max(data_dict['cooling_setpoint']), np.max(data_dict['zone_temperature'])) + 1
            ax1.fill_between(time_hours, y_min, y_max, where=peak_mask,
                             alpha=0.15, color=self.colors['peak'], label='Peak Period', zorder=0)

        ax1.plot(time_hours, data_dict['zone_temperature'],
                 color='#2C3E50', linewidth=1.2, label='Zone Temp')
        ax1.plot(time_hours, data_dict['cooling_setpoint'], '--',
                 color='#7F8C8D', linewidth=1, label='Cooling SP', alpha=0.8)
        ax1.plot(time_hours, data_dict['heating_setpoint'], '--',
                 color='#7F8C8D', linewidth=1, label='Heating SP', alpha=0.8)
        ax1.set_ylabel('Temperature (°C)', fontsize=FONT_SIZE)
        ax1.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, ncol=4, frameon=False,
                   handlelength=1, handletextpad=0.3, columnspacing=0.5)
        ax1.tick_params(axis='both', labelsize=FONT_SIZE)
        ax1.set_xticklabels([])

        # ===== Subplot 2: HVAC Power =====
        ax2 = axes[1]
        if peak_mask.any():
            y_max = max(np.max(data_dict['hvac_thermal_load']), np.max(data_dict['hvac_power'])) / 1000 * 1.1
            ax2.fill_between(time_hours, 0, y_max, where=peak_mask,
                             alpha=0.15, color=self.colors['peak'], zorder=0)

        ax2.plot(time_hours, data_dict['hvac_thermal_load'] / 1000,
                 color='#E74C3C', linewidth=1.2, label='Thermal Load')
        ax2.plot(time_hours, data_dict['hvac_power'] / 1000,
                 color='#3498DB', linewidth=1.2, label='Elec Power', linestyle='-.')
        ax2.set_ylabel('Power (kW)', fontsize=FONT_SIZE)
        ax2.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, ncol=2, frameon=False,
                   handlelength=1, handletextpad=0.3, columnspacing=0.5)
        ax2.tick_params(axis='both', labelsize=FONT_SIZE)
        ax2.set_xticklabels([])

        # ===== Subplot 3: Supply Air Temperature =====
        ax3 = axes[2]
        if peak_mask.any():
            y_min = min(np.min(data_dict['supply_air_temp_setpoint']), np.min(data_dict['supply_air_temp_actual'])) - 1
            y_max = max(np.max(data_dict['supply_air_temp_setpoint']), np.max(data_dict['supply_air_temp_actual'])) + 1
            ax3.fill_between(time_hours, y_min, y_max, where=peak_mask,
                             alpha=0.15, color=self.colors['peak'], zorder=0)

        ax3.plot(time_hours, data_dict['supply_air_temp_setpoint'], '--',
                 color='#7F8C8D', linewidth=1, label='Setpoint', alpha=0.8)
        ax3.plot(time_hours, data_dict['supply_air_temp_actual'],
                 color='#16A085', linewidth=1.2, label='Actual')
        ax3.set_ylabel('SAT (°C)', fontsize=FONT_SIZE)
        ax3.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, ncol=2, frameon=False,
                   handlelength=1, handletextpad=0.3, columnspacing=0.5)
        ax3.tick_params(axis='both', labelsize=FONT_SIZE)
        ax3.set_xticklabels([])

        # ===== Subplot 4: Supply Air Flow =====
        ax4 = axes[3]
        if peak_mask.any():
            y_max = max(np.max(data_dict['supply_air_flow_setpoint']),
                        np.max(data_dict['supply_air_flow_actual'])) * 1000 * 1.1
            ax4.fill_between(time_hours, 0, y_max, where=peak_mask,
                             alpha=0.15, color=self.colors['peak'], zorder=0)

        ax4.plot(time_hours, data_dict['supply_air_flow_setpoint'] * 1000, '--',
                 color='#7F8C8D', linewidth=1, label='Setpoint', alpha=0.8)
        ax4.plot(time_hours, data_dict['supply_air_flow_actual'] * 1000,
                 color='#8E44AD', linewidth=1.2, label='Actual')
        ax4.set_ylabel('SAF (L/s)', fontsize=FONT_SIZE)
        ax4.set_xlabel('Time (hours)', fontsize=FONT_SIZE)
        ax4.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, ncol=2, frameon=False,
                   handlelength=1, handletextpad=0.3, columnspacing=0.5)
        ax4.tick_params(axis='both', labelsize=FONT_SIZE)

        # Common styling for all axes
        for ax in axes:
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)

        plt.tight_layout(h_pad=0.5)

        if save_figure:
            fig_path = self.base_path / f"{simulation_name}_thermal.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            pdf_path = self.base_path / f"{simulation_name}_thermal.pdf"
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {fig_path} and {pdf_path}")

        if show_figure:
            plt.show()

        return fig

    def plot_comparison_thermal(self,
                                data_dict1: Dict,
                                data_dict2: Dict,
                                label1: str = "Simulation 1",
                                label2: str = "Simulation 2",
                                save_name: str = "comparison_thermal",
                                save_figure: bool = True,
                                show_figure: bool = True) -> plt.Figure:
        """Compare two simulations' thermal results."""

        timesteps = data_dict1['timesteps']
        time_hours = timesteps * 0.25
        peak_mask = data_dict1['is_peak'].astype(bool)

        FONT_SIZE = 7
        LEGEND_FONT_SIZE = 5.5

        # Colors for two simulations
        color1 = '#2C3E50'  # Dark blue-gray for sim1
        color2 = '#E74C3C'  # Red for sim2

        fig, axes = plt.subplots(4, 1, figsize=(3.5, 5.0), dpi=300, facecolor='white',
                                 gridspec_kw={'height_ratios': [1, 1, 1, 1]})
        plt.rcParams.update({'font.size': FONT_SIZE})

        # ===== Subplot 1: Zone Temperature =====
        ax1 = axes[0]
        if peak_mask.any():
            y_min = min(np.min(data_dict1['heating_setpoint']), np.min(data_dict2['heating_setpoint']),
                        np.min(data_dict1['zone_temperature']), np.min(data_dict2['zone_temperature'])) - 1
            y_max = max(np.max(data_dict1['cooling_setpoint']), np.max(data_dict2['cooling_setpoint']),
                        np.max(data_dict1['zone_temperature']), np.max(data_dict2['zone_temperature'])) + 1
            ax1.fill_between(time_hours, y_min, y_max, where=peak_mask,
                             alpha=0.15, color=self.colors['peak'], label='Peak', zorder=0)

        # Setpoints (gray, assume same for both)
        ax1.plot(time_hours, data_dict1['cooling_setpoint'], '--',
                 color='#7F8C8D', linewidth=1, label='Setpoints', alpha=0.8)
        ax1.plot(time_hours, data_dict1['heating_setpoint'], '--',
                 color='#7F8C8D', linewidth=1, alpha=0.8)
        # Zone temps
        ax1.plot(time_hours, data_dict1['zone_temperature'],
                 color=color1, linewidth=1.2, label=label1)
        ax1.plot(time_hours, data_dict2['zone_temperature'],
                 color=color2, linewidth=1.2, label=label2)
        ax1.set_ylabel('Temperature (°C)', fontsize=FONT_SIZE)
        ax1.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, ncol=4, frameon=False,
                   handlelength=1, handletextpad=0.3, columnspacing=0.5)
        ax1.tick_params(axis='both', labelsize=FONT_SIZE)
        ax1.set_xticklabels([])

        # ===== Subplot 2: HVAC Power =====
        ax2 = axes[1]
        if peak_mask.any():
            y_max = max(np.max(data_dict1['hvac_power']), np.max(data_dict2['hvac_power'])) / 1000 * 1.1
            ax2.fill_between(time_hours, 0, y_max, where=peak_mask,
                             alpha=0.15, color=self.colors['peak'], zorder=0)

        ax2.plot(time_hours, data_dict1['hvac_power'] / 1000,
                 color=color1, linewidth=1.2, label=label1)
        ax2.plot(time_hours, data_dict2['hvac_power'] / 1000,
                 color=color2, linewidth=1.2, label=label2)
        ax2.set_ylabel('HVAC Power (kW)', fontsize=FONT_SIZE)
        ax2.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, ncol=2, frameon=False,
                   handlelength=1, handletextpad=0.3, columnspacing=0.5)
        ax2.tick_params(axis='both', labelsize=FONT_SIZE)
        ax2.set_xticklabels([])

        # ===== Subplot 3: Supply Air Temperature =====
        ax3 = axes[2]
        if peak_mask.any():
            y_min = min(np.min(data_dict1['supply_air_temp_actual']),
                        np.min(data_dict2['supply_air_temp_actual'])) - 1
            y_max = max(np.max(data_dict1['supply_air_temp_actual']),
                        np.max(data_dict2['supply_air_temp_actual'])) + 1
            ax3.fill_between(time_hours, y_min, y_max, where=peak_mask,
                             alpha=0.15, color=self.colors['peak'], zorder=0)

        ax3.plot(time_hours, data_dict1['supply_air_temp_setpoint'], '--',
                 color='#7F8C8D', linewidth=1, label='Setpoint', alpha=0.8)
        ax3.plot(time_hours, data_dict1['supply_air_temp_actual'],
                 color=color1, linewidth=1.2, label=label1)
        ax3.plot(time_hours, data_dict2['supply_air_temp_actual'],
                 color=color2, linewidth=1.2, label=label2)
        ax3.set_ylabel('SAT (°C)', fontsize=FONT_SIZE)
        ax3.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, ncol=3, frameon=False,
                   handlelength=1, handletextpad=0.3, columnspacing=0.5)
        ax3.tick_params(axis='both', labelsize=FONT_SIZE)
        ax3.set_xticklabels([])

        # ===== Subplot 4: Supply Air Flow =====
        ax4 = axes[3]
        if peak_mask.any():
            y_max = max(np.max(data_dict1['supply_air_flow_actual']),
                        np.max(data_dict2['supply_air_flow_actual'])) * 1000 * 1.1
            ax4.fill_between(time_hours, 0, y_max, where=peak_mask,
                             alpha=0.15, color=self.colors['peak'], zorder=0)

        ax4.plot(time_hours, data_dict1['supply_air_flow_setpoint'] * 1000, '--',
                 color='#7F8C8D', linewidth=1, label='Setpoint', alpha=0.8)
        ax4.plot(time_hours, data_dict1['supply_air_flow_actual'] * 1000,
                 color=color1, linewidth=1.2, label=label1)
        ax4.plot(time_hours, data_dict2['supply_air_flow_actual'] * 1000,
                 color=color2, linewidth=1.2, label=label2)
        ax4.set_ylabel('SAF (L/s)', fontsize=FONT_SIZE)
        ax4.set_xlabel('Time (hours)', fontsize=FONT_SIZE)
        ax4.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, ncol=3, frameon=False,
                   handlelength=1, handletextpad=0.3, columnspacing=0.5)
        ax4.tick_params(axis='both', labelsize=FONT_SIZE)

        for ax in axes:
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)

        plt.tight_layout(h_pad=0.5)

        if save_figure:
            fig_path = self.base_path / f"{save_name}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            pdf_path = self.base_path / f"{save_name}.pdf"
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Comparison thermal plot saved to: {fig_path}")

        if show_figure:
            plt.show()

        return fig

    def plot_comparison_electrical(self,
                                   data_dict1: Dict,
                                   data_dict2: Dict,
                                   label1: str = "Simulation 1",
                                   label2: str = "Simulation 2",
                                   save_name: str = "comparison_electrical",
                                   save_figure: bool = True,
                                   show_figure: bool = True) -> plt.Figure:
        """
        Compare two simulations' electrical results.

        3 subplots:
        1. Demand breakdown (stacked) - keep only one (sim1), as disaggregate is complex
        2. Battery SOC - both simulations
        3. Net grid load - both simulations
        """
        timesteps = data_dict1['timesteps']
        time_hours = timesteps * 0.25
        peak_mask = data_dict1['is_peak'].astype(bool)

        FONT_SIZE = 7
        LEGEND_FONT_SIZE = 5.5

        color1 = '#2C3E50'
        color2 = '#E74C3C'

        fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.0), dpi=300, facecolor='white',
                                 gridspec_kw={'height_ratios': [2, 1, 2]})
        plt.rcParams.update({'font.size': FONT_SIZE})

        # ===== Subplot 1: Demand & Generation (from sim1 only) =====
        ax1 = axes[0]

        demand_components = [
            data_dict1['hvac_power_kw'],
            data_dict1['lighting'],
            data_dict1['cooking'],
            data_dict1['pc'],
            data_dict1['tv'],
            data_dict1['ev_charging']
        ]
        demand_labels = ['HVAC', 'Lighting', 'Cooking', 'PC', 'TV', 'EV']
        demand_colors = [self.colors['hvac'], self.colors['lighting'],
                         self.colors['cooking'], self.colors['pc'],
                         self.colors['tv'], self.colors['ev']]

        ax1.stackplot(time_hours, demand_components,
                      labels=demand_labels, colors=demand_colors, alpha=0.8, baseline='zero')

        ax1.plot(time_hours, data_dict1['total_demand'],
                 color='#2C3E50', linewidth=1, label='Total Demand', linestyle='-')

        pv_neg = -data_dict1['pv_generation']
        ax1.fill_between(time_hours, 0, pv_neg,
                         color=self.colors['pv'], alpha=0.6, label='PV')
        ax1.plot(time_hours, pv_neg, color='#E67E22', linewidth=0.8)

        ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax1.set_ylabel('Power (kW)', fontsize=FONT_SIZE)
        ax1.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, ncol=4, frameon=False,
                   handlelength=1, handletextpad=0.3, columnspacing=0.5)
        ax1.tick_params(axis='both', labelsize=FONT_SIZE)
        ax1.set_xticklabels([])

        # ===== Subplot 2: Battery SOC Comparison =====
        ax2 = axes[1]

        if peak_mask.any():
            ax2.fill_between(time_hours, 0, 100, where=peak_mask,
                             alpha=0.15, color=self.colors['peak'],
                             label='Peak', zorder=0)

        ax2.plot(time_hours, data_dict1['battery_soc'] * 100,
                 color=color1, linewidth=1.2, label=label1)
        ax2.plot(time_hours, data_dict2['battery_soc'] * 100,
                 color=color2, linewidth=1.2, label=label2)

        ax2.set_ylabel('Battery SOC (%)', fontsize=FONT_SIZE)
        ax2.set_ylim([0, 100])
        ax2.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, ncol=3, frameon=False,
                   handlelength=1, handletextpad=0.3, columnspacing=0.5)
        ax2.tick_params(axis='both', labelsize=FONT_SIZE)
        ax2.set_xticklabels([])

        # ===== Subplot 3: Net Grid Load Comparison =====
        ax3 = axes[2]

        net_grid1 = data_dict1['net_grid']
        net_grid2 = data_dict2['net_grid']
        curtailment1 = data_dict1['curtailment']
        curtailment2 = data_dict2['curtailment']

        if peak_mask.any():
            y_max = max(np.max(net_grid1), np.max(net_grid2)) * 1.2
            y_min = min(-np.max(curtailment1), -np.max(curtailment2), 0) * 1.1
            ax3.fill_between(time_hours, y_min, y_max, where=peak_mask,
                             alpha=0.15, color=self.colors['peak'],
                             label='Peak', zorder=0)

        # Grid import lines
        ax3.plot(time_hours, net_grid1, color=color1, linewidth=1.2, label=f'{label1} Grid')
        ax3.plot(time_hours, net_grid2, color=color2, linewidth=1.2, label=f'{label2} Grid')

        # Curtailment (dashed lines below zero)
        if np.any(curtailment1 > 0) or np.any(curtailment2 > 0):
            ax3.plot(time_hours, -curtailment1, color=color1, linewidth=0.8,
                     linestyle='--', alpha=0.7, label=f'{label1} Curtail')
            ax3.plot(time_hours, -curtailment2, color=color2, linewidth=0.8,
                     linestyle='--', alpha=0.7, label=f'{label2} Curtail')

        ax3.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
        ax3.set_ylabel('Power (kW)', fontsize=FONT_SIZE)
        ax3.set_xlabel('Time (hours)', fontsize=FONT_SIZE)
        ax3.legend(loc='upper center', fontsize=LEGEND_FONT_SIZE, ncol=3, frameon=False,
                   handlelength=1, handletextpad=0.3, columnspacing=0.5)
        ax3.tick_params(axis='both', labelsize=FONT_SIZE)

        for ax in axes:
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)

        plt.tight_layout(h_pad=0.5)

        if save_figure:
            fig_path = self.base_path / f"{save_name}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            pdf_path = self.base_path / f"{save_name}.pdf"
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Comparison electrical plot saved to: {fig_path}")

        if show_figure:
            plt.show()

        return fig

    def compare_simulations(self,
                            data_dict1: Dict,
                            data_dict2: Dict,
                            label1: str = "Baseline",
                            label2: str = "Optimized") -> Dict:
        """
        Compare two simulations and return KPI comparison table.

        KPIs: Flexibility, Energy, Cost, Comfort metrics
        """
        time_step_hours = 0.25

        def calc_kpis(data_dict):
            is_peak = data_dict['is_peak'].astype(bool)
            net_grid = data_dict['net_grid']
            pv_gen = data_dict['pv_generation']
            total_demand = data_dict['total_demand']
            zone_temp = data_dict['zone_temperature']
            cool_sp = data_dict['cooling_setpoint']
            heat_sp = data_dict['heating_setpoint']
            bat_soc = data_dict['battery_soc']
            bat_power = data_dict['battery_power']
            elec_price = data_dict['electricity_price']
            curtailment = data_dict['curtailment']

            # Comfort KPIs
            violations_above = np.sum(zone_temp > cool_sp)
            violations_below = np.sum(zone_temp < heat_sp)
            violation_rate = (violations_above + violations_below) / len(zone_temp) * 100
            temp_deviation = np.mean(np.maximum(0, zone_temp - cool_sp) +
                                     np.maximum(0, heat_sp - zone_temp))

            # Energy KPIs
            total_demand_kwh = np.sum(total_demand) * time_step_hours
            pv_generation_kwh = np.sum(pv_gen) * time_step_hours
            grid_import_kwh = np.sum(net_grid) * time_step_hours
            curtailment_kwh = np.sum(curtailment) * time_step_hours
            peak_grid_kwh = np.sum(net_grid[is_peak]) * time_step_hours
            offpeak_grid_kwh = np.sum(net_grid[~is_peak]) * time_step_hours
            self_consumption = (
                                           pv_generation_kwh - curtailment_kwh) / pv_generation_kwh * 100 if pv_generation_kwh > 0 else 0

            # Flexibility KPIs
            peak_reduction = np.max(net_grid[is_peak]) if is_peak.any() else 0
            load_factor = np.mean(net_grid) / np.max(net_grid) if np.max(net_grid) > 0 else 0
            peak_to_offpeak_ratio = peak_grid_kwh / offpeak_grid_kwh if offpeak_grid_kwh > 0 else float('inf')

            # Battery utilization
            cycles_efc = np.sum(np.abs(np.diff(bat_soc))) / 2.0

            # Cost KPIs
            total_cost = np.sum(net_grid * elec_price * time_step_hours) / 100
            peak_cost = np.sum(net_grid[is_peak] * elec_price[is_peak] * time_step_hours) / 100

            return {
                # Comfort
                'Violation Rate (%)': violation_rate,
                'Avg Temp Deviation (°C)': temp_deviation,
                'Comfort Violations (steps)': violations_above + violations_below,

                # Energy
                'Total Demand (kWh)': total_demand_kwh,
                'PV Generation (kWh)': pv_generation_kwh,
                'Grid Import (kWh)': grid_import_kwh,
                'Curtailment (kWh)': curtailment_kwh,
                'Self-Consumption (%)': self_consumption,
                'Peak Grid Import (kWh)': peak_grid_kwh,
                'Off-Peak Grid Import (kWh)': offpeak_grid_kwh,

                # Flexibility
                'Peak Power (kW)': peak_reduction,
                'Load Factor': load_factor,
                'Peak/Off-Peak Ratio': peak_to_offpeak_ratio,
                'Battery Cycles (EFC)': cycles_efc,

                # Cost
                'Total Cost ($)': total_cost,
                'Peak Cost ($)': peak_cost,
            }

        kpis1 = calc_kpis(data_dict1)
        kpis2 = calc_kpis(data_dict2)

        # Build comparison table
        comparison = {}
        for key in kpis1:
            val1 = kpis1[key]
            val2 = kpis2[key]
            if val1 != 0:
                change_pct = (val2 - val1) / abs(val1) * 100
            else:
                change_pct = 0 if val2 == 0 else float('inf')

            comparison[key] = {
                label1: val1,
                label2: val2,
                'Change (%)': change_pct
            }

        # Print comparison table
        print("\n" + "=" * 70)
        print(f"{'KPI COMPARISON':^70}")
        print("=" * 70)
        print(f"{'Metric':<30} {label1:>12} {label2:>12} {'Change':>12}")
        print("-" * 70)

        categories = {
            'COMFORT': ['Violation Rate (%)', 'Avg Temp Deviation (°C)', 'Comfort Violations (steps)'],
            'ENERGY': ['Total Demand (kWh)', 'PV Generation (kWh)', 'Grid Import (kWh)',
                       'Curtailment (kWh)', 'Self-Consumption (%)', 'Peak Grid Import (kWh)'],
            'FLEXIBILITY': ['Peak Power (kW)', 'Load Factor', 'Peak/Off-Peak Ratio', 'Battery Cycles (EFC)'],
            'COST': ['Total Cost ($)', 'Peak Cost ($)']
        }

        for category, metrics in categories.items():
            print(f"\n{category}:")
            for metric in metrics:
                if metric in comparison:
                    v1 = comparison[metric][label1]
                    v2 = comparison[metric][label2]
                    chg = comparison[metric]['Change (%)']
                    print(f"  {metric:<28} {v1:>12.2f} {v2:>12.2f} {chg:>+11.1f}%")

        print("=" * 70)

        return comparison

    def load_simulation_data(self, simulation_name: str) -> Tuple[Dict, Dict]:
        """Load saved simulation data."""
        load_path = self.base_path / 'operation' / f"{simulation_name}.pkl"

        if not load_path.exists():
            load_path = self.base_path / simulation_name
            if not load_path.exists():
                raise FileNotFoundError(f"Simulation data not found: {simulation_name}")

        with open(load_path, 'rb') as f:
            saved_data = pickle.load(f)

        return saved_data['data'], saved_data.get('metadata', {})

    def analyze_simulation(self,
                           simulation_name: str = None,
                           data_dict: Dict = None) -> Dict:
        """Analyze simulation with simplified power flow model."""
        if data_dict is None:
            data_dict, metadata = self.load_simulation_data(simulation_name)

        analysis = {}
        time_step_hours = 0.25

        # Comfort Analysis
        zone_temp = data_dict['zone_temperature']
        cool_sp = data_dict['cooling_setpoint']
        heat_sp = data_dict['heating_setpoint']

        comfort = {
            'temp_violations_above': np.sum(zone_temp > cool_sp),
            'temp_violations_below': np.sum(zone_temp < heat_sp),
            'violation_rate': (np.sum(zone_temp > cool_sp) + np.sum(zone_temp < heat_sp)) / len(zone_temp) * 100,
            'avg_temperature': np.mean(zone_temp),
            'temp_std': np.std(zone_temp)
        }
        analysis['comfort'] = comfort

        # Energy Analysis
        is_peak = data_dict['is_peak'].astype(bool)
        net_grid = data_dict['net_grid']
        pv_gen = data_dict['pv_generation']
        total_demand = data_dict['total_demand']

        energy = {
            'total_demand_kwh': np.sum(total_demand) * time_step_hours,
            'pv_generation_kwh': np.sum(pv_gen) * time_step_hours,
            'grid_import_kwh': np.sum(net_grid) * time_step_hours,
            'curtailment_kwh': np.sum(data_dict['curtailment']) * time_step_hours,
            'peak_grid_import_kwh': np.sum(net_grid[is_peak]) * time_step_hours,
            'offpeak_grid_import_kwh': np.sum(net_grid[~is_peak]) * time_step_hours,
            'self_consumption_rate': (np.sum(pv_gen) - np.sum(data_dict['curtailment'])) / np.sum(
                pv_gen) * 100 if np.sum(pv_gen) > 0 else 0,
            'grid_dependency': np.sum(net_grid) / np.sum(total_demand) * 100 if np.sum(total_demand) > 0 else 0
        }
        analysis['energy'] = energy

        # Battery Analysis
        bat_soc = data_dict['battery_soc']
        bat_power = data_dict['battery_power']

        def calc_efc(soc):
            return np.sum(np.abs(np.diff(soc))) / 2.0

        battery = {
            'avg_soc': np.mean(bat_soc) * 100,
            'min_soc': np.min(bat_soc) * 100,
            'max_soc': np.max(bat_soc) * 100,
            'cycles_efc': calc_efc(bat_soc),
            'total_charged_kwh': np.sum(bat_power[bat_power > 0]) * time_step_hours,
            'total_discharged_kwh': -np.sum(bat_power[bat_power < 0]) * time_step_hours
        }
        analysis['battery'] = battery

        # Cost Analysis
        elec_price = data_dict['electricity_price']
        cost = {
            'total_cost': np.sum(net_grid * elec_price * time_step_hours) / 100,
            'peak_cost': np.sum(net_grid[is_peak] * elec_price[is_peak] * time_step_hours) / 100,
            'offpeak_cost': np.sum(net_grid[~is_peak] * elec_price[~is_peak] * time_step_hours) / 100
        }
        analysis['cost'] = cost

        # Print summary
        print("\n" + "=" * 50)
        print("SIMULATION ANALYSIS")
        print("=" * 50)
        print(f"\n Comfort: {comfort['violation_rate']:.1f}% violations")
        print(f" Energy: {energy['grid_import_kwh']:.1f} kWh from grid, "
              f"{energy['self_consumption_rate']:.1f}% PV self-consumed")
        print(f"� Battery: {battery['cycles_efc']:.2f} EFC cycles")
        print(f" Cost: ${cost['total_cost']:.2f}")

        return analysis


# Example usage
# Example usage - Fixed version
if __name__ == "__main__":
    def set_seed(seed):
        import torch
        import random
        import numpy as np
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


    seed_value = 142857
    set_seed(seed_value)

    for level in ['no']:  # 0, 1, 2, 3
        # Initialize the simulation manager
        sim_manager = SimulationManager(base_path="./simulation_results")

        from bestopt.env.core.config_manager import ConfigurationManager
        from bestopt.env.core.environment import BESTOptEnvironment
        import os
        from pathlib import Path

        PROJECT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(PROJECT_ROOT_PATH))

        # Load configuration
        config_path = os.path.join(PROJECT_ROOT_PATH, "examples", "SFH_1_Building", f"precool_agentic.json")
        cm = ConfigurationManager(config_path)
        env = BESTOptEnvironment(cm.config)

        # Define simulation name
        # simulation_name = f"llm_level_{level}"
        simulation_name = f"precool_agentic"

        # Run simulation and save - returns the data_dict and save_path
        data_dict, save_path = sim_manager.run_and_save_simulation(
            env=env,
            total_steps=96,
            simulation_name=simulation_name,  # Use the variable
            # config_info={"Running": f"llm_level_{level}"}
            config_info={"Running": f"precool_agentic"}
        )

        # Option 1: Plot using the returned data_dict directly (no need to load)
        thermal_fig = sim_manager.plot_thermal_results(
            simulation_name=simulation_name,
            data_dict=data_dict  # Pass data directly - faster, no file loading
        )
        electrical_fig = sim_manager.plot_electrical_results(
            simulation_name=simulation_name,
            data_dict=data_dict  # Pass data directly
        )

        # Option 2: Or plot by loading from file (use the same name you saved with)
        # thermal_fig = sim_manager.plot_thermal_results(simulation_name)
        # electrical_fig = sim_manager.plot_electrical_results(simulation_name)

        # Analyze simulation
        analysis = sim_manager.analyze_simulation(
            simulation_name=simulation_name,
            data_dict=data_dict
        )